from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
import gamspy._algebra.operation as operation
import gamspy._symbols as syms
import gamspy._validation as validation
import gamspy.utils as utils
from gamspy._symbols.implicits.implicit_symbol import ImplicitSymbol
from gamspy._symbols.implicits.implicit_variable import ImplicitVariable
from gamspy.exceptions import ValidationError
from gamspy.math.matrix import permute

if TYPE_CHECKING:
    import pandas as pd

    from gamspy import (
        Equation,
        Parameter,
        Set,
        Variable,
    )
    from gamspy._algebra.expression import Expression
    from gamspy._algebra.operation import Operation

logger = logging.getLogger("GAMSPy")
logger.setLevel(logging.WARNING)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.WARNING)
formatter = logging.Formatter("[%(name)s - %(levelname)s] %(message)s")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

ATTR_MAPPING = {
    "l": "level",
    "m": "marginal",
    "lo": "lower",
    "up": "upper",
    "scale": "scale",
}

SET_ATTR_MAPPING = {
    "pos": "position",
    "ord": "order",
    "off": "off",
    "rev": "reverse",
    "uel": "uel_position",
    "len": "length",
    "tlen": "text_length",
    "val": "value",
    "tval": "text_value",
    "first": "is_first",
    "last": "is_last",
}


class ImplicitParameter(ImplicitSymbol, operable.Operable):
    def __init__(
        self,
        parent: Parameter | Variable | Equation,
        name: str,
        domain: list[Set | str] = [],
        records: Any | None = None,
        permutation: list[int] | None = None,
        scalar_domains: list[tuple[int, Set]] | None = None,
    ) -> None:
        """Implicit Parameter

        Parameters
        ----------
        parent : Parameter | Variable | Equation
        name : str
        domain : list[Set | str], optional
        records : Any, optional
        """
        super().__init__(parent, name, domain, permutation, scalar_domains)
        self._records = records
        self._assignment = None

    def __getitem__(self, indices: Iterable | str) -> ImplicitParameter:
        domain = validation.validate_domain(self, indices)

        return ImplicitParameter(
            parent=self.parent,
            name=self.name,
            domain=domain,
            permutation=self.permutation,
            scalar_domains=self._scalar_domains,
        )

    def __setitem__(
        self,
        indices: Iterable | str,
        rhs: Expression | Operation | ImplicitParameter | int | float,
    ) -> None:
        if (
            isinstance(self.parent, (syms.Variable, syms.Equation))
            and len(self.parent.domain) > 0
            and all(len(elem) == 0 for elem in self.parent.domain)
        ):
            logger.warning(
                f"Domain was not initialized. Default values for {self.gamsRepr()} will be used."
            )
        # self[domain] = rhs
        domain = validation.validate_domain(self, indices)

        if isinstance(rhs, float):
            rhs = utils._map_special_values(rhs)  # type: ignore

        statement = expression.Expression(
            ImplicitParameter(
                parent=self.parent,
                name=self.name,
                domain=domain,
                permutation=self.permutation,
                scalar_domains=self._scalar_domains,
            ),
            "=",
            rhs,
        )

        statement._validate_definition(utils._unpack(domain))

        self.container._add_statement(statement)
        self.parent._assignment = statement

        self.container._synch_with_gams(gams_to_gamspy=True)
        self.parent._winner = "gams"

    def __eq__(self, other):
        op = "eq"
        if isinstance(
            other,
            (ImplicitVariable, expression.Expression, operation.Operation),
        ):
            op = "=e="
        return expression.Expression(self, op, other)

    def __ne__(self, other):
        return expression.Expression(self, "ne", other)

    def __repr__(self) -> str:
        return f"ImplicitParameter(parent={self.parent}, name='{self.name}', domain={self.domain}, permutation={self.permutation}), parent_scalar_domains={self.parent_scalar_domains})"

    def _add_new_attr_column(
        self, extension: str, recs: pd.DataFrame
    ) -> pd.DataFrame:
        parent_recs: pd.DataFrame = self.parent.records
        if extension == "range":
            recs[extension] = parent_recs["lower"] - parent_recs["upper"]
        elif extension == "slacklo":
            recs[extension] = (
                parent_recs["level"] - parent_recs["lower"]
            ).clip(0)
        elif extension == "slackup":
            recs[extension] = (
                parent_recs["upper"] - parent_recs["level"]
            ).clip(0)
        elif extension == "slack":
            slacklo = (parent_recs["level"] - parent_recs["lower"]).clip(0)
            slackup = (parent_recs["upper"] - parent_recs["level"]).clip(0)
            recs[extension] = slacklo.combine(slackup, min)
        elif extension == "infeas":
            distance_to_lower = parent_recs["lower"] - parent_recs["level"]
            distance_to_upper = parent_recs["level"] - parent_recs["upper"]
            max_distance = distance_to_lower.combine(distance_to_upper, max)
            recs[extension] = max_distance.clip(lower=0)

        return recs

    @property
    def records(self) -> pd.DataFrame | float | None:
        if self.parent.records is None:
            return None

        if isinstance(self.parent, (syms.Set, syms.Alias)):
            temp_name = "p" + utils._get_unique_name()
            temp_param = syms.Parameter._constructor_bypass(
                self.container, temp_name, [self.parent, "*"]
            )
            column_name = SET_ATTR_MAPPING[self.name.split(".")[1]]
            temp_param[self.parent, column_name] = self
            del self.container.data[temp_name]
            return temp_param.records
        elif isinstance(self.parent, syms.Parameter):
            recs = self.parent.records
            for idx, literal in self._scalar_domains:
                column_name = recs.columns[idx]
                recs = recs[recs[column_name] == literal]

            if all(
                isinstance(elem, str) and elem != "*" for elem in self.domain
            ):
                return float(recs["value"].squeeze())

            return recs
        elif isinstance(self.parent, (syms.Variable, syms.Equation)):
            extension = self.name.split(".")[-1]
            column_names: list[str] = self.parent.domain_labels

            if extension in ATTR_MAPPING:
                extension = ATTR_MAPPING[extension]
                column_names.append(extension)

            recs = self.parent.records[column_names].copy()
            if extension not in ATTR_MAPPING:
                # eq.range, eq.slacklo, eq.slackup, eq.slack, eq.infeas
                recs = self._add_new_attr_column(extension, recs)

            for idx, literal in self._scalar_domains:
                column_name = recs.columns[idx]
                recs = recs[recs[column_name] == literal]

            if all(
                isinstance(elem, str) and elem != "*" for elem in self.domain
            ):
                return float(recs[extension].squeeze())

            return recs

        return None

    @property
    def T(self) -> ImplicitParameter:
        """See gamspy.ImplicitParameter.t"""
        return self.t()

    def t(self) -> ImplicitParameter:
        """Returns an ImplicitParameter derived from this
        implicit parameter by swapping its last two indices.
        This operation does not generate a new parameter in GAMS.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=['i1','i2'])
        >>> j = gp.Set(m, "j", records=['j1','j2'])
        >>> v = gp.Parameter(m, "v", domain=[i, j])
        >>> v_t = v.t() # v_t is an ImplicitParameter
        >>> v_t_t = v_t.t() # you can get transpose of ImplicitParameter as well
        >>> v_t_t.domain
        [Set(name='i', domain=['*']), Set(name='j', domain=['*'])]

        """
        dims = [x for x in range(len(self.domain))]
        if len(dims) < 2:
            raise ValidationError(
                "Parameter must contain at least 2 dimensions to transpose"
            )

        x = dims[-1]
        dims[-1] = dims[-2]
        dims[-2] = x
        return permute(self, dims)  # type: ignore

    def gamsRepr(self) -> str:
        """Representation of the parameter in GAMS syntax.

        Returns:
            str: String representation of the parameter in GAMS syntax.
        """
        representation = self.name
        domain = list(self.domain)
        if domain and self.permutation is not None:
            domain = utils._permute_domain(domain, self.permutation)

        for i, d in self._scalar_domains:
            domain.insert(i, d)

        if domain:
            representation += utils._get_domain_str(domain)

        return representation
