from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
import gamspy._algebra.operation as operation
import gamspy._symbols as syms
import gamspy._validation as validation
import gamspy.utils as utils
from gamspy._internals import DataSource
from gamspy._symbols.implicits.implicit_symbol import ImplicitSymbol
from gamspy._symbols.implicits.implicit_variable import ImplicitVariable
from gamspy._symbols.utils import assign
from gamspy._universe import UNIVERSE
from gamspy.math.matrix import _transpose

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import pandas as pd

    from gamspy import Alias, Equation, Parameter, Set, Variable
    from gamspy._algebra.expression import Expression
    from gamspy._algebra.operation import Card, Operation, Ord
    from gamspy._algebra.sparse import SparseAssignment
    from gamspy._notation import Notation
    from gamspy._symbols.implicits import ImplicitSet
    from gamspy._types import IndexType, OperableType, ValidatedIndexType
    from gamspy.math.misc import MathOp

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
    "range": "range",
    "slack": "slack",
    "slacklo": "slacklo",
    "slackup": "slackup",
    "infeas": "infeas",
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
        parent: Set | Alias | Parameter | Variable | Equation,
        name: str,
        domain: Sequence[ValidatedIndexType] | None = None,
        permutation: list[int] | None = None,
        scalar_domains: list[tuple[int, str]] | None = None,
    ) -> None:
        """Implicit Parameter

        Parameters
        ----------
        parent : Parameter | Variable | Equation
        name : str
        domain : Sequence[Set | Alias | ImplicitSet | Expression | str], optional
        records : Any, optional
        """
        self.parent = parent
        if domain is None:
            domain = []

        super().__init__(name, domain, permutation, scalar_domains)
        self._assignment = None

    def _math_name(self, notation: Notation) -> str:
        # Attributes such as x.l or i.pos are written with the notation of attributes: x^{l}
        name = self.parent._math_name(notation)
        _, separator, attribute = self.name.rpartition(".")
        if separator:
            name = notation.attribute(name, attribute)

        return name

    def __getitem__(self, indices: IndexType) -> ImplicitParameter:
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
        indices: IndexType,
        rhs: Expression
        | Operation
        | MathOp
        | ImplicitParameter
        | ImplicitSet
        | int
        | float
        | Parameter
        | Card
        | Ord
        | SparseAssignment,
    ) -> None:
        if (
            self.parent.container._in_loop == 0
            and isinstance(self.parent, (syms.Variable, syms.Equation))
            and len(self.parent.domain) > 0
            and all(
                isinstance(elem, (syms.Set, syms.Alias))
                and elem._should_load_from is DataSource.NONE
                and len(elem) == 0
                for elem in self.parent.domain
            )
        ):
            logger.warning(
                f"Domain was not initialized. Default values for {self.gamsRepr()} will be used."
            )
        # self[domain] = rhs
        domain = validation.validate_domain(self, indices)
        assign(
            self.parent,
            ImplicitParameter(
                parent=self.parent,
                name=self.name,
                domain=domain,
                permutation=self.permutation,
                scalar_domains=self._scalar_domains,
            ),
            rhs,
            domain,
        )

    def __eq__(self, other: OperableType) -> Expression:
        op = "eq"
        if isinstance(
            other,
            (ImplicitVariable, expression.Expression, operation.Operation),
        ):
            op = "=e="
        return expression.Expression(self, op, other)

    def __ne__(self, other: OperableType) -> Expression:
        return expression.Expression(self, "ne", other)

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return f"ImplicitParameter(parent={self.parent}, name='{self.name}', domain={self.domain}, permutation={self.permutation}), parent_scalar_domains={self.parent_scalar_domains})"

    @property
    def records(self) -> pd.DataFrame | None:
        if self.parent.records is None:
            return None

        if isinstance(self.parent, (syms.Set, syms.Alias)):
            with self.container._temp_symbol(
                syms.Parameter, [self.parent, UNIVERSE]
            ) as temp_param:
                column_name = SET_ATTR_MAPPING[self.name.split(".")[1]]
                temp_param[self.parent, column_name] = self
            return temp_param.records
        elif isinstance(self.parent, syms.Parameter):
            given_domain, declaration_domain = self._get_temp_domain()
            if given_domain == []:
                given_domain = [...]

            with self.container._temp_symbol(
                syms.Parameter, declaration_domain
            ) as temp_param:
                temp_param[given_domain] = self[...]

            return temp_param.records
        elif isinstance(self.parent, (syms.Variable, syms.Equation)):
            extension = self.name.split(".")[-1]
            given_domain, declaration_domain = self._get_temp_domain()
            if given_domain == []:
                given_domain = [...]

            with self.container._temp_symbol(
                syms.Parameter, declaration_domain
            ) as temp_param:
                temp_param[given_domain] = self

            recs = temp_param.records
            if recs is None:
                return recs

            extension = ATTR_MAPPING[extension]
            columns = recs.columns.to_list()
            columns[columns.index("value")] = extension
            recs.columns = columns

            return recs

        return None

    def toDense(self) -> np.ndarray:
        """
        Converts records to a dense numpy.array format.

        The shape of the returned array follows the domain of this implicit
        parameter: literal indices reduce the dimensionality and a
        transpose/permutation reorders the axes accordingly.

        Returns
        -------
        np.ndarray
            A numpy array with the records. An array of zeros if the parent
            symbol has no records.

        Examples
        --------
        >>> import numpy as np
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=["i1", "i2"])
        >>> j = gp.Set(m, "j", records=["j1", "j2", "j3"])
        >>> p = gp.Parameter(m, "p", domain=[i, j], records=np.array([[1, 2, 3], [4, 5, 6]]))
        >>> print(p.t().toDense())
        [[1. 4.]
         [2. 5.]
         [3. 6.]]
        >>> print(p[i, "j2"].toDense())
        [2. 5.]

        """
        if isinstance(self.parent, (syms.Set, syms.Alias)):
            # Set attributes (e.g. i.pos) are indexed by the parent set.
            domain: list = [self.parent]
        else:
            domain = list(self.domain)

        with self.container._temp_symbol(syms.Parameter, domain) as temp_param:
            temp_param[domain if domain else [...]] = self
            return temp_param.toDense()

    def toValue(self) -> float:
        """
        Returns the numerical value of a scalar (fully indexed) implicit
        parameter.

        Returns
        -------
        float

        Raises
        ------
        TypeError
            If the implicit parameter is not scalar (all indices must be
            literals).

        Examples
        --------
        >>> import numpy as np
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=["i1", "i2"])
        >>> j = gp.Set(m, "j", records=["j1", "j2", "j3"])
        >>> p = gp.Parameter(m, "p", domain=[i, j], records=np.array([[1, 2, 3], [4, 5, 6]]))
        >>> p["i1", "j2"].toValue()
        np.float64(2.0)

        """
        if isinstance(self.parent, (syms.Set, syms.Alias)):
            # Set attributes (e.g. i.pos) are indexed by the parent set.
            domain: list = [self.parent]
        else:
            domain = list(self.domain)

        with self.container._temp_symbol(syms.Parameter, domain) as temp_param:
            temp_param[domain if domain else [...]] = self
            return temp_param.toValue()

    def toList(self) -> list:
        """
        Converts the records of the implicit parameter to a Python list.

        Returns
        -------
        list
            For scalars, a list with a single numerical value. For
            multi-dimensional implicit parameters, a list of tuples where the
            last element of each tuple is the value. An empty list if the
            parent symbol has no records.

        Examples
        --------
        >>> import numpy as np
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=["i1", "i2"])
        >>> j = gp.Set(m, "j", records=["j1", "j2", "j3"])
        >>> p = gp.Parameter(m, "p", domain=[i, j], records=np.array([[1, 2, 3], [4, 5, 6]]))
        >>> p[i, "j2"].toList()
        [('i1', 2.0), ('i2', 5.0)]

        """
        if self.parent.records is None:
            return []

        if isinstance(self.parent, (syms.Set, syms.Alias)):
            # Set attributes (e.g. i.pos) are indexed by the parent set.
            domain: list = [self.parent]
        else:
            domain = list(self.domain)

        with self.container._temp_symbol(syms.Parameter, domain) as temp_param:
            temp_param[domain if domain else [...]] = self
            return temp_param.toList()

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
        return _transpose(self, "Parameter")  # ty: ignore[invalid-return-type]

    def gamsRepr(self) -> str:
        """Representation of the parameter in GAMS syntax.

        Returns:
            str: String representation of the parameter in GAMS syntax.
        """
        representation = self.name
        domain = self._reference_domain()
        if domain:
            representation += utils._get_domain_str(domain)

        return representation
