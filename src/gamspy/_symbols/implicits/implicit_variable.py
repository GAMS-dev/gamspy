from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
import gamspy._symbols.implicits as implicits
import gamspy._validation as validation
import gamspy.utils as utils
from gamspy._symbols.implicits.implicit_symbol import ImplicitSymbol
from gamspy._types import EllipsisType
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    import pandas as pd

    from gamspy import Set, Variable
    from gamspy._algebra.expression import Expression


class ImplicitVariable(ImplicitSymbol, operable.Operable):
    def __init__(
        self,
        parent: Variable,
        name: str,
        domain: list[Set | str],
        permutation: list[int] | None = None,
        scalar_domains: list[tuple[int | Set]] | None = None,
    ):
        """
        Implicit Variable

        Parameters
        ----------
        parent : Variable
        name : str
        domain : list[Set | str]
        """
        super().__init__(parent, name, domain, permutation, scalar_domains)

    def __repr__(self) -> str:
        return f"ImplicitVariable(parent={self.parent}, name='{self.name}', domain={self.domain}, permutation={self.permutation}, parent_scalar_domains={self.parent_scalar_domains})"

    def __getitem__(
        self, indices: Sequence | str | EllipsisType | slice
    ) -> ImplicitVariable:
        domain = validation.validate_domain(self, indices)
        return ImplicitVariable(
            parent=self.parent,
            name=self.name,
            domain=domain,
            permutation=self.permutation,
            scalar_domains=self._scalar_domains,
        )

    @property
    def l(self):  # noqa: E741,E743
        return implicits.ImplicitParameter(
            self.parent,
            name=self.parent.l.name,
            domain=self.domain,
            scalar_domains=self._scalar_domains,
        )

    @l.setter
    def l(self, value: int | float | Expression):
        # b[t].l = 30 -> b.l[t] = 30
        self.parent.l[self.domain] = value

    @property
    def m(self):
        return implicits.ImplicitParameter(
            self.parent,
            name=self.parent.m.name,
            domain=self.domain,
            scalar_domains=self._scalar_domains,
        )

    @m.setter
    def m(self, value: int | float | Expression):
        # b[t].m = 30 -> b.m[t] = 30
        self.parent.m[self.domain] = value

    @property
    def lo(self):
        return implicits.ImplicitParameter(
            self.parent,
            name=self.parent.lo.name,
            domain=self.domain,
            scalar_domains=self._scalar_domains,
        )

    @lo.setter
    def lo(self, value: int | float | Expression):
        # b[t].lo = 30 -> b.lo[t] = 30
        self.parent.lo[self.domain] = value

    @property
    def up(self):
        return implicits.ImplicitParameter(
            self.parent,
            name=self.parent.up.name,
            domain=self.domain,
            scalar_domains=self._scalar_domains,
        )

    @up.setter
    def up(self, value: int | float | Expression):
        # b[t].up = 30 -> b.up[t] = 30
        self.parent.up[self.domain] = value

    @property
    def scale(self):
        return implicits.ImplicitParameter(
            self.parent,
            name=self.parent.scale.name,
            domain=self.domain,
            scalar_domains=self._scalar_domains,
        )

    @scale.setter
    def scale(self, value: int | float | Expression):
        if self.parent.type in ("integer", "binary"):
            raise ValidationError(
                "Scales cannot be applied to discrete variables."
            )

        # b[t].scale = 30 -> b.scale[t] = 30
        self.parent.scale[self.domain] = value

    @property
    def fx(self):
        return implicits.ImplicitParameter(
            self.parent,
            name=self.parent.fx.name,
            domain=self.domain,
            scalar_domains=self._scalar_domains,
        )

    @fx.setter
    def fx(self, value: int | float | Expression):
        # b[t].fx = 30 -> b.fx[t] = 30
        self.parent.fx[self.domain] = value

    @property
    def prior(self):
        return implicits.ImplicitParameter(
            self.parent,
            name=self.parent.prior.name,
            domain=self.domain,
            scalar_domains=self._scalar_domains,
        )

    @prior.setter
    def prior(self, value: int | float | Expression):
        if self.parent.type not in ("integer", "binary"):
            raise ValidationError(
                "Priorities can only be used on discrete variables."
            )

        # b[t].prior = 30 -> b.prior[t] = 30
        self.parent.prior[self.domain] = value

    @property
    def stage(self):
        return implicits.ImplicitParameter(
            self.parent,
            name=self.parent.stage.name,
            domain=self.domain,
            scalar_domains=self._scalar_domains,
        )

    @stage.setter
    def stage(self, value: int | float | Expression):
        # b[t].stage = 30 -> b.stage[t] = 30
        self.parent.stage[self.domain] = value

    @property
    def T(self) -> implicits.ImplicitVariable:
        """See gamspy.ImplicitVariable.t"""
        return self.t()

    def t(self) -> implicits.ImplicitVariable:
        """Returns an ImplicitVariable derived from this
        implicit variable by swapping its last two indices.
        This operation does not generate a new variable in GAMS.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=['i1','i2'])
        >>> j = gp.Set(m, "j", records=['j1','j2'])
        >>> v = gp.Variable(m, "v", domain=[i, j])
        >>> v_t = v.t() # v_t is an ImplicitVariable
        >>> v_t_t = v_t.t() # you can get transpose of ImplicitVariable as well
        >>> v_t_t.domain
        [Set(name='i', domain=['*']), Set(name='j', domain=['*'])]

        """
        from gamspy.math.matrix import permute

        dims = [x for x in range(len(self.domain))]
        if len(dims) < 2:
            raise ValidationError(
                "Variable must contain at least 2 dimensions to transpose"
            )

        x = dims[-1]
        dims[-1] = dims[-2]
        dims[-2] = x
        return permute(self, dims)  # type: ignore

    @property
    def records(self) -> pd.DataFrame | float | None:
        if self.parent.records is None:
            return None

        recs = self.parent.records
        for idx, literal in self._scalar_domains:
            column_name = recs.columns[idx]
            recs = recs[recs[column_name] == literal]

        return recs

    def __eq__(self, other):
        return expression.Expression(self, "=e=", other)

    def __ne__(self, other):
        return expression.Expression(self, "ne", other)

    def gamsRepr(self) -> str:
        representation = self.name
        domain = list(self.domain)
        if domain and self.permutation is not None:
            domain = utils._permute_domain(domain, self.permutation)

        for i, d in self._scalar_domains:
            domain.insert(i, d)

        if domain:
            representation += utils._get_domain_str(domain)

        return representation
