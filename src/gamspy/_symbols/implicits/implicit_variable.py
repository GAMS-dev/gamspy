from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
import gamspy._algebra.operation as operation
import gamspy._symbols.implicits as implicits
import gamspy._validation as validation
import gamspy.utils as utils
from gamspy._symbols.implicits.implicit_symbol import ImplicitSymbol
from gamspy._types import EllipsisType
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from gamspy import (
        Alias,
        Product,
        Sand,
        Set,
        Smax,
        Smin,
        Sor,
        Sum,
        Variable,
    )
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

    def __getitem__(self, indices: list | str) -> ImplicitVariable:
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
        if self.parent.type in ["integer", "binary"]:
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
        if self.parent.type not in ["integer", "binary"]:
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

    def sum(
        self, *indices: Sequence[Set | Alias | str | EllipsisType | slice]
    ) -> Sum:
        """
        Equivalent to Sum(indices, obj[obj.domain]). For example:

        v = Variable(m, domain=[i, j, k])
        v.sum() is equivalent to Sum((i,j), v[i, j, k])
        v.sum(i) is equivalent to Sum(i, v[i, j, k])
        v.sum(i, j) is equivalent to Sum((i, j), v[i, j, k])

        Returns
        -------
        Sum
            Generated Sum operation.

        Raises
        ------
        ValidationError
            In case the symbol is scalar.
        """
        if not self.domain:
            raise ValidationError(
                "Sum operation is not possible on scalar parameters."
            )

        if not indices:
            indices = self.domain

        return operation.Sum(indices, self[self.domain])

    def product(
        self, *indices: Sequence[Set | Alias | str | EllipsisType | slice]
    ) -> Product:
        """
        Equivalent to Product(indices, obj[obj.domain]). For example:

        v = Variable(m, domain=[i, j, k])
        v.product() is equivalent to Product((i,j), v[i, j, k])
        v.product(i) is equivalent to Product(i, v[i, j, k])
        v.product(i, j) is equivalent to Product((i, j), v[i, j, k])

        Returns
        -------
        Product
            Generated Product operation.

        Raises
        ------
        ValidationError
            In case the symbol is scalar.
        """
        if not self.domain:
            raise ValidationError(
                "Product operation is not possible on scalar parameters."
            )

        if not indices:
            indices = self.domain

        return operation.Product(indices, self[self.domain])

    def smin(
        self, *indices: Sequence[Set | Alias | str | EllipsisType | slice]
    ) -> Smin:
        """
        Equivalent to Smin(indices, obj[obj.domain]). For example:

        v = Variable(m, domain=[i, j, k])
        v.smin() is equivalent to Smin((i,j), v[i, j, k])
        v.smin(i) is equivalent to Smin(i, v[i, j, k])
        v.smin(i, j) is equivalent to Smin((i, j), v[i, j, k])

        Returns
        -------
        Smin
            Generated Smin operation.

        Raises
        ------
        ValidationError
            In case the symbol is scalar.
        """
        if not self.domain:
            raise ValidationError(
                "Smin operation is not possible on scalar parameters."
            )

        if not indices:
            indices = self.domain

        return operation.Smin(indices, self[self.domain])

    def smax(
        self, *indices: Sequence[Set | Alias | str | EllipsisType | slice]
    ) -> Smax:
        """
        Equivalent to Smax(indices, obj[obj.domain]). For example:

        v = Variable(m, domain=[i, j, k])
        v.smax() is equivalent to Smax((i,j), v[i, j, k])
        v.smax(i) is equivalent to Smax(i, v[i, j, k])
        v.smax(i, j) is equivalent to Smax((i, j), v[i, j, k])

        Returns
        -------
        Smax
            Generated Smax operation.

        Raises
        ------
        ValidationError
            In case the symbol is scalar.
        """
        if not self.domain:
            raise ValidationError(
                "Smax operation is not possible on scalar parameters."
            )

        if not indices:
            indices = self.domain

        return operation.Smax(indices, self[self.domain])

    def sand(
        self, *indices: Sequence[Set | Alias | str | EllipsisType | slice]
    ) -> Sand:
        """
        Equivalent to Sand(indices, obj[obj.domain]). For example:

        v = Variable(m, domain=[i, j, k])
        v.sand() is equivalent to Sand((i,j), v[i, j, k])
        v.sand(i) is equivalent to Sand(i, v[i, j, k])
        v.sand(i, j) is equivalent to Sand((i, j), v[i, j, k])

        Returns
        -------
        Sand
            Generated Sand operation.

        Raises
        ------
        ValidationError
            In case the symbol is scalar.
        """
        if not self.domain:
            raise ValidationError(
                "Sand operation is not possible on scalar parameters."
            )

        if not indices:
            indices = self.domain

        return operation.Sand(indices, self[self.domain])

    def sor(
        self, *indices: Sequence[Set | Alias | str | EllipsisType | slice]
    ) -> Sor:
        """
        Equivalent to Sor(indices, obj[obj.domain]). For example:

        v = Variable(m, domain=[i, j, k])
        v.sor() is equivalent to Sor((i,j), v[i, j, k])
        v.sor(i) is equivalent to Sor(i, v[i, j, k])
        v.sor(i, j) is equivalent to Sor((i, j), v[i, j, k])

        Returns
        -------
        Sor
            Generated Sor operation.

        Raises
        ------
        ValidationError
            In case the symbol is scalar.
        """
        if not self.domain:
            raise ValidationError(
                "Sor operation is not possible on scalar parameters."
            )

        if not indices:
            indices = self.domain

        return operation.Sor(indices, self[self.domain])

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
        >>> m = operation.Container()
        >>> i = operation.Set(m, "i", records=['i1','i2'])
        >>> j = operation.Set(m, "j", records=['j1','j2'])
        >>> v = operation.Variable(m, "v", domain=[i, j])
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

    def __neg__(self):
        return expression.Expression(None, "-", self)

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
