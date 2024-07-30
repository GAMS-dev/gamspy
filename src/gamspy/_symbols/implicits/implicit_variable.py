from __future__ import annotations

from typing import TYPE_CHECKING

import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
import gamspy._symbols.implicits as implicits
import gamspy._validation as validation
import gamspy.utils as utils
from gamspy._symbols.implicits.implicit_symbol import ImplicitSymbol
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from gams.transfer import Set, Variable


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
        return f"ImplicitVariable(parent={self.parent}, name={self.name}, domain={self.domain}, permutation={self.permutation}, parent_scalar_domains={self.parent_scalar_domains})"

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
        >>> v_t_t.domain # doctest: +ELLIPSIS
        [<Set `i` (0x...)>, <Set `j` (0x...)>]

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
        return permute(self, dims)

    def __neg__(self):
        return expression.Expression(None, "-", self)

    def __eq__(self, other):  # type: ignore
        return expression.Expression(self, "=e=", other)

    def __ne__(self, other):  # type: ignore
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
