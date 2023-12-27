from __future__ import annotations

from typing import TYPE_CHECKING

import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
import gamspy._symbols.implicits as implicits
import gamspy.utils as utils
from gamspy._symbols.implicits.implicit_symbol import ImplicitSymbol

if TYPE_CHECKING:
    from gams.transfer import Set, Variable


class ImplicitVariable(ImplicitSymbol, operable.Operable):
    def __init__(
        self,
        parent: Variable,
        name: str,
        domain: list[Set | str],
        permutation: List[int] | None = None,
    ):
        """
        Implicit Variable

        Parameters
        ----------
        parent : Variable
        name : str
        domain : List[Set | str]
        """
        super().__init__(parent, name, domain)
        self.permutation = permutation
        self._l, self._m, self._lo, self._up, self._s = self._init_attributes()
        self._fx = self._create_attr("fx")
        self._prior = self._create_attr("prior")
        self._stage = self._create_attr("stage")

    def _create_attr(self, attr_name: str):
        return implicits.ImplicitParameter(
            self.parent,
            name=f"{self.gamsRepr()}.{attr_name}",
            permutation=self.permutation,
        )

    def _init_attributes(self):
        level = self._create_attr("l")
        marginal = self._create_attr("m")
        lower = self._create_attr("lo")
        upper = self._create_attr("up")
        scale = self._create_attr("scale")
        return level, marginal, lower, upper, scale

    def __getitem__(self, indices: list | str) -> ImplicitVariable:
        domain = self.domain if indices == ... else utils._to_list(indices)
        return ImplicitVariable(
            parent=self.parent,
            name=self.name,
            domain=domain,
            permutation=self.permutation,
        )

    def t(self) -> implicits.ImplicitVariable:
        from gamspy.math.matrix import permute

        # Ask if exceptions need to be re-raised as GamspyException
        # If  implicit variable needs to be subscriptable since we can
        # create it by transpose
        dims = [x for x in range(len(self.domain))]
        if len(dims) < 2:
            raise GamspyException(
                "Variable must contain at least 2 dimensions to transpose"
            )

        x = dims[-1]
        dims[-1] = dims[-2]
        dims[-2] = x
        return permute(self, dims)

    @property
    def l(self) -> implicits.ImplicitParameter:  # noqa: E741, E743
        return self._l

    @property
    def m(self) -> implicits.ImplicitParameter:
        return self._m

    @property
    def lo(self) -> implicits.ImplicitParameter:
        return self._lo

    @property
    def up(self) -> implicits.ImplicitParameter:
        return self._up

    @property
    def scale(self) -> implicits.ImplicitParameter:
        return self._s

    @property
    def fx(self) -> implicits.ImplicitParameter:
        return self._fx

    @property
    def prior(self) -> implicits.ImplicitParameter:
        return self._prior

    @property
    def stage(self) -> implicits.ImplicitParameter:
        return self._stage

    def __neg__(self):
        return implicits.ImplicitVariable(
            self.parent, name=f"-{self.name}", domain=self.domain
        )

    def __eq__(self, other):  # type: ignore
        return expression.Expression(self, "=e=", other)

    def __ne__(self, other):  # type: ignore
        return expression.Expression(self, "ne", other)

    def gamsRepr(self) -> str:
        representation = self.name
        if self.domain:
            domain = self.domain
            if self.permutation is not None:
                domain = utils._permute_domain(domain, self.permutation)

            representation += utils._get_domain_str(domain)

        return representation
