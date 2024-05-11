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
    """
    Implicit Variable

    Parameters
    ----------
    parent : Variable
    name : str
    domain : List[Set | str]
    """

    def __init__(
        self,
        parent: Variable,
        name: str,
        domain: list[Set | str],
    ):
        super().__init__(parent, name, domain)
        self._l, self._m, self._lo, self._up, self._s = self._init_attributes()
        self._fx = self._create_attr("fx")
        self._prior = self._create_attr("prior")
        self._stage = self._create_attr("stage")

    def _create_attr(self, attr_name: str):
        return implicits.ImplicitParameter(
            self.parent, f"{self.gamsRepr()}.{attr_name}"
        )

    def _init_attributes(self):
        level = self._create_attr("l")
        marginal = self._create_attr("m")
        lower = self._create_attr("lo")
        upper = self._create_attr("up")
        scale = self._create_attr("scale")
        return level, marginal, lower, upper, scale

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
            representation += utils._get_domain_str(self.domain)

        return representation
