from __future__ import annotations

from typing import TYPE_CHECKING

import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
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

    def __neg__(self):
        return expression.Expression(None, "-", self)

    def __eq__(self, other):  # type: ignore
        return expression.Expression(self, "=e=", other)

    def __ne__(self, other):  # type: ignore
        return expression.Expression(self, "ne", other)

    def gamsRepr(self) -> str:
        representation = self.name
        if self.domain:
            representation += utils._get_domain_str(self.domain)

        return representation
