from __future__ import annotations

from typing import TYPE_CHECKING

import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
import gamspy.utils as utils
from gamspy._symbols.implicits.implicit_symbol import ImplicitSymbol

if TYPE_CHECKING:
    from gamspy import Alias, Set
    from gamspy._algebra.expression import Expression


class ImplicitSet(ImplicitSymbol, operable.Operable):
    """
    Implicit Set

    Parameters
    ----------
    container : Container
    name : str
    domain : List[Set | str], optional
    """

    def __init__(
        self,
        parent: Set | Alias,
        name: str,
        domain: list[Set | str] = ["*"],
    ) -> None:
        super().__init__(parent, name, domain)

    def __invert__(self) -> Expression:
        return expression.Expression("", "not", self)

    def __ge__(self, other) -> Expression:
        return expression.Expression(self, ">=", other)

    def __le__(self, other) -> Expression:
        return expression.Expression(self, "<=", other)

    @property
    def dimension(self):
        return self.parent.dimension

    def gamsRepr(self) -> str:
        representation = self.name

        if self.domain != ["*"]:
            representation += utils._get_domain_str(self.domain)

        return representation
