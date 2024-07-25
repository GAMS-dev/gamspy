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
    domain : list[Set | str], optional
    """

    def __init__(
        self,
        parent: Set | Alias,
        name: str,
        domain: list[Set | str] = ["*"],
        scalar_domains: list[tuple[int, Set]] | None = None,
        extension: str | None = None,
    ) -> None:
        super().__init__(
            parent, name, domain, parent_scalar_domains=scalar_domains
        )
        self.extension = extension

    def __invert__(self) -> Expression:
        return expression.Expression("", "not", self)

    def __ge__(self, other) -> Expression:
        return expression.Expression(self, ">=", other)

    def __le__(self, other) -> Expression:
        return expression.Expression(self, "<=", other)

    def __repr__(self) -> str:
        return f"ImplicitSet(parent={self.parent}, name={self.name}, domain={self.domain}, extension={self.extension}, parent_scalar_domains={self.parent_scalar_domains})"

    @property
    def dimension(self):
        return self.parent.dimension

    def gamsRepr(self) -> str:
        representation = self.name

        if self.extension is not None:
            representation += f"{self.extension}"

        if self.domain != ["*"]:
            domain = list(self.domain)
            for i, d in self._scalar_domains:
                domain.insert(i, d)

            representation += utils._get_domain_str(domain)

        return representation
