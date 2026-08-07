from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, final

if TYPE_CHECKING:
    from collections.abc import Sequence


@final
class Universe:
    """
    The universe sentinel: the set of every label known to the model (``*`` in GAMS).
    """

    __slots__ = ()

    _instance: ClassVar[Universe | None] = None

    is_universe: Literal[True] = True
    name: Literal["*"] = "*"
    dimension: Literal[1] = 1

    def __new__(cls) -> Universe:
        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance

    def __repr__(self) -> str:
        return "'*'"

    def __str__(self) -> str:
        return "*"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Universe):
            return True

        return isinstance(other, str) and other == "*"

    def __hash__(self) -> int:
        return hash("*")

    def gamsRepr(self) -> Literal["*"]:
        return "*"

    def latexRepr(self) -> Literal["*"]:
        return "*"


UNIVERSE = Universe()


def is_universe(elem) -> bool:
    if isinstance(elem, str):
        return elem == "*"

    return getattr(elem, "is_universe", False)


def is_universe_domain(domain: Sequence) -> bool:
    return len(domain) == 1 and is_universe(domain[0])
