from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

from gamspy._algebra.operable import Operable

OperableType: TypeAlias = Operable | int | float

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import EllipsisType

    from gamspy._symbols import Alias, Set
    from gamspy._symbols.implicits import ImplicitSet

    IndexType: TypeAlias = (
        EllipsisType | slice | Set | Alias | ImplicitSet | Sequence | str | int
    )
