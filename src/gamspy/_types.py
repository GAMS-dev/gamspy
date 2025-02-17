from __future__ import annotations

from typing import Union

from typing_extensions import TypeAlias

from gamspy._algebra.operable import Operable

OperableType: TypeAlias = Union[Operable, int, float]
EllipsisType: TypeAlias = type(...)  # type: ignore
