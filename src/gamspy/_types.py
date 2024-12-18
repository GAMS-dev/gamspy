from __future__ import annotations

import builtins
from typing import Union

from typing_extensions import TypeAlias

from gamspy._algebra.operable import Operable

OperableType: TypeAlias = Union[Operable, int, float]
EllipsisType = builtins.ellipsis
