from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

from gamspy._algebra.operable import Operable

OperableType: TypeAlias = Operable | int | float

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import EllipsisType

    import numpy as np
    import pandas as pd

    from gamspy._algebra.condition import Condition
    from gamspy._algebra.domain import Domain
    from gamspy._algebra.expression import Expression
    from gamspy._algebra.operation import Card, Operation, Ord
    from gamspy._symbols import Alias, Equation, Parameter, Set, UniverseAlias, Variable
    from gamspy._symbols.implicits import (
        ImplicitParameter,
        ImplicitSet,
        ImplicitVariable,
    )
    from gamspy.math import Dim, MathOp

    SymbolType: TypeAlias = (
        Set | Alias | Parameter | Variable | Equation | UniverseAlias
    )
    DomainType: TypeAlias = Sequence[Set | Alias | str] | Set | Alias | Dim | str
    IndexType: TypeAlias = (
        EllipsisType | slice | Set | Alias | ImplicitSet | Sequence | str | int
    )
    OperationIndexType: TypeAlias = (
        Set | Alias | ImplicitSet | Sequence[Set | Alias] | Domain | Condition
    )
    OperationRhsType: TypeAlias = (
        Operation
        | Expression
        | MathOp
        | Variable
        | Parameter
        | ImplicitVariable
        | ImplicitParameter
        | ImplicitSet
        | int
        | Ord
        | Card
        | bool
    )

    SetRecordsType: TypeAlias = Sequence | pd.DataFrame | pd.Series
    ParameterRecordsType: TypeAlias = SetRecordsType | np.ndarray | int | float
    VarEquRecordsType: TypeAlias = ParameterRecordsType | dict
