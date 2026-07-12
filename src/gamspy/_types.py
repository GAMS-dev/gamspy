from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias

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
    from gamspy._algebra.number import Number
    from gamspy._algebra.operation import Card, Operation, Ord
    from gamspy._symbols import Alias, Equation, Parameter, Set, UniverseAlias, Variable
    from gamspy._symbols.implicits import (
        ImplicitEquation,
        ImplicitParameter,
        ImplicitSet,
        ImplicitVariable,
    )
    from gamspy.math import Dim, MathOp

    SymbolType: TypeAlias = (
        Set | Alias | Parameter | Variable | Equation | UniverseAlias
    )
    ImplicitSymbolType: TypeAlias = (
        ImplicitSet | ImplicitParameter | ImplicitVariable | ImplicitEquation
    )
    SymbolWithRecordsType: TypeAlias = Set | Parameter | Variable | Equation

    # Possible types that the user can provide as a domain
    DomainType: TypeAlias = (
        Sequence[Set | Alias | UniverseAlias | Literal["*"]]
        | Set
        | Alias
        | UniverseAlias
        | Dim
        | Literal["*"]
    )

    # Possible types after normalization of the provided domain.
    NormalizedDomainType: TypeAlias = Sequence[
        Set | Alias | UniverseAlias | Literal["*"]
    ]
    IndexType: TypeAlias = (
        EllipsisType
        | slice
        | Set
        | Alias
        | UniverseAlias
        | ImplicitSet
        | ImplicitParameter
        | Sequence
        | str
        | int
        | Condition
    )
    OperationIndexType: TypeAlias = (
        Set | Alias | ImplicitSet | Sequence[Set | Alias] | Domain | Condition | MathOp
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
        | Number
    )

    SetRecordsType: TypeAlias = Sequence | pd.DataFrame | pd.Series | dict[str, float]
    ParameterRecordsType: TypeAlias = SetRecordsType | np.ndarray | int | float
    VarEquRecordsType: TypeAlias = ParameterRecordsType | dict
