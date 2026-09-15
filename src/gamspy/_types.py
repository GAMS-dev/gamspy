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
    from gamspy._algebra.expression import Expression, ShiftExpression
    from gamspy._algebra.number import Number
    from gamspy._algebra.operation import Card, Operation, Ord
    from gamspy._internals import DataSource
    from gamspy._model_instance import ModelInstance
    from gamspy._symbols import Alias, Equation, Parameter, Set, UniverseAlias, Variable
    from gamspy._symbols.implicits import (
        ImplicitEquation,
        ImplicitParameter,
        ImplicitSet,
        ImplicitVariable,
    )
    from gamspy._universe import Universe
    from gamspy.math import Dim, MathOp

    SymbolType: TypeAlias = (
        Set | Alias | Parameter | Variable | Equation | UniverseAlias
    )
    ImplicitSymbolType: TypeAlias = (
        ImplicitSet | ImplicitParameter | ImplicitVariable | ImplicitEquation
    )
    SymbolWithRecordsType: TypeAlias = Set | Parameter | Variable | Equation

    # Where the records of a symbol must be read from
    RecordsSourceType: TypeAlias = DataSource | ModelInstance

    # Possible types that the user can provide as a domain. A `str` is either
    # the universe label "*" or a relaxed domain.
    DomainType: TypeAlias = (
        Sequence[Set | Alias | UniverseAlias | Universe | str]
        | Set
        | Alias
        | UniverseAlias
        | Universe
        | Dim
        | str
    )

    # Possible types after normalization of the provided domain. The universe
    # label "*" has been replaced by the `UNIVERSE` sentinel, so a remaining
    # `str` is always a relaxed domain.
    NormalizedDomainType: TypeAlias = Sequence[
        Set | Alias | UniverseAlias | Universe | str
    ]
    IndexType: TypeAlias = (
        EllipsisType
        | slice
        | Set
        | Alias
        | UniverseAlias
        | Universe
        | ImplicitSet
        | Expression
        | ImplicitParameter
        | ShiftExpression
        | Sequence
        | str
        | int
        | Condition
    )
    OperationIndexType: TypeAlias = (
        Set
        | Alias
        | ImplicitSet
        | Domain
        | Condition
        | MathOp
        | Sequence[Set | Alias | ImplicitSet | Domain | Condition | MathOp]
    )

    # Possible types of a dollar condition.
    ConditionType: TypeAlias = (
        Expression
        | Operation
        | Condition
        | MathOp
        | Card
        | Ord
        | Set
        | Alias
        | UniverseAlias
        | Parameter
        | Variable
        | ImplicitSet
        | ImplicitParameter
        | ImplicitVariable
    )
    OperationRhsType: TypeAlias = (
        Operation
        | Expression
        | Condition
        | MathOp
        | Variable
        | Parameter
        | ImplicitVariable
        | ImplicitParameter
        | ImplicitSet
        | int
        | float
        | Ord
        | Card
        | bool
        | Number
    )

    SetRecordsType: TypeAlias = (
        Sequence | pd.DataFrame | np.ndarray | pd.Series | dict[str, float]
    )
    ParameterRecordsType: TypeAlias = SetRecordsType | np.ndarray | int | float
    VarEquRecordsType: TypeAlias = ParameterRecordsType | dict
