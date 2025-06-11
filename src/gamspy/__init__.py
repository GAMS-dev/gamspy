from __future__ import annotations

from gams.transfer import SpecialValues

import gamspy.formulations as formulations
import gamspy.math as math
import gamspy.utils as utils
from gamspy._algebra import (
    Card,
    Domain,
    Number,
    Ord,
    Product,
    Sand,
    Smax,
    Smin,
    Sor,
    Sum,
)
from gamspy._algebra.expression import Expression
from gamspy._backend.engine import EngineClient
from gamspy._backend.neos import NeosClient
from gamspy._config import _set_default_options, get_option, set_options
from gamspy._container import Container
from gamspy._model import (
    FileFormat,
    Model,
    ModelStatus,
    Problem,
    Sense,
    SolveStatus,
)
from gamspy._options import (
    ConvertOptions,
    FreezeOptions,
    ModelInstanceOptions,
    Options,
)
from gamspy._serialization import deserialize, serialize
from gamspy._symbols import (
    Alias,
    Equation,
    EquationType,
    Parameter,
    Set,
    UniverseAlias,
    Variable,
    VariableType,
)

from .version import __version__

_ctx_managers: dict[tuple[int, int], Container] = dict()
_set_default_options()

__all__ = [
    "Container",
    "Set",
    "Alias",
    "UniverseAlias",
    "Parameter",
    "Variable",
    "Equation",
    "Model",
    "Problem",
    "Sense",
    "VariableType",
    "EquationType",
    "ModelStatus",
    "SolveStatus",
    "Sum",
    "Product",
    "Smax",
    "Smin",
    "Sand",
    "Sor",
    "Domain",
    "Number",
    "Ord",
    "Card",
    "Options",
    "FreezeOptions",
    "ModelInstanceOptions",
    "ConvertOptions",
    "FileFormat",
    "Expression",
    "EngineClient",
    "NeosClient",
    "math",
    "formulations",
    "utils",
    "SpecialValues",
    "__version__",
    "get_option",
    "set_options",
    "serialize",
    "deserialize",
]
