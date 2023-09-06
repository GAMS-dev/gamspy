# flake8: noqa

from gamspy._container import Container
from gamspy._model import Model, ModelStatus, Problem, Sense
from gamspy._symbols import (
    Alias,
    Set,
    Parameter,
    Variable,
    VariableType,
    Equation,
    EquationType,
)
from gamspy._algebra import Domain, Number, Sum, Product, Smax, Smin, Ord, Card

_order = 0  # Global order for newly generated symbols with no name
__version__ = "0.1.0"

__all__ = [
    "Container",
    "Model",
    "ModelStatus",
    "Alias",
    "Set",
    "Parameter",
    "Variable",
    "VariableType",
    "Equation",
    "EquationType",
    "Domain",
    "Number",
    "Sum",
    "Product",
    "Smax",
    "Smin",
    "Ord",
    "Card",
    "Problem",
    "Sense",
]
