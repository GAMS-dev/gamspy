#
# GAMS - General Algebraic Modeling System Python API
#
# Copyright (c) 2023 GAMS Development Corp. <support@gams.com>
# Copyright (c) 2023 GAMS Software GmbH <support@gams.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
from __future__ import annotations

from gams.transfer import SpecialValues

import gamspy.math as math
import gamspy.utils as utils
from gamspy._algebra import Card, Domain, Number, Ord, Product, Smax, Smin, Sum
from gamspy._algebra.expression import Expression
from gamspy._backend.engine import EngineClient
from gamspy._backend.neos import NeosClient
from gamspy._container import Container
from gamspy._model import Model, ModelStatus, Problem, Sense, SolveStatus
from gamspy._options import ModelInstanceOptions, Options
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
    "Domain",
    "Number",
    "Ord",
    "Card",
    "Options",
    "ModelInstanceOptions",
    "Expression",
    "EngineClient",
    "NeosClient",
    "math",
    "utils",
    "SpecialValues",
    "__version__",
]
