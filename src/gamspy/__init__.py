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
# flake8: noqa
from gamspy._algebra import Card
from gamspy._algebra import Domain
from gamspy._algebra import Number
from gamspy._algebra import Ord
from gamspy._algebra import Product
from gamspy._algebra import Smax
from gamspy._algebra import Smin
from gamspy._algebra import Sum
from gamspy._container import Container
from gamspy._model import Model
from gamspy._model import ModelStatus
from gamspy._model import Problem
from gamspy._model import Sense
from gamspy._symbols import Alias
from gamspy._symbols import Equation
from gamspy._symbols import EquationType
from gamspy._symbols import Parameter
from gamspy._symbols import Set
from gamspy._symbols import Variable
from gamspy._symbols import VariableType

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
