#
# GAMS - General Algebraic Modeling System Python API
#
# Copyright (c) 2017-2023 GAMS Development Corp. <support@gams.com>
# Copyright (c) 2017-2023 GAMS Software GmbH <support@gams.com>
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

import gamspy._algebra._operable as _operable
import gamspy.utils as utils
import gamspy._algebra._expression as _expression
from typing import List, Union, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from gams.transfer import Set, Alias
    from gamspy._algebra import Domain


class Operation(_operable.Operable):
    def __init__(
        self,
        domain: List[Union["Domain", "Set", "Alias", str]],
        expression: _expression.Expression,
        op_name: str,
    ):
        self.domain = utils._toList(domain)
        assert len(self.domain) > 0, "Operation requires at least one index"
        self._expression = expression
        self._op_name = op_name

    def _get_index_str(self):
        if len(self.domain) == 1:
            return self.domain[0].gamsRepr()

        return (
            "(" + ",".join([index.gamsRepr() for index in self.domain]) + ")"
        )

    def __eq__(self, other):  # type: ignore
        return _expression.Expression(self, "=e=", other)

    def gamsRepr(self) -> str:
        # Ex: sum((i,j), c(i,j) * x(i,j))
        output = f"{self._op_name}("

        index_str = self._get_index_str()

        output += index_str
        output += ","

        expression_str = self._expression.gamsRepr()

        output += expression_str
        output += ")"

        return output


class Sum(Operation):
    def __init__(
        self,
        domain: List[Union["Domain", "Set", "Alias", str]],
        expression: _expression.Expression,
    ):
        super().__init__(domain, expression, "sum")


class Product(Operation):
    def __init__(
        self,
        domain: List[Union["Domain", "Set", "Alias", str]],
        expression: _expression.Expression,
    ):
        super().__init__(domain, expression, "prod")


class Smin(Operation):
    def __init__(
        self,
        domain: List[Union["Domain", "Set", "Alias", str]],
        expression: _expression.Expression,
    ):
        super().__init__(domain, expression, "smin")


class Smax(Operation):
    def __init__(
        self,
        domain: List[Union["Domain", "Set", "Alias", str]],
        expression: _expression.Expression,
    ):
        super().__init__(domain, expression, "smax")


class Ord(_operable.Operable):
    """
    Operator ord may be used only with one-dimensional sets.
    """

    def __init__(self, set: "Set"):
        self._set = set

    def __eq__(self, other) -> _expression.Expression:  # type: ignore
        return _expression.Expression(self, "==", other)

    def gamsRepr(self) -> str:
        return f"ord({self._set.name})"


class Card(_operable.Operable):
    """
    The operator card may be used with any set.
    """

    def __init__(self, set: "Set") -> None:
        self._set = set

    def __eq__(self, other) -> _expression.Expression:  # type: ignore
        return _expression.Expression(self, "==", other)

    def gamsRepr(self) -> str:
        return f"card({self._set.name})"
