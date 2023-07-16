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
import gamspy._algebra._expression as expression
import gamspy._algebra._domain as domain
from typing import Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from gams.transfer import Set, Alias
    from gamspy._algebra import Domain
    from gamspy._algebra._expression import Expression


class Operation(_operable.Operable):
    def __init__(
        self,
        domain: Union[
            "Set",
            "Alias",
            Tuple[Union["Set", "Alias"]],
            "Domain",
            "Expression",
        ],
        expression: Union["Expression", int, bool],
        op_name: str,
    ):
        self.domain = utils._toList(domain)
        assert len(self.domain) > 0, "Operation requires at least one index"
        self.expression = expression
        self._op_name = op_name

    def _get_index_str(self) -> str:
        if len(self.domain) == 1:
            index_str = self.domain[0].gamsRepr()

            if isinstance(self.domain[0], expression.Expression):
                if (
                    "$" in index_str
                    and not isinstance(self.domain[0]._left, domain.Domain)
                    and index_str[0] == "("
                ):
                    index_str = index_str[1:-1]

            return index_str

        return (
            "(" + ",".join([index.gamsRepr() for index in self.domain]) + ")"
        )

    def __eq__(self, other):  # type: ignore
        return expression.Expression(self, "=e=", other)

    def _replace_operations(self, output: str) -> str:
        output = output.replace("=l=", "<=")
        output = output.replace("=g=", ">=")
        output = output.replace("=e=", "==")

        return output

    def gamsRepr(self) -> str:
        # Ex: sum((i,j), c(i,j) * x(i,j))
        output = f"{self._op_name}("

        index_str = self._get_index_str()

        output += index_str
        output += ","

        if isinstance(self.expression, bool):
            self.expression = (
                "yes" if self.expression is True else "no"  # type: ignore
            )

        expression_str = (
            str(self.expression)
            if isinstance(self.expression, (int, str))
            else self.expression.gamsRepr()
        )

        output += expression_str
        output += ")"

        output = self._replace_operations(output)

        return output


class Sum(Operation):
    """
    Represents a sum operation over a domain.

    Parameters
    ----------
    domain : Set | Alias | Tuple[Set | Alias], Domain, Expression
    expression : Union[Expression, int, bool]

    Example
    ----------
    >>> i = gp.Set(m, "i", records=['i1','i2', 'i3'])
    >>> v = gp.Variable(m, "v")
    >>> e = gp.Equation(m, "e", type="eq", domain=[i])
    >>> e[i] = gp.Sum(i, 3) <= v
    """

    def __init__(
        self,
        domain: Union[
            "Set",
            "Alias",
            Tuple[Union["Set", "Alias"]],
            "Domain",
            "Expression",
        ],
        expression: Union["Expression", int, bool],
    ):
        super().__init__(domain, expression, "sum")


class Product(Operation):
    """
    Represents a product operation over a domain.

    Parameters
    ----------
    domain : Set | Alias | Tuple[Set | Alias], Domain, Expression
    expression : Union[Expression, int, bool]

    Example
    ----------
    >>> i = gp.Set(m, "i", records=['i1','i2', 'i3'])
    >>> v = gp.Variable(m, "v")
    >>> e = gp.Equation(m, "e", type="eq", domain=[i])
    >>> e[i] = gp.Product(i, 3) <= v
    """

    def __init__(
        self,
        domain: Union[
            "Set",
            "Alias",
            Tuple[Union["Set", "Alias"]],
            "Domain",
            "Expression",
        ],
        expression: Union["Expression", int, bool],
    ):
        super().__init__(domain, expression, "prod")


class Smin(Operation):
    """
    Represents a smin operation over a domain.

    Parameters
    ----------
    domain : Set | Alias | Tuple[Set | Alias], Domain, Expression
    expression : Union[Expression, int, bool]

    Example
    ----------
    >>> i = gp.Set(m, "i", records=['i1','i2', 'i3'])
    >>> v = gp.Variable(m, "v")
    >>> e = gp.Equation(m, "e", type="eq", domain=[i])
    >>> e[i] = gp.Smin(i, 3) <= v
    """

    def __init__(
        self,
        domain: Union[
            "Set",
            "Alias",
            Tuple[Union["Set", "Alias"]],
            "Domain",
            "Expression",
        ],
        expression: Union["Expression", int, bool],
    ):
        super().__init__(domain, expression, "smin")


class Smax(Operation):
    """
    Represents a smax operation over a domain.

    Parameters
    ----------
    domain : Set | Alias | Tuple[Set | Alias], Domain, Expression
    expression : Union[Expression, int, bool]

    Example
    ----------
    >>> i = gp.Set(m, "i", records=['i1','i2', 'i3'])
    >>> v = gp.Variable(m, "v")
    >>> e = gp.Equation(m, "e", type="eq", domain=[i])
    >>> e[i] = gp.Smax(i, 3) <= v
    """

    def __init__(
        self,
        domain: Union[
            "Set",
            "Alias",
            Tuple[Union["Set", "Alias"]],
            "Domain",
            "Expression",
        ],
        expression: Union["Expression", int, bool],
    ):
        super().__init__(domain, expression, "smax")


class Ord(_operable.Operable):
    """
    Operator ord may be used only with one-dimensional sets.

    Parameters
    ----------
    set : Set | Alias
    """

    def __init__(self, set: Union["Set", "Alias"]):
        self._set = set

    def __eq__(self, other) -> "Expression":  # type: ignore
        return expression.Expression(self, "==", other)

    def __ge__(self, other):
        return expression.Expression(self, ">=", other)

    def __le__(self, other):
        return expression.Expression(self, "<=", other)

    def gamsRepr(self) -> str:
        return f"ord({self._set.name})"


class Card(_operable.Operable):
    """
    The operator card may be used with any set.

    Parameters
    ----------
    set : Set | Alias
    """

    def __init__(self, set: Union["Set", "Alias"]) -> None:
        self._set = set

    def __eq__(self, other) -> "Expression":  # type: ignore
        return expression.Expression(self, "==", other)

    def gamsRepr(self) -> str:
        return f"card({self._set.name})"
