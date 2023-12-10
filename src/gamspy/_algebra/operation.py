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

from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union

import gamspy._algebra.condition as condition
import gamspy._algebra.domain as domain
import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
import gamspy._symbols.implicits as implicits
import gamspy.utils as utils

if TYPE_CHECKING:
    from gams.transfer import Set, Alias, Parameter
    from gamspy._algebra import Domain
    from gamspy._algebra.expression import Expression
    from gamspy._symbols.implicits import ImplicitVariable
    from gamspy._symbols.implicits import ImplicitParameter


class Operation(operable.Operable):
    def __init__(
        self,
        domain: Union[
            Set,
            Alias,
            Tuple[Union[Set, Alias]],
            Domain,
            Expression,
        ],
        expression: Union[
            Expression, ImplicitVariable, ImplicitParameter, int, bool
        ],
        op_name: str,
    ):
        self.domain = utils._to_list(domain)
        assert len(self.domain) > 0, "Operation requires at least one index"
        self.expression = expression
        self._op_name = op_name

        # allow conditions
        self.where = condition.Condition(self)

    def _extract_variables(self):
        if isinstance(self.expression, expression.Expression):
            return self.expression.find_variables()
        elif isinstance(self.expression, implicits.ImplicitVariable):
            return [self.expression.parent.name]
        return []

    def _get_index_str(self) -> str:
        if len(self.domain) == 1:
            index_str = self.domain[0].gamsRepr()

            if isinstance(self.domain[0], expression.Expression):
                if (
                    "$" in index_str
                    and not isinstance(self.domain[0].left, domain.Domain)
                    and index_str[0] == "("
                ):
                    # sum((tt(t)) $ (ord(t) <= pMinDown(g,t1)), ...) ->
                    # sum(tt(t) $ (ord(t) <= pMinDown(g,t1)), ...)
                    index_str = index_str[1:-1]

            return index_str

        return (
            "(" + ",".join([index.gamsRepr() for index in self.domain]) + ")"
        )

    def __eq__(self, other):  # type: ignore
        return expression.Expression(self, "=e=", other)

    def __ne__(self, other):  # type: ignore
        return expression.Expression(self, "ne", other)

    def __neg__(self):
        return expression.Expression(None, "-", self)

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
    expression : Expression | int | bool

    Examples
    --------
    >>> i = gp.Set(m, "i", records=['i1','i2', 'i3'])
    >>> v = gp.Variable(m, "v")
    >>> e = gp.Equation(m, "e", type="eq", domain=[i])
    >>> e[i] = gp.Sum(i, 3) <= v
    """

    def __init__(
        self,
        domain: Union[
            Set,
            Alias,
            Tuple[Union[Set, Alias]],
            Domain,
            Expression,
        ],
        expression: Union[Expression, int, bool],
    ):
        super().__init__(domain, expression, "sum")


class Product(Operation):
    """
    Represents a product operation over a domain.

    Parameters
    ----------
    domain : Set | Alias | Tuple[Set | Alias], Domain, Expression
    expression : Expression | int | bool

    Examples
    --------
    >>> i = gp.Set(m, "i", records=['i1','i2', 'i3'])
    >>> v = gp.Variable(m, "v")
    >>> e = gp.Equation(m, "e", type="eq", domain=[i])
    >>> e[i] = gp.Product(i, 3) <= v
    """

    def __init__(
        self,
        domain: Union[
            Set,
            Alias,
            Tuple[Union[Set, Alias]],
            Domain,
            Expression,
        ],
        expression: Union[Expression, int, bool],
    ):
        super().__init__(domain, expression, "prod")


class Smin(Operation):
    """
    Represents a smin operation over a domain.

    Parameters
    ----------
    domain : Set | Alias | Tuple[Set | Alias], Domain, Expression
    expression : Expression | int | bool

    Examples
    --------
    >>> i = gp.Set(m, "i", records=['i1','i2', 'i3'])
    >>> v = gp.Variable(m, "v")
    >>> e = gp.Equation(m, "e", type="eq", domain=[i])
    >>> e[i] = gp.Smin(i, 3) <= v
    """

    def __init__(
        self,
        domain: Union[
            Set,
            Alias,
            Tuple[Union[Set, Alias]],
            Domain,
            Expression,
        ],
        expression: Union[Expression, int, bool],
    ):
        super().__init__(domain, expression, "smin")


class Smax(Operation):
    """
    Represents a smax operation over a domain.

    Parameters
    ----------
    domain : Set | Alias | Tuple[Set | Alias], Domain, Expression
    expression : Expression | int | bool

    Examples
    --------
    >>> i = gp.Set(m, "i", records=['i1','i2', 'i3'])
    >>> v = gp.Variable(m, "v")
    >>> e = gp.Equation(m, "e", type="eq", domain=[i])
    >>> e[i] = gp.Smax(i, 3) <= v
    """

    def __init__(
        self,
        domain: Union[
            Set,
            Alias,
            Tuple[Union[Set, Alias]],
            Domain,
            Expression,
        ],
        expression: Union[Expression, int, bool],
    ):
        super().__init__(domain, expression, "smax")


class Ord(operable.Operable):
    """
    The operator ord may be used only with one-dimensional sets.

    Parameters
    ----------
    set : Set | Alias

    Examples
    --------
    >>> import gamspy as gp
    >>>
    >>> m = gp.Container()
    >>> t = gp.Set(
    >>>     m,
    >>>     name="t",
    >>>     description="time periods",
    >>>     records=[str(x) for x in range(1985, 1996)],
    >>> )
    >>> val = gp.Parameter(m, name="val", domain=[t])
    >>> val[t] = gp.Ord(t)
    """

    def __init__(self, set: Union[Set, Alias]):
        self._set = set

    def __eq__(self, other) -> Expression:  # type: ignore
        return expression.Expression(self, "==", other)

    def __ge__(self, other):
        return expression.Expression(self, ">=", other)

    def __le__(self, other):
        return expression.Expression(self, "<=", other)

    def __ne__(self, other):  # type: ignore
        return expression.Expression(self, "ne", other)

    def gamsRepr(self) -> str:
        return f"ord({self._set.name})"


class Card(operable.Operable):
    """
    The operator card may be used with any symbol and returns its number of records.

    Parameters
    ----------
    symbol : Set | Alias | Parameter | Variable | Equation | Model

    Examples
    --------
    >>> import gamspy as gp
    >>>
    >>> m = gp.Container()
    >>>
    >>> t = gp.Set(
    >>>     m,
    >>>     name="t",
    >>>     description="time periods",
    >>>     records=[str(x) for x in range(1985, 1996)],
    >>> )
    >>> s = gp.Parameter(m, name="s")
    >>> s[...] = gp.Card(t)
    """

    def __init__(
        self,
        symbol: Union[
            Set,
            Alias,
            Parameter,
        ],
    ) -> None:
        self._symbol = symbol

    def __eq__(self, other) -> Expression:  # type: ignore
        return expression.Expression(self, "==", other)

    def __ne__(self, other):  # type: ignore
        return expression.Expression(self, "ne", other)

    def gamsRepr(self) -> str:
        return f"card({self._symbol.name})"
