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
import gamspy._algebra.expression as expression
import gamspy._symbols as syms
import gamspy._symbols.implicits as implicits
import gamspy.utils as utils
from gamspy._symbols.implicits.implicit_symbol import ImplicitSymbol


class Condition:
    """
    Condition class allows symbols to be conditioned.

    Parameters
    ----------
    symbol: Alias | Set | Parameter | Variable | Equation | Expression
        Reference to the symbol to be conditioned.

    >>> muf[i, j] = (2.48 + 0.0084 * rd[i, j]).where[rd[i, j]]
    >>> minw[t].where[tm[t]] = Sum(w.where[td[w, t]], x[w, t]) >= tm[t]
    """

    def __init__(self, symbol):
        self._symbol = symbol

    def __getitem__(self, condition_expression):
        condition = utils._replaceEqualitySigns(
            condition_expression.gamsRepr()
        )
        return expression.Expression(self._symbol, "$", condition)

    def __setitem__(self, condition_expression, right_hand_expression) -> None:
        assert hasattr(
            self._symbol, "container"  # pragma: no cover
        ), f"Container must be defined for symbol {self._symbol.name}"

        if isinstance(self._symbol, ImplicitSymbol):
            self._symbol.container[self._symbol.parent.name]._is_dirty = True
        else:
            self._symbol.container[self._symbol.name]._is_dirty = True

        if isinstance(right_hand_expression, bool):
            right_hand_expression = (
                "yes" if right_hand_expression is True else "no"
            )

        op_type = (
            ".."
            if isinstance(
                self._symbol, (syms.Equation, implicits.ImplicitEquation)
            )
            else "="
        )

        condition = utils._replaceEqualitySigns(
            condition_expression.gamsRepr()
        )

        statement = expression.Expression(
            expression.Expression(self._symbol, "$", condition),
            op_type,
            right_hand_expression,
        )

        self._symbol.container._addStatement(statement)

        if self._symbol.container.debug:
            self._symbol.container._loadOnDemand()
