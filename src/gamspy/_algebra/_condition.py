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

import gamspy._algebra._expression as expression


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
        return expression.Expression(self._symbol, "$", condition_expression)

    def __setitem__(self, condition_expression, right_handexpression) -> None:
        import gamspy._symbols._implicits as implicits
        import gamspy._symbols as syms

        if not hasattr(self._symbol, "ref_container"):
            raise Exception(
                f"Container must be defined for symbol {self._symbol.name}"
            )

        self._symbol._is_dirty = True

        op_type = (
            ".."
            if isinstance(
                self._symbol, (syms.Equation, implicits.ImplicitEquation)
            )
            else "="
        )

        condition = condition_expression.gamsRepr()

        if "=l=" in condition:
            condition = condition.replace("=l=", "<=")

        if "=g=" in condition:
            condition = condition.replace("=g=", ">=")

        statement = expression.Expression(
            expression.Expression(self._symbol, "$", condition),
            op_type,
            right_handexpression,
        )

        self._symbol.ref_container._addStatement(statement)
