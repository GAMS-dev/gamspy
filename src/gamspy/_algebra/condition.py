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

from typing import TYPE_CHECKING

import gamspy._algebra.expression as expression
import gamspy._symbols as syms
import gamspy._symbols.implicits as implicits
from gamspy._symbols.implicits.implicit_symbol import ImplicitSymbol

if TYPE_CHECKING:
    from gamspy._algebra.expression import Expression
    from gamspy._symbols.implicits import (
        ImplicitParameter,
        ImplicitSet,
    )
    from gamspy._symbols.implicits.implicit_symbol import ImplicitSymbol


class Condition:
    """
    Condition class allows symbols to be conditioned.

    Parameters
    ----------
    symbol: ImplicitSymbol | Expression
        Reference to the symbol to be conditioned.

    >>> muf[i, j] = (2.48 + 0.0084 * rd[i, j]).where[rd[i, j]]
    >>> minw[t].where[tm[t]] = Sum(w.where[td[w, t]], x[w, t]) >= tm[t]
    """

    def __init__(self, symbol: ImplicitSymbol | Expression):
        self._symbol = symbol

    def __getitem__(
        self, condition: Expression | ImplicitParameter | ImplicitSet
    ) -> Expression:
        if isinstance(condition, expression.Expression):
            condition._fix_equalities()

        statement = expression.Expression(self._symbol, "$", condition)
        return statement

    def __setitem__(self, condition, rhs):
        # symbol.where[condition] = rhs
        eq_types = (syms.Equation, implicits.ImplicitEquation)
        if isinstance(rhs, bool):
            rhs = "yes" if rhs is True else "no"

        op_type = ".." if isinstance(self._symbol, eq_types) else "="

        if isinstance(condition, expression.Expression):
            condition._fix_equalities()

        lhs = expression.Expression(self._symbol, "$", condition)
        statement = expression.Expression(lhs, op_type, rhs)

        self._symbol.container._add_statement(statement)

        if isinstance(self._symbol, ImplicitSymbol):
            self._symbol.container[self._symbol.parent.name]._is_dirty = True
            self._symbol.parent._assignment = statement

        self._symbol.container._run()
