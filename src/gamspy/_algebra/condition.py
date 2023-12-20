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
import gamspy.utils as utils
from gamspy._symbols.implicits.implicit_symbol import ImplicitSymbol

if TYPE_CHECKING:
    from gamspy._symbols.implicits import (
        ImplicitSet,
        ImplicitParameter,
        ImplicitEquation,
    )
    from gamspy._algebra.expression import Expression


class Condition:
    """
    Condition class allows symbols to be conditioned.

    Parameters
    ----------
    symbol: ImplicitSet | ImplicitParameter | ImplicitEquation | Expression
        Reference to the symbol to be conditioned.

    >>> muf[i, j] = (2.48 + 0.0084 * rd[i, j]).where[rd[i, j]]
    >>> minw[t].where[tm[t]] = Sum(w.where[td[w, t]], x[w, t]) >= tm[t]
    """

    def __init__(
        self,
        symbol: (
            "ImplicitSet"
            | "ImplicitParameter"
            | "ImplicitEquation"
            | "Expression"
        ),
    ):
        self._symbol = symbol

    def __getitem__(
        self, condition: "Expression" | "ImplicitParameter" | "ImplicitSet"
    ) -> Expression:
        sign_map = {"=g=": ">=", "=e=": "==", "=l=": "<="}
        if (
            isinstance(condition, expression.Expression)
            and condition.data in sign_map.keys()
        ):
            condition.data = sign_map[condition.data]
            condition.representation = condition._create_representation()
        return expression.Expression(self._symbol, "$", condition)

    def __setitem__(self, condition_expression, right_hand_expression) -> None:
        if isinstance(self._symbol, ImplicitSymbol):
            self._symbol.container[self._symbol.parent.name]._is_dirty = True

        if isinstance(right_hand_expression, bool):
            right_hand_expression = (
                "yes" if right_hand_expression is True else "no"
            )

        if isinstance(
            self._symbol, (syms.Equation, implicits.ImplicitEquation)
        ):
            op_type = ".."
        else:
            op_type = "="

        condition = utils._replace_equality_signs(
            condition_expression.gamsRepr()
        )

        statement = expression.Expression(
            expression.Expression(self._symbol, "$", condition),
            op_type,
            right_hand_expression,
        )

        if isinstance(self._symbol, implicits.ImplicitEquation):
            self._symbol.parent._definition = statement

        self._symbol.container._add_statement(statement)

        if not self._symbol.container.delayed_execution and not isinstance(
            self._symbol, implicits.ImplicitEquation
        ):
            self._symbol.container._run()
