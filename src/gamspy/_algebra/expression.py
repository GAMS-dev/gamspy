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

import gamspy._algebra.condition as condition
import gamspy._algebra.domain as domain
import gamspy._algebra.operable as operable
import gamspy._symbols as syms
import gamspy._symbols.implicits as implicits
import gamspy.utils as utils


class Expression(operable.Operable):
    """
    Expression of two operands and an operation.

    Parameters
    ----------
    left: str | int | float | Parameter | Variable
        Left operand
    op_type: str
        Operation
    right: str | int | float | Parameter | Variable
        Right operand

    Examples
    --------
    >>> a = Parameter(name="a", records=[["a", 1], ["b", 2], ["c", 3]]))
    >>> b = Parameter(name="b", records=[["a", 1], ["b", 2], ["c", 3]]))
    >>> expression = a * b
    Expression(a, "*", b)
    >>> expression.gamsRepr()
    (a * b)
    """

    def __init__(self, left, op_type, right) -> None:
        self.name = utils._getUniqueName()
        self._left = left
        self._op_type = op_type
        self._right = right
        self.where = condition.Condition(self)

    def __eq__(self, other):  # type: ignore
        return Expression(self, "=e=", other)

    def _fix_condition_paranthesis(self, string: str) -> str:
        if self._op_type == "$":
            left, right = string.split("$", 1)
            right = right.strip()

            if right[0] != "(":
                right = f"({right})"

            string = f"{left}$ {right}"

        return string

    def _get_operand_representations(self) -> Tuple[str, str]:
        # Builtin Python types do not have gams representation
        left_str = (
            str(self._left)
            if isinstance(self._left, (str, int, float))
            else self._left.gamsRepr()
        )
        right_str = (
            str(self._right)
            if isinstance(self._right, (str, int, float))
            else self._right.gamsRepr()
        )

        # negative sign causes an extra operation if not in paranthesis
        # ((((ord(n) - 1) / 10) * -1) + ((ord(n) / 10) * 0)); -> not valid
        # ((((ord(n) - 1) / 10) * (-1)) + ((ord(n) / 10) * 0)); -> valid
        if isinstance(self._left, (int, float)) and self._left < 0:
            left_str = f"({left_str})"

        if isinstance(self._right, (int, float)) and self._right < 0:
            right_str = f"({right_str})"

        if self._op_type == "=" and isinstance(
            self._left,
            (
                syms.Set,
                syms.Parameter,
                syms.Variable,
                implicits.ImplicitSet,
                implicits.ImplicitParameter,
                implicits.ImplicitVariable,
            ),
        ):
            right_str = right_str.replace("=e=", "=")
            right_str = right_str.replace("=l=", "<=")
            right_str = right_str.replace("=g=", ">=")

        return left_str, right_str

    def gamsRepr(self) -> str:
        """Representation of this Expression in GAMS language.

        Returns
        -------
        str
        """
        left_str, right_str = self._get_operand_representations()

        out_str = f"{left_str} {self._op_type} {right_str}"

        if self._op_type not in ["..", "="]:
            # add paranthesis for right ordering
            out_str = f"({out_str})"

        if isinstance(self._left, (domain.Domain, syms.Set, syms.Alias)):
            return out_str[1:-1]

        if self._op_type in [
            "=g=",
            "=l=",
            "=e=",
            "=n=",
            "=x=",
            "=c=",
            "=b=",
            ".",
        ]:
            # (test.. a =g= b) -> test.. a =g= b
            out_str = out_str[1:-1]  # remove the paranthesis

        out_str = self._fix_condition_paranthesis(out_str)

        if self._op_type == ".":
            # name . pos -> name.pos
            out_str = out_str.replace(" ", "")  # remove spaces

        if self._op_type in ["..", "="]:
            # add ; to equation expressions and assignments
            out_str += ";"

        if self._op_type == "==":
            # volume.lo(t)$(ord(t) = card(t)) = 2000;
            out_str = out_str.replace("==", "=")

        if self._op_type in ["=", ".."] and out_str[0] == "(":
            # (voycap(j,k)$vc(j,k)).. sum(.) -> voycap(j,k)$vc(j,k).. sum(.)
            indices = utils._getMatchingParanthesisIndices(out_str)
            match_index = indices[0]
            out_str = out_str[1:match_index] + out_str[match_index + 1 :]

        return out_str

    def getStatement(self) -> str:
        """Conditioned equations become an Expression.

        Returns
        -------
        str
        """
        return self.gamsRepr()
