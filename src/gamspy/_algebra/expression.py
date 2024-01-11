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

import gamspy as gp
import gamspy._algebra.condition as condition
import gamspy._algebra.domain as domain
import gamspy._algebra.operable as operable
import gamspy._algebra.operation as operation
import gamspy._symbols as syms
import gamspy._symbols.implicits as implicits
import gamspy.utils as utils
from gamspy.math.misc import MathOp

if TYPE_CHECKING:
    from gamspy import Variable

GMS_MAX_LINE_LENGTH = 80000
LINE_LENGTH_OFFSET = 79000


class Expression(operable.Operable):
    """
    Expression of two operands and an operation.

    Parameters
    ----------
    left: str | int | float | Parameter | Variable | None
        Left operand
    data: str
        Operation
    right: str | int | float | Parameter | Variable | None
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

    def __init__(self, left, data, right) -> None:
        self.left = left
        self.data = data
        self.right = right
        self.representation = self._create_representation()
        self.where = condition.Condition(self)

    def _create_representation(self):
        left_str, right_str = self._get_operand_representations()
        out_str = self._create_output_str(left_str, right_str)

        # Adapt to GAMS quirks
        if isinstance(self.left, (domain.Domain, syms.Set, syms.Alias)):
            return out_str[1:-1]

        if self.data == "$":
            # defopLS(o,p) $ sumc(o,p) <= 0.5 .. op(o,p) =e= 1;   -> not valid
            # defopLS(o,p) $ (sumc(o,p) <= 0.5) .. op(o,p) =e= 1; -> valid
            out_str = self._fix_condition_paranthesis(out_str)

        if self.data in ["=", ".."] and out_str[0] == "(":
            # (voycap(j,k)$vc(j,k)).. sum(.) -> not valid
            # voycap(j,k)$vc(j,k).. sum(.)   -> valid
            indices = utils._get_matching_paranthesis_indices(out_str)
            match_index = indices[0]
            out_str = out_str[1:match_index] + out_str[match_index + 1 :]

        return out_str

    def _get_operand_representations(self) -> tuple[str, str]:
        if isinstance(self.left, float):
            self.left = utils._map_special_values(self.left)

        if isinstance(self.right, float):
            self.right = utils._map_special_values(self.right)

        if self.left is None:
            left_str = ""
        else:
            left_str = (
                str(self.left)
                if isinstance(self.left, (int, float, str))
                else self.left.gamsRepr()
            )

        if self.right is None:
            right_str = ""
        else:
            right_str = (
                str(self.right)
                if isinstance(self.right, (int, float, str))
                else self.right.gamsRepr()
            )

        # ((((ord(n) - 1) / 10) * -1) + ((ord(n) / 10) * 0)); -> not valid
        # ((((ord(n) - 1) / 10) * (-1)) + ((ord(n) / 10) * 0)); -> valid
        if isinstance(self.left, (int, float)) and self.left < 0:
            left_str = f"({left_str})"

        if isinstance(self.right, (int, float)) and self.right < 0:
            right_str = f"({right_str})"

        if self.data == "=" and isinstance(
            self.left,
            (implicits.ImplicitSet, implicits.ImplicitParameter),
        ):
            # error02(s1,s2) = (lfr(s1,s2) and sum(l(root,s,s1,s2),1) =e= 0); -> not valid
            # error02(s1,s2) = (lfr(s1,s2) and sum(l(root,s,s1,s2),1) = 0); -> valid
            right_str = utils._replace_equality_signs(right_str)

        return left_str, right_str

    def _create_output_str(self, left_str: str, right_str: str) -> str:
        # get around 80000 line length limitation in GAMS
        length = len(left_str) + len(self.data) + len(right_str)
        if length >= GMS_MAX_LINE_LENGTH - LINE_LENGTH_OFFSET:
            out_str = f"{left_str} {self.data}\n {right_str}"
        else:
            out_str = f"{left_str} {self.data} {right_str}"

        if self.data in ["..", "="]:
            return f"{out_str};"
        elif self.data in ["=g=", "=l=", "=e=", "=n=", "=x=", "=c=", "=b="]:
            return out_str

        return f"({out_str})"

    def __eq__(self, other):  # type: ignore
        return Expression(self, "=e=", other)

    def __ne__(self, other):  # type: ignore
        return Expression(self, "ne", other)

    def __neg__(self):
        return Expression(None, "-", self)

    def _fix_condition_paranthesis(self, string: str) -> str:
        left, right = string.split("$", 1)
        right = right.strip()

        if right[0] != "(":
            right = f"({right})"

        string = f"{left}$ {right}"

        return string

    def replace_operator(self, operator: str):
        self.data = operator
        self.representation = self._create_representation()

    def gamsRepr(self) -> str:
        """
        Representation of this Expression in GAMS language.

        Returns
        -------
        str
        """
        return self.representation

    def getStatement(self) -> str:
        """
        Statement of this Expression in .gms file.

        Returns
        -------
        str
        """
        return self.gamsRepr()

    def find_variables(self) -> list[Variable]:
        current = self

        stack = []
        variables: list[Variable] = []

        while True:
            if current is not None:
                stack.append(current)

                current = current.left if hasattr(current, "left") else None
            elif stack:
                current = stack.pop()

                if hasattr(current, "data") and isinstance(
                    current.data, MathOp
                ):
                    variables += current.data.find_variables()

                if isinstance(current, gp.Variable):
                    variables.append(current.name)
                elif isinstance(current, implicits.ImplicitVariable):
                    variables.append(current.parent.name)
                elif isinstance(current, operation.Operation):
                    operation_variables = current._extract_variables()
                    variables += operation_variables

                current = current.right if hasattr(current, "right") else None
            else:
                break

        return list(set(variables))
