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
from typing import TYPE_CHECKING
from typing import Union

import gamspy._algebra.expression as expression
import gamspy.utils as utils

if TYPE_CHECKING:
    from gamspy._algebra.expression import Expression


def ifthen(
    condition: "Expression",
    yes_return: Union[float, "Expression"],
    no_return: Union[float, "Expression"],
) -> "Expression":
    """
    If the logical condition is true, the function returns iftrue,
    else it returns else

    Parameters
    ----------
    condition : Expression
    yes_return : float | Expression
    no_return : float | Expression

    Returns
    -------
    Expression

    Examples
    --------
    >>> x = ifthen(tt == 2, 3, 4 + y)
    """
    condition_str = condition.gamsRepr()
    condition_str = utils._replaceEqualitySigns(condition_str)

    ifthen_str = f"ifthen({condition_str}, {yes_return}, {no_return})"
    return expression.Expression(ifthen_str, "", "")
