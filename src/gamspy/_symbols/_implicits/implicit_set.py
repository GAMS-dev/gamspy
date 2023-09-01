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

import gamspy._algebra.condition as condition
import gamspy._algebra.operable as operable
import gamspy._algebra.expression as expression
import gamspy.utils as utils
from typing import List, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from gamspy import Set, Container
    from gamspy._algebra.expression import Expression


class ImplicitSet(operable.Operable):
    """
    Implicit Set

    Parameters
    ----------
    container : Container
    name : str
    domain : List[Set | str], optional
    """

    def __init__(
        self,
        container: "Container",
        name: str,
        domain: List[Union["Set", str]] = [],
    ) -> None:
        self.ref_container = container
        self.name = name
        self.domain = domain
        self.where = condition.Condition(self)

    def __invert__(self) -> "Expression":
        return expression.Expression("", "not", self)

    def __ge__(self, other) -> "Expression":
        return expression.Expression(self, ">=", other)

    def __le__(self, other) -> "Expression":
        return expression.Expression(self, "<=", other)

    def gamsRepr(self) -> str:
        representation = self.name

        if self.domain:
            representation += utils._getDomainStr(self.domain)

        return representation
