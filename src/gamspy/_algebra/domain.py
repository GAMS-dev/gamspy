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
import gamspy.utils as utils

if TYPE_CHECKING:
    from gamspy import Set, Alias
    from gamspy._symbols.implicits import ImplicitSet


class DomainException(Exception):
    """Exception raised if a domain is not valid."""


class Domain:
    """
    Domain class needed for where statements on multidimensional index list
    in operations

    Parameters
    ----------
    sets: tuple[Set | str]

    >>> equation = Equation(name="equation", domain=[i,j])
    >>> equation[i,j] = Sum(Domain(i,j).where[i], a[i] + b[j])
    """

    def __init__(self, *sets: Union["Set", "Alias", "ImplicitSet"]) -> None:
        self._sanity_check(sets)
        self.sets = sets
        self.container = self._find_container()  # type: ignore
        self.where = condition.Condition(self)

    def _sanity_check(
        self, sets: Tuple[Union["Set", "Alias", "ImplicitSet"], ...]
    ):
        if len(sets) < 2:
            raise DomainException("Domain requires at least 2 sets")

        if all(not hasattr(set, "container") for set in sets):
            raise DomainException(
                "At least one of the sets in the domain must be a Set or Alias"
            )

    def _find_container(self):
        for set in self.sets:
            if hasattr(set, "container"):
                return set.container

    def gamsRepr(self) -> str:
        return utils._getDomainStr(self.sets)
