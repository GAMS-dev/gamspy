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

import gamspy._algebra.condition as condition
import gamspy.utils as utils
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from gamspy import Set, Alias
    from gamspy._symbols.implicits import ImplicitSet


class Domain:
    """
    Domain class needed for where statements on multidimensional index list
    in operations

    Parameters
    ----------
    sets: tuple[Set | str]

    Examples
    --------
    >>> from gamspy import Container, Set, Ord, Card, Variable, Equation, Sum, Domain
    >>> m = Container()
    >>> X = Set(m, name="X", records=[f"I{i}" for i in range(1, 22)])
    >>> Y = Set(m, name="Y", records=[f"J{j}" for j in range(1, 22)])
    >>> inside = Set(m, name="inside", domain=[X, Y])
    >>> inside[X, Y].where[~((Ord(X) == 1) & (Ord(X) == Card(X)))] = True
    >>> f = Variable(m, name="f", domain=[X, Y], type="positive")
    >>> obj = Variable(m, name="obj")
    >>> objfun = Equation(m, name="objfun", type="regular")
    >>> objfun[...] = obj == Sum(Domain(X, Y).where[inside[X, Y]], f[X.lead(1), Y] - f[X, Y])

    """

    def __init__(self, *sets: Set | Alias | ImplicitSet) -> None:
        self._sanity_check(sets)
        self.sets = sets
        self.container = self._find_container()  # type: ignore
        self.where = condition.Condition(self)

    def _sanity_check(self, sets: tuple[Set | Alias | ImplicitSet, ...]):
        if len(sets) < 2:
            raise ValidationError("Domain requires at least 2 sets")

        if all(not hasattr(set, "container") for set in sets):
            raise ValidationError(
                "At least one of the sets in the domain must be a Set or Alias"
            )

    def _find_container(self):
        for set in self.sets:
            if hasattr(set, "container"):
                return set.container

    def gamsRepr(self) -> str:
        """
        Representation of this Domain in GAMS language.

        Returns
        -------
        str
        """
        return utils._get_domain_str(self.sets)
