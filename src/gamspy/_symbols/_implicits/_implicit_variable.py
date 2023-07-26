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

import gamspy._algebra._operable as _operable
import gamspy._algebra._condition as _condition
import gamspy.utils as utils
import gamspy._algebra._expression as _expression
import gamspy._symbols._implicits as implicits
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from gams.transfer import Container
    from gams.transfer import Set


class ImplicitVariable(_operable.Operable):
    """
    Implicit Variable

    Parameters
    ----------
    container : Container
    name : str
    domain : List[Set | str]
    """

    def __init__(
        self,
        container: "Container",
        name: str,
        domain: list[Union["Set", str]],
    ):
        self.ref_container = container
        self.name = name
        self.domain = domain
        self._l, self._m, self._lo, self._up, self._s = self._init_attributes()
        self._fx = self._create_attr("fx")
        self._prior = self._create_attr("prior")
        self._stage = self._create_attr("stage")
        self.where = _condition.Condition(self)

    def _create_attr(self, attr_name: str):
        return implicits.ImplicitParameter(
            self.ref_container, f"{self.gamsRepr()}.{attr_name}"
        )

    def _init_attributes(self):
        level = self._create_attr("l")
        marginal = self._create_attr("m")
        lower = self._create_attr("lo")
        upper = self._create_attr("up")
        scale = self._create_attr("scale")
        return level, marginal, lower, upper, scale

    @property
    def l(self) -> implicits.ImplicitParameter:  # noqa: E741, E743
        return self._l

    @property
    def m(self) -> implicits.ImplicitParameter:
        return self._m

    @property
    def lo(self) -> implicits.ImplicitParameter:
        return self._lo

    @property
    def up(self) -> implicits.ImplicitParameter:
        return self._up

    @property
    def scale(self) -> implicits.ImplicitParameter:
        return self._s

    @property
    def fx(self) -> implicits.ImplicitParameter:
        return self._fx

    @property
    def prior(self) -> implicits.ImplicitParameter:
        return self._prior

    @property
    def stage(self) -> implicits.ImplicitParameter:
        return self._stage

    def __neg__(self):
        return implicits.ImplicitVariable(
            self.ref_container, name=f"-{self.name}", domain=self.domain
        )

    def __eq__(self, other):  # type: ignore
        return _expression.Expression(self, "=e=", other)

    def gamsRepr(self) -> str:
        representation = self.name
        if self.domain:
            representation += utils._getDomainStr(self.domain)

        return representation
