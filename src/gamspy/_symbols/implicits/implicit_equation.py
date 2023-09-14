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

from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union

import gamspy._algebra.expression as expression
import gamspy._symbols.alias as alias
import gamspy._symbols.implicits as implicits
import gamspy._symbols.set as gams_set
from gamspy._symbols.implicits.implicit_parameter import ImplicitParameter
from gamspy._symbols.implicits.implicit_symbol import ImplicitSymbol

if TYPE_CHECKING:
    from gamspy import Set, Equation
    from gamspy._algebra.expression import Expression


class ImplicitEquation(ImplicitSymbol):
    def __init__(
        self,
        parent: "Equation",
        name: str,
        type: str,
        domain: List[Union["Set", str]],
    ) -> None:
        """Implicit Equation

        Parameters
        ----------
        parent : Equation
        name : str
        domain : List[Set | str]
        """
        super().__init__(parent, name, domain)
        self.type = type

        # level, marginal, lower, upper, scale
        self._l, self._m, self._lo, self._up, self._s = self._init_attributes()
        self._stage = self._create_attr("stage")
        self._range = self._create_attr("range")
        self._slacklo = self._create_attr("slacklo")
        self._slackup = self._create_attr("slackup")
        self._slack = self._create_attr("slack")
        self._infeas = self._create_attr("infeas")

    def _create_attr(self, attr_name: str):
        return ImplicitParameter(self.parent, f"{self.gamsRepr()}.{attr_name}")

    def _init_attributes(self) -> tuple:
        level = self._create_attr("l")
        marginal = self._create_attr("m")
        lower = self._create_attr("lo")
        upper = self._create_attr("up")
        scale = self._create_attr("scale")
        return level, marginal, lower, upper, scale

    @property
    def l(self) -> ImplicitParameter:  # noqa: E741, E743
        return self._l

    @property
    def m(self) -> ImplicitParameter:
        return self._m

    @property
    def lo(self) -> ImplicitParameter:
        return self._lo

    @property
    def up(self) -> ImplicitParameter:
        return self._up

    @property
    def scale(self) -> ImplicitParameter:
        return self._s

    @property
    def stage(self) -> ImplicitParameter:
        return self._stage

    @property
    def range(self) -> ImplicitParameter:
        return self._range

    @property
    def slacklo(self) -> ImplicitParameter:
        return self._slacklo

    @property
    def slackup(self) -> ImplicitParameter:
        return self._slackup

    @property
    def slack(self) -> ImplicitParameter:
        return self._slack

    @property
    def infeas(self) -> ImplicitParameter:
        return self._infeas

    @property
    def expr(self) -> Optional["Expression"]:
        return self._expr  # pragma: no cover

    @expr.setter
    def expr(self, assignment: Optional["Expression"] = None) -> None:
        """Needed for scalar equations
        >>> eq..  sum(wh,build(wh)) =l= 1;
        >>> eq.expr = Sum(wh, build[wh]) <= 1
        """
        equation_map = {
            "nonbinding": "=n=",
            "external": "=x=",
            "cone": "=c=",
            "boolean": "=b=",
        }

        if assignment is not None and self.type in equation_map.keys():
            assignment._op_type = equation_map[self.type]

        statement = expression.Expression(
            implicits.ImplicitEquation(
                self.parent,
                name=self.name,
                type=self.type,
                domain=self.domain,
            ),
            "..",
            assignment,
        )

        self.container._addStatement(statement)
        self._expr = statement

        if self.container.debug:
            self.container._loadOnDemand()

    def gamsRepr(self) -> str:
        representation = f"{self.name}"
        if len(self.domain):
            set_strs = []
            for set in self.domain:
                if isinstance(
                    set, (gams_set.Set, alias.Alias, implicits.ImplicitSet)
                ):
                    set_strs.append(set.gamsRepr())
                elif isinstance(set, str):
                    if set == "*":
                        set_strs.append(set)
                    else:
                        set_strs.append('"' + set + '"')

            domain_str = "(" + ",".join(set_strs) + ")"

            representation += domain_str

        return representation
