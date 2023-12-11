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

from typing import Any
from typing import TYPE_CHECKING

import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
import gamspy.utils as utils
from gamspy._symbols.implicits.implicit_symbol import ImplicitSymbol

if TYPE_CHECKING:
    from gamspy import Set, Parameter, Variable, Equation
    from gamspy._algebra.expression import Expression


class ImplicitParameter(ImplicitSymbol, operable.Operable):
    def __init__(
        self,
        parent: Parameter | Variable | Equation,
        name: str,
        domain: list[Set | str] = [],
        records: Any | None = None,
    ) -> None:
        """Implicit Parameter

        Parameters
        ----------
        parent : Parameter | Variable | Equation
        name : str
        domain : List[Set | str], optional
        records : Any, optional
        """
        super().__init__(parent, name, domain)
        self._records = records
        self._assignment = None

    def __neg__(self) -> ImplicitParameter:
        return ImplicitParameter(
            parent=self.parent,
            name=f"-{self.name}",
            domain=self.domain,
        )

    def __invert__(self):
        return expression.Expression("", "not", self)

    def __getitem__(self, indices: list | str) -> ImplicitParameter:
        domain = self.domain if indices == ... else utils._to_list(indices)
        return ImplicitParameter(
            parent=self.parent, name=self.name, domain=domain
        )

    def __setitem__(self, indices: list | str, assignment: Expression) -> None:
        domain = self.domain if indices == ... else utils._to_list(indices)

        statement = expression.Expression(
            ImplicitParameter(
                parent=self.parent, name=self.name, domain=domain
            ),
            "=",
            assignment,
        )

        self.container._add_statement(statement)

        self.parent._is_dirty = True
        if not self.container.delayed_execution:
            self.container._run(is_implicit=True)

    def __eq__(self, other):  # type: ignore
        return expression.Expression(self, "==", other)

    def __ne__(self, other):  # type: ignore
        return expression.Expression(self, "ne", other)

    def gamsRepr(self) -> str:
        """Representation of the parameter in GAMS syntax.

        Returns:
            str: String representation of the parameter in GAMS syntax.
        """
        representation = self.name
        if self.domain:
            representation += utils._get_domain_str(self.domain)

        return representation
