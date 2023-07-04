#
# GAMS - General Algebraic Modeling System Python API
#
# Copyright (c) 2017-2023 GAMS Development Corp. <support@gams.com>
# Copyright (c) 2017-2023 GAMS Software GmbH <support@gams.com>
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

from typing import TYPE_CHECKING, Any, Optional, Union

import gamspy._algebra._condition as _condition
import gamspy._algebra._expression as _expression
import gamspy._algebra._operable as _operable
import gamspy.utils as utils

if TYPE_CHECKING:
    from gamspy import Container, Set


class ImplicitParameter(_operable.OperableMixin):
    def __init__(
        self,
        container: "Container",
        name: str,
        domain: list[Union["Set", str]] = [],
        records: Optional[Any] = None,
    ) -> None:
        """Implicit Parameter

        Parameters
        ----------
        container : Container
        name : str
        domain : Union[Set;, str], optional
        records : Any, optional
        """
        self.ref_container = container
        self.name = name
        self.domain = domain
        self._records = records
        self._assignment = None
        self.where = _condition.Condition(self)

    @property
    def assign(self):
        return self._records

    @assign.setter
    def assign(self, assignment) -> None:
        statement = _expression.Expression(
            ImplicitParameter(
                name=self.name, domain=self.domain, container=self.ref_container
            ),
            "=",
            assignment,
        )

        self.ref_container._addStatement(statement)

    def __neg__(self) -> ImplicitParameter:
        return ImplicitParameter(
            name=f"-{self.name}", domain=self.domain, container=self.ref_container
        )

    def __invert__(self):
        return _expression.Expression("", "not", self)

    def __getitem__(self, indices: Union[list, str]) -> ImplicitParameter:
        domain: list = utils._toList(indices)
        return ImplicitParameter(
            container=self.ref_container, name=self.name, domain=domain
        )

    def __setitem__(
        self, indices: Union[list, str], assignment: _expression.Expression
    ) -> None:
        domain: list = utils._toList(indices)

        statement = _expression.Expression(
            ImplicitParameter(
                name=self.name, domain=domain, container=self.ref_container
            ),
            "=",
            assignment,
        )

        self.ref_container._addStatement(statement)

    def __eq__(self, other):  # type: ignore
        return _expression.Expression(self, "==", other)

    def gamsRepr(self) -> str:
        """Representation of the parameter in GAMS syntax.

        Returns:
            str: String representation of the parameter in GAMS syntax.
        """
        representation = self.name
        if self.domain:
            representation += utils._getDomainStr(self.domain)

        return representation
