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
from typing import Any
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union

import gams.transfer as gt
import pandas as pd

import gamspy._algebra.condition as condition
import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
import gamspy._symbols.implicits as implicits
import gamspy.utils as utils
from gamspy._symbols.symbol import Symbol

if TYPE_CHECKING:
    from gamspy import Set, Container
    from gamspy._algebra.expression import Expression


class Parameter(gt.Parameter, operable.Operable, Symbol):
    """
    Represents a parameter symbol in GAMS.
    https://www.gams.com/latest/docs/UG_DataEntry.html#UG_DataEntry_Parameters

    Parameters
    ----------
    container : Container
    name : str
    domain : list, optional
    records : Any, optional
    domain_forwarding : bool, optional
    description : str, optional
    uels_on_axes : bool

    Examples
    --------
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=['i1','i2'])
    >>> a = gp.Parameter(m, "a", [i], records=[['i1',1],['i2',2]])
    """

    def __init__(
        self,
        container: "Container",
        name: str,
        domain: Optional[List[Union[str, "Set"]]] = None,
        records: Optional[Any] = None,
        domain_forwarding: bool = False,
        description: str = "",
        uels_on_axes: bool = False,
    ):
        super().__init__(
            container,
            name,
            domain,
            records,
            domain_forwarding,
            description,
            uels_on_axes,
        )

        # enable load on demand
        self._is_dirty = False

        # allow freezing
        self._is_frozen = False

        # allow conditions
        self.where = condition.Condition(self)

        # add statement
        self.container._addStatement(self)

    def __getitem__(
        self, indices: Union[tuple, str]
    ) -> implicits.ImplicitParameter:
        domain = utils._toList(indices)
        return implicits.ImplicitParameter(self, name=self.name, domain=domain)

    def __setitem__(
        self,
        indices: Union[tuple, str, implicits.ImplicitSet],
        assignment: "Expression",
    ) -> None:
        domain = utils._toList(indices)

        statement = expression.Expression(
            implicits.ImplicitParameter(self, name=self.name, domain=domain),
            "=",
            assignment,
        )

        self.container._addStatement(statement)

        self._is_dirty = True
        if self.container.debug:
            self.container._loadOnDemand()

    def __eq__(self, other):  # type: ignore
        return expression.Expression(self, "==", other)

    def __neg__(self):
        return implicits.ImplicitParameter(
            self, name=f"-{self.name}", domain=self._domain
        )

    @property
    def assign(self):
        return self._assignment

    @assign.setter
    def assign(self, assignment):
        self._assignment = assignment

        self._is_dirty = True
        statement = expression.Expression(
            implicits.ImplicitParameter(self, name=self.name),
            "=",
            assignment,
        )

        self.container._addStatement(statement)

        if self.container.debug:
            self.container._loadOnDemand()

    @property
    def records(self):
        if not self._is_dirty:
            return self._records

        self.container._loadOnDemand()

        return self._records

    @records.setter
    def records(self, records):
        if records is not None:
            if not isinstance(records, pd.DataFrame):
                raise TypeError("Symbol 'records' must be type DataFrame")

        # set records
        self._records = records

        self._requires_state_check = True
        self.modified = True

        self.container._requires_state_check = True
        self.container.modified = True

        if self._records is not None:
            if self.domain_forwarding:  # pragma: no cover
                self._domainForwarding()

                # reset state check flags for all symbols in the container
                for symnam, symobj in self.container.data.items():
                    symobj._requires_state_check = True

    def gamsRepr(self) -> str:
        """
        Representation of this Parameter in GAMS language.

        Returns
        -------
        str
        """
        return self.name

    def getStatement(self) -> str:
        """
        Statement of the Parameter definition

        Returns
        -------
        str
        """
        statement_name = self.name
        if self.domain:
            statement_name += utils._getDomainStr(self.domain)

        output = f"Parameter {statement_name}"

        if self.description:
            output += ' "' + self.description + '"'

        output += ";"

        return output
