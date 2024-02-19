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

import itertools
from typing import Any
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union

import gams.transfer as gt
import pandas as pd
from gams.core.gdx import GMS_DT_PAR

import gamspy as gp
import gamspy._algebra.condition as condition
import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
import gamspy._symbols.implicits as implicits
import gamspy._validation as validation
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
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=['i1','i2'])
    >>> a = gp.Parameter(m, "a", [i], records=[['i1',1],['i2',2]])

    """

    @classmethod
    def _constructor_bypass(
        cls,
        container: Container,
        name: str,
        domain: List[Union[str, Set]] = [],
        records: Optional[Any] = None,
        description: str = "",
    ):
        obj = Parameter.__new__(
            cls, container, name, domain, records, description=description
        )

        # set private properties directly
        obj._requires_state_check = False
        obj._container = container
        container._requires_state_check = True
        obj._name = name
        obj._domain = domain
        obj._domain_forwarding = False
        obj._description = description
        obj._records = records
        obj._modified = True

        # typing
        obj._gams_type = GMS_DT_PAR
        obj._gams_subtype = 0

        # add to container
        container.data.update({name: obj})

        # gamspy attributes
        obj._is_dirty = False
        obj._is_frozen = False

        obj.where = condition.Condition(obj)
        obj.container._add_statement(obj)

        return obj

    def __new__(
        cls,
        container: Container,
        name: str,
        domain: Optional[List[Union[str, Set]]] = None,
        records: Optional[Any] = None,
        domain_forwarding: bool = False,
        description: str = "",
        uels_on_axes: bool = False,
    ):
        if not isinstance(container, gp.Container):
            raise TypeError(
                "Container must of type `Container` but found"
                f" {type(container)}"
            )

        if not isinstance(name, str):
            raise TypeError(f"Name must of type `str` but found {type(name)}")

        try:
            symbol = container[name]
            if isinstance(symbol, cls):
                return symbol

            raise TypeError(
                f"Cannot overwrite symbol `{name}` in container"
                " because it is not a Parameter object"
            )
        except KeyError:
            return object.__new__(cls)

    def __init__(
        self,
        container: Container,
        name: str,
        domain: Optional[List[Union[str, Set]]] = None,
        records: Optional[Any] = None,
        domain_forwarding: bool = False,
        description: str = "",
        uels_on_axes: bool = False,
    ):
        # domain handling
        if domain is None:
            domain = []

        if isinstance(domain, (gp.Set, gp.Alias, str)):
            domain = [domain]

        # does symbol exist
        has_symbol = False
        if isinstance(getattr(self, "container", None), gp.Container):
            has_symbol = True

        if has_symbol:
            if any(
                d1 != d2
                for d1, d2 in itertools.zip_longest(self.domain, domain)
            ):
                raise ValueError(
                    "Cannot overwrite symbol in container unless symbol"
                    " domains are equal"
                )

            if self.domain_forwarding != domain_forwarding:
                raise ValueError(
                    "Cannot overwrite symbol in container unless"
                    " 'domain_forwarding' is left unchanged"
                )

            # reset some properties
            self._requires_state_check = True
            self.container._requires_state_check = True
            if description != "":
                self.description = description
            self.records = None
            self.modified = True

            # only set records if records are provided
            if records is not None:
                self.setRecords(records, uels_on_axes=uels_on_axes)
        else:
            self._is_dirty = False
            self._is_frozen = False
            name = validation.validate_name(name)

            super().__init__(
                container,
                name,
                domain,
                domain_forwarding=domain_forwarding,
                description=description,
                uels_on_axes=uels_on_axes,
            )

            validation.validate_container(self, self.domain)
            self.where = condition.Condition(self)
            self.container._add_statement(self)

            if records is not None:
                self.setRecords(records, uels_on_axes=uels_on_axes)
            else:
                self.container._run()

    def __getitem__(
        self, indices: Union[tuple, str]
    ) -> implicits.ImplicitParameter:
        domain = validation.validate_domain(self, indices)

        return implicits.ImplicitParameter(self, name=self.name, domain=domain)

    def __setitem__(
        self,
        indices: Union[tuple, str, implicits.ImplicitSet],
        assignment: Union[Expression, float, int],
    ) -> None:
        domain = validation.validate_domain(self, indices)

        if isinstance(assignment, float):
            assignment = utils._map_special_values(assignment)  # type: ignore

        statement = expression.Expression(
            implicits.ImplicitParameter(self, name=self.name, domain=domain),
            "=",
            assignment,
        )

        self.container._add_statement(statement)
        self._assignment = statement

        self._is_dirty = True
        if not self.container.delayed_execution:
            self.container._run()

    def __eq__(self, other):  # type: ignore
        return expression.Expression(self, "eq", other)

    def __neg__(self):
        return implicits.ImplicitParameter(
            self, name=f"-{self.name}", domain=self._domain
        )

    @property
    def records(self):
        """
        Records of the Parameter

        Returns
        -------
        DataFrame
        """
        if not self._is_dirty:
            return self._records

        self.container._run()

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
                for symbol in self.container.data.values():
                    symbol._requires_state_check = True

    def setRecords(self, records: Any, uels_on_axes: bool = False) -> None:
        super().setRecords(records, uels_on_axes)
        self.container._run()

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
            statement_name += self._get_domain_str()

        output = f"Parameter {statement_name}"

        if self.description:
            output += ' "' + self.description + '"'

        output += ";"

        return output
