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
from typing import List
from typing import Literal
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
from gamspy.exceptions import GamspyException

if TYPE_CHECKING:
    from gamspy._symbols.implicits.implicit_set import ImplicitSet
    from gamspy import Alias, Container
    from gamspy._algebra.operable import Operable
    from gamspy._algebra.expression import Expression


class Set(gt.Set, operable.Operable, Symbol):
    """
    Represents a Set symbol in GAMS.
    https://www.gams.com/latest/docs/UG_SetDefinition.html

    Parameters
    ----------
    container : Container
    name : str
    domain : list, optional
    is_singleton : bool, optional
    records : int | float | DataFrame, optional
    domain_forwarding : bool, optional
    description : str, optional
    uels_on_axes : bool

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=['i1','i2'])

    """

    def __new__(cls, *args, **kwargs):
        try:
            name = kwargs["name"] if "name" in kwargs.keys() else args[1]
        except IndexError:
            raise GamspyException("Name of the symbol must be provided!")

        try:
            container = (
                kwargs["container"]
                if "container" in kwargs.keys()
                else args[0]
            )
        except IndexError:
            raise GamspyException("Container of the symbol must be provided!")
        try:
            symobj = container[name]
        except KeyError:
            symobj = None

        if symobj is None:
            return object.__new__(Set)
        else:
            if isinstance(symobj, Set):
                return symobj
            else:
                raise TypeError(
                    f"Cannot overwrite symbol `{symobj.name}` in container"
                    " because it is not a Set object)"
                )

    def __init__(
        self,
        container: "Container",
        name: str,
        domain: Optional[List[Union[Set, str]]] = None,
        is_singleton: bool = False,
        records: Optional[Any] = None,
        domain_forwarding: bool = False,
        description: str = "",
        uels_on_axes: bool = False,
    ):
        # enable load on demand
        self._is_dirty = False

        # allow conditions
        self.where = condition.Condition(self)

        # check if the name is a reserved word
        name = utils._reservedCheck(name)

        singleton_check(is_singleton, records)

        super().__init__(
            container,
            name,
            domain,
            is_singleton,
            records,
            domain_forwarding,
            description,
            uels_on_axes,
        )

        # add statement
        self.container._addStatement(self)

        # iterator index
        self._current_index = 0

    def __len__(self):
        if self.records is not None:
            return len(self.records.index)

        return 0

    def __next__(self):
        if self._current_index < len(self):
            row = self.records.iloc[self._current_index]
            self._current_index += 1
            return row

        self._current_index = 0
        raise StopIteration

    def __iter__(self):
        return self

    def __getitem__(self, indices: Union[tuple, str]) -> implicits.ImplicitSet:
        domain = self.domain if indices == ... else utils._toList(indices)
        return implicits.ImplicitSet(self, name=self.name, domain=domain)

    def __setitem__(
        self,
        indices: Union[tuple, str],
        assignment,
    ):
        domain = self.domain if indices == ... else utils._toList(indices)

        if isinstance(assignment, bool):
            assignment = "yes" if assignment is True else "no"  # type: ignore

        statement = expression.Expression(
            implicits.ImplicitSet(self, name=self.name, domain=domain),
            "=",
            assignment,
        )

        self.container._addStatement(statement)

        if self.container.delayed_execution:
            self._is_dirty = True
        else:
            self.container._run()

    # Set Attributes
    @property
    def pos(self):
        """
        Element position in the current set, starting with 1.

        Returns
        -------
        Expression
        """
        return expression.Expression(f"{self.name}", ".", "pos")

    @property
    def ord(self):
        """
        Same as .pos but for ordered sets only.

        Returns
        -------
        Expression
        """
        return expression.Expression(f"{self.name}", ".", "ord")

    @property
    def off(self):
        """
        Element position in the current set minus 1. So .off = .pos - 1

        Returns
        -------
        Expression
        """
        return expression.Expression(f"{self.name}", ".", "off")

    @property
    def rev(self):
        """
        Reverse element position in the current set, so the value for
        the last element is 0, the value for the penultimate is 1, etc.

        Returns
        -------
        Expression
        """
        return expression.Expression(f"{self.name}", ".", "rev")

    @property
    def uel(self):
        """
        Element position in the unique element list.

        Returns
        -------
        Expression
        """
        return expression.Expression(f"{self.name}", ".", "uel")

    @property
    def len(self):
        """
        Length of the set element name (a count of the number of characters).

        Returns
        -------
        Expression
        """
        return expression.Expression(f"{self.name}", ".", "len")

    @property
    def tlen(self):
        """
        Length of the set element text (a count of the number of characters).

        Returns
        -------
        Expression
        """
        return expression.Expression(f"{self.name}", ".", "tlen")

    @property
    def val(self):
        """
        If a set element is a number, this attribute gives the value of the number.
        For extended range arithmetic symbols, the symbols are reproduced.
        If a set element is a string that is not a number, then this attribute is
        not defined and trying to use it results in an error.

        Returns
        -------
        Expression
        """
        return expression.Expression(f"{self.name}", ".", "val")

    @property
    def tval(self):
        """
        If a set element text is a number, this attribute gives the value of the number.
        For extended range arithmetic symbols, the symbols are reproduced.
        If a set element text is a string that is not a number, then this attribute is
        not defined and trying to use it results in an error.

        Returns
        -------
        Expression
        """
        return expression.Expression(f"{self.name}", ".", "tval")

    @property
    def first(self):
        """
        Returns 1 for the first set element, otherwise 0.

        Returns
        -------
        Expression
        """
        return expression.Expression(f"{self.name}", ".", "first")

    @property
    def last(self):
        """
        Returns 1 for the last set element, otherwise 0.

        Returns
        -------
        Expression
        """
        return expression.Expression(f"{self.name}", ".", "last")

    def lag(
        self,
        n: Union[int, "Operable"],
        type: Literal["linear", "circular"] = "linear",
    ) -> "ImplicitSet":
        """
        Lag operation shifts the values of a Set or Alias by one to the left

        Parameters
        ----------
        n : int | Operable
        type : 'linear' or 'circular', optional

        Returns
        -------
        ImplicitSet

        Raises
        ------
        ValueError
            When type is not circular or linear

        Examples
        --------
        >>> import gamspy as gp
        >>>
        >>> m = gp.Container()
        >>> t = gp.Set(m, name="t", description="time sequence", records=[f"y-{x}" for x in range(1987, 1992)])
        >>> a = gp.Parameter(m, name="a", domain=[t])
        >>> b = gp.Parameter(m, name="b", domain=[t])
        >>>
        >>> a[t] = 1986 + gp.Ord(t)
        >>> b[t] = -1
        >>> b[t] = a[t.lag(1, "linear")]

        """
        jump = n if isinstance(n, int) else n.gamsRepr()  # type: ignore

        if type == "circular":
            return implicits.ImplicitSet(self, name=f"{self.name} -- {jump}")
        elif type == "linear":
            return implicits.ImplicitSet(self, name=f"{self.name} - {jump}")

        raise ValueError("Lag type must be linear or circular")

    def lead(
        self,
        n: Union[int, "Operable"],
        type: Literal["linear", "circular"] = "linear",
    ) -> "ImplicitSet":
        """
        Lead shifts the values of a Set or Alias by one to the right

        Parameters
        ----------
        n : int | Operable
        type : 'linear' or 'circular', optional

        Returns
        -------
        ImplicitSet

        Raises
        ------
        ValueError
            When type is not circular or linear

        Examples
        --------
        >>> import gamspy as gp
        >>>
        >>> m = gp.Container()
        >>> t = gp.Set(m, name="t", description="time sequence", records=[f"y-{x}" for x in range(1987, 1992)])
        >>> a = gp.Parameter(m, name="a", domain=[t])
        >>> c = gp.Parameter(m, name="c", domain=[t])
        >>>
        >>> a[t] = 1986 + gp.Ord(t)
        >>> c[t] = -1
        >>> c[t.lead(2, "linear")] = a[t]

        """
        jump = n if isinstance(n, int) else n.gamsRepr()  # type: ignore

        if type == "circular":
            return implicits.ImplicitSet(self, name=f"{self.name} ++ {jump}")
        elif type == "linear":
            return implicits.ImplicitSet(self, name=f"{self.name} + {jump}")

        raise ValueError("Lead type must be linear or circular")

    @property
    def records(self):
        """
        Records of the Set

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
                for symnam, symobj in self.container.data.items():
                    symobj._requires_state_check = True

    def sameAs(self, other: Union["Set", "Alias"]) -> "Expression":
        return expression.Expression(
            "sameAs(", ",".join([self.gamsRepr(), other.gamsRepr()]), ")"
        )

    def gamsRepr(self) -> str:
        """
        Representation of this Set in GAMS language.

        Returns
        -------
        str
        """
        return self.name

    def getStatement(self) -> str:
        """
        Statement of the Set definition

        Returns
        -------
        str
        """
        output = f"Set {self.name}"

        if self._is_singleton:
            output = f"Singleton {output}"

        domain_str = ",".join(
            [set if isinstance(set, str) else set.name for set in self.domain]
        )
        output += f"({domain_str})"

        if self.description:
            output += f' "{self.description}"'

        output += ";"

        return output


def singleton_check(is_singleton: bool, records: Union[Any, None]):
    if is_singleton:
        if records is not None and len(records) > 1:
            raise GamspyException(
                "Singleton set records size cannot be more than one."
            )
