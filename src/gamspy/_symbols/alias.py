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

from typing import Literal
from typing import TYPE_CHECKING
from typing import Union

import gams.transfer as gt

import gamspy as gp
import gamspy._algebra.condition as condition
import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
import gamspy._symbols.implicits as implicits
import gamspy.utils as utils
from gamspy._symbols.symbol import Symbol

if TYPE_CHECKING:
    from gamspy._symbols.implicits.implicit_set import ImplicitSet
    from gamspy import Set, Container
    from gamspy._algebra.expression import Expression


class Alias(gt.Alias, operable.Operable, Symbol):
    """
    Represents an Alias symbol in GAMS.
    https://www.gams.com/latest/docs/UG_SetDefinition.html#UG_SetDefinition_TheAliasStatementMultipleNamesForASet

    Parameters
    ----------
    container : Container
    name : str
    alias_with : Set

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i")
    >>> j = gp.Alias(m, "j", i)

    """

    def __new__(cls, container: Container, name: str, alias_with: Set):
        if not isinstance(container, gp.Container):
            raise TypeError(
                "Container must of type `Container` but found"
                f" {type(container)}"
            )

        if not isinstance(name, str):
            raise TypeError(f"Name must of type `str` but found {type(name)}")

        try:
            symobj = container[name]
            if isinstance(symobj, cls):
                return symobj
            else:
                raise TypeError(
                    f"Cannot overwrite symbol `{name}` in container"
                    " because it is not an Alias object)"
                )
        except KeyError:
            return object.__new__(Alias)

    def __init__(self, container: Container, name: str, alias_with: Set):
        # enable load on demand
        self._is_dirty = False

        # check if the name is a reserved word
        name = utils._reserved_check(name)

        super().__init__(container, name, alias_with)

        self._container_check(self.domain)

        # allow conditions
        self.where = condition.Condition(self)

        # add statement
        self.container._add_statement(self)

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

    # Set Attributes
    @property
    def pos(self):
        """
        Element position in the current set, starting with 1.

        Returns
        -------
        Expression
        """
        return expression.Expression(None, f"{self.name}.pos", None)

    @property
    def ord(self):
        """
        Same as .pos but for ordered sets only.

        Returns
        -------
        Expression
        """
        return expression.Expression(None, f"{self.name}.ord", None)

    @property
    def off(self):
        """
        Element position in the current set minus 1. So .off = .pos - 1

        Returns
        -------
        Expression
        """
        return expression.Expression(None, f"{self.name}.off", None)

    @property
    def rev(self):
        """
        Reverse element position in the current set, so the value for
        the last element is 0, the value for the penultimate is 1, etc.

        Returns
        -------
        Expression
        """
        return expression.Expression(None, f"{self.name}.rev", None)

    @property
    def uel(self):
        """
        Element position in the unique element list.

        Returns
        -------
        Expression
        """
        return expression.Expression(None, f"{self.name}.uel", None)

    @property
    def len(self):
        """
        Length of the set element name (a count of the number of characters).

        Returns
        -------
        Expression
        """
        return expression.Expression(None, f"{self.name}.len", None)

    @property
    def tlen(self):
        """
        Length of the set element text (a count of the number of characters).

        Returns
        -------
        Expression
        """
        return expression.Expression(None, f"{self.name}.tlen", None)

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
        return expression.Expression(None, f"{self.name}.val", None)

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
        return expression.Expression(None, f"{self.name}.tval", None)

    @property
    def first(self):
        """
        Returns 1 for the first set element, otherwise 0.

        Returns
        -------
        Expression
        """
        return expression.Expression(None, f"{self.name}.first", None)

    @property
    def last(self):
        """
        Returns 1 for the last set element, otherwise 0.

        Returns
        -------
        Expression
        """
        return expression.Expression(None, f"{self.name}.last", None)

    def lag(
        self,
        n: Union[int, Symbol, Expression],
        type: Literal["linear", "circular"] = "linear",
    ) -> ImplicitSet:
        """Lag operation shifts the values of a Set or Alias by one to the left

        Parameters
        ----------
        n : int | Symbol | Expression
        type : 'linear' or 'circular', optional

        Returns
        -------
        ImplicitSet

        Raises
        ------
        ValueError
            When type is not circular or linear
        """
        jump = n if isinstance(n, int) else n.gamsRepr()  # type: ignore

        if type == "circular":
            return implicits.ImplicitSet(self, name=f"{self.name} -- {jump}")
        elif type == "linear":
            return implicits.ImplicitSet(self, name=f"{self.name} - {jump}")

        raise ValueError("Lag type must be linear or circular")

    def lead(
        self,
        n: Union[int, Symbol, Expression],
        type: Literal["linear", "circular"] = "linear",
    ) -> ImplicitSet:
        """
        Lead shifts the values of a Set or Alias by one to the right

        Parameters
        ----------
        n : int | Symbol | Expression
        type : 'linear' or 'circular', optional

        Returns
        -------
        ImplicitSet

        Raises
        ------
        ValueError
            When type is not circular or linear
        """
        jump = n if isinstance(n, int) else n.gamsRepr()  # type: ignore

        if type == "circular":
            return implicits.ImplicitSet(self, name=f"{self.name} ++ {jump}")
        elif type == "linear":
            return implicits.ImplicitSet(self, name=f"{self.name} + {jump}")

        raise ValueError("Lead type must be linear or circular")

    def sameAs(self, other: Union[Set, Alias]) -> Expression:
        return expression.Expression(
            "sameAs(", ",".join([self.gamsRepr(), other.gamsRepr()]), ")"
        )

    def gamsRepr(self) -> str:
        """
        Representation of this Alias in GAMS language.

        Returns
        -------
        str
        """
        return self.name

    def getStatement(self) -> str:
        """
        Statement of the Alias definition

        Returns
        -------
        str
        """
        return f"Alias({self.alias_with.name},{self.name});"
