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

from enum import Enum
from typing import Any
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union

import gams.transfer as gt
import pandas as pd

import gamspy as gp
import gamspy._algebra.condition as condition
import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
import gamspy._symbols.implicits as implicits
import gamspy.utils as utils
from gamspy._symbols.symbol import Symbol

if TYPE_CHECKING:
    from gamspy import Set, Container


class VariableType(Enum):
    BINARY = "binary"
    INTEGER = "integer"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    FREE = "free"
    SOS1 = "sos1"
    SOS2 = "sos2"
    SEMICONT = "semicont"
    SEMIINT = "semiint"

    @classmethod
    def values(cls):
        return list(cls._value2member_map_.keys())

    def __str__(self) -> str:
        return self.value


class Variable(gt.Variable, operable.Operable, Symbol):
    """
    Represents a variable symbol in GAMS.
    https://www.gams.com/latest/docs/UG_Variables.html

    Parameters
    ----------
    container : Container
    name : str
    type : str, optional
    domain : list, optional
    records : DataFrame, optional
    domain_forwarding : bool, optional
    description : str, optional

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=['i1','i2'])
    >>> v = gp.Variable(m, "a", domain=[i])

    """

    def __new__(
        cls,
        container: Container,
        name: str,
        type: str = "free",
        domain: Optional[List[Union[str, Set]]] = None,
        records: Optional[Any] = None,
        domain_forwarding: bool = False,
        description: str = "",
        uels_on_axes: bool = False,
    ):
        if not isinstance(container, gp.Container):
            raise TypeError(
                f"Container must of type `Container` but found {container}"
            )

        if not isinstance(name, str):
            raise TypeError(f"Name must of type `str` but found {type(name)}")

        try:
            symbol = container[name]
            if isinstance(symbol, cls):
                return symbol
            else:
                raise TypeError(
                    f"Cannot overwrite symbol `{symbol.name}` in container"
                    " because it is not a Variable object)"
                )
        except KeyError:
            return object.__new__(cls)

    def __init__(
        self,
        container: Container,
        name: str,
        type: str = "free",
        domain: Optional[List[Union[str, Set]]] = None,
        records: Optional[Any] = None,
        domain_forwarding: bool = False,
        description: str = "",
        uels_on_axes: bool = False,
    ):
        type = cast_type(type)

        # enable load on demand
        self._is_dirty = False

        # allow freezing
        self._is_frozen = False

        # check if the name is a reserved word
        name = utils._reserved_check(name)

        super().__init__(
            container,
            name,
            type,
            domain,
            records,
            domain_forwarding,
            description,
            uels_on_axes,
        )

        self._container_check(self.domain)

        # allow conditions
        self.where = condition.Condition(self)

        # add statement
        self.container._add_statement(self)

        # create attributes
        self._l, self._m, self._lo, self._up, self._s = self._init_attributes()
        self._fx = self._create_attr("fx")
        self._prior = self._create_attr("prior")
        self._stage = self._create_attr("stage")

    def __getitem__(
        self, indices: Union[tuple, str]
    ) -> implicits.ImplicitVariable:
        domain = self.domain if indices == ... else utils._to_list(indices)
        return implicits.ImplicitVariable(self, name=self.name, domain=domain)

    def __neg__(self):
        return implicits.ImplicitVariable(
            self, name=f"-{self.name}", domain=self.domain
        )

    def __eq__(self, other):  # type: ignore
        return expression.Expression(self, "=e=", other)

    def _init_attributes(self):
        level = self._create_attr("l")
        marginal = self._create_attr("m")
        lower = self._create_attr("lo")
        upper = self._create_attr("up")
        scale = self._create_attr("scale")
        return level, marginal, lower, upper, scale

    def _create_attr(self, attr_name):
        domain = self.domain
        return implicits.ImplicitParameter(
            self,
            name=f"{self.name}.{attr_name}",
            records=self.records,
            domain=domain,
        )

    @property
    def l(self):  # noqa: E741,E743
        """
        Level

        Returns
        -------
        ImplicitParameter
        """
        return self._l

    @property
    def m(self):
        """
        Marginal

        Returns
        -------
        ImplicitParameter
        """
        return self._m

    @property
    def lo(self):
        """
        Lower bound

        Returns
        -------
        ImplicitParameter
        """
        return self._lo

    @property
    def up(self):
        """
        Upper bound

        Returns
        -------
        ImplicitParameter
        """
        return self._up

    @property
    def scale(self):
        """
        Scale

        Returns
        -------
        ImplicitParameter
        """
        return self._s

    @property
    def fx(self):
        """
        Fx

        Returns
        -------
        ImplicitParameter
        """
        return self._fx

    @property
    def prior(self):
        """
        Prior

        Returns
        -------
        ImplicitParameter
        """
        return self._prior

    @property
    def stage(self):
        """
        Stage

        Returns
        -------
        ImplicitParameter
        """
        return self._stage

    @property
    def records(self):
        """
        Records of the Variable

        Returns
        -------
        DataFrame
        """
        if not self._is_dirty:
            return self._records

        self.container._run(is_implicit=True)

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
                for _, symbol in self.container.data.items():
                    symbol._requires_state_check = True

    def gamsRepr(self) -> str:
        """
        Representation of this Variable in GAMS language.

        Returns
        -------
        str
        """
        return self.name

    def _get_domain_str(self):
        set_strs = []
        for set in self.domain:
            if isinstance(set, (gt.Set, gt.Alias, implicits.ImplicitSet)):
                set_strs.append(set.gamsRepr())
            elif isinstance(set, str):
                set_strs.append("*")

        return "(" + ",".join(set_strs) + ")"

    def getStatement(self) -> str:
        """
        Statement of the Variable definition

        Returns
        -------
        str
        """
        output = self.type + " "

        statement_name = self.name
        if self.domain:
            statement_name += self._get_domain_str()

        output += f"Variable {statement_name}"

        if self.description:
            output += ' "' + self.description + '"'

        output += ";"

        return output


def cast_type(type: Union[str, VariableType]) -> str:
    if isinstance(type, str) and type.lower() not in VariableType.values():
        raise ValueError(
            f"Allowed variable types: {VariableType.values()} but"
            f" found {type}."
        )

    if isinstance(type, VariableType):
        type = type.value

    return type.lower()
