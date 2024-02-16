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
from enum import Enum
from typing import Any
from typing import TYPE_CHECKING

import gams.transfer as gt
import pandas as pd
from gams.core.gdx import GMS_DT_VAR
from gams.transfer._internals import (
    TRANSFER_TO_GAMS_VARIABLE_SUBTYPES,
)

import gamspy as gp
import gamspy._algebra.condition as condition
import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
import gamspy._symbols.implicits as implicits
import gamspy._validation as validation
from gamspy._symbols.symbol import Symbol

if TYPE_CHECKING:
    from gamspy import Set, Container
    from gamspy._algebra.expression import Expression


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
        """Convenience function to return all values of enum"""
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

    @classmethod
    def _constructor_bypass(
        cls,
        container: Container,
        name: str,
        type: str = "free",
        domain: list[str | Set] | None = None,
        records: Any | None = None,
        description: str = "",
    ):
        # create new symbol object
        obj = Variable.__new__(
            cls,
            container,
            name,
            type,
            domain,
            records,
            description=description,
        )

        # set private properties directly
        obj._type = type
        obj._gams_type = GMS_DT_VAR
        obj._gams_subtype = TRANSFER_TO_GAMS_VARIABLE_SUBTYPES[type]

        obj._requires_state_check = False
        obj._container = container
        container._requires_state_check = True
        obj._name = name
        obj._domain = domain
        obj._domain_forwarding = False
        obj._description = description
        obj._records = records
        obj._modified = True

        # add to container
        container.data.update({name: obj})

        # gamspy attributes
        obj.where = condition.Condition(obj)
        obj.container._add_statement(obj)
        obj._is_dirty = False
        obj._is_frozen = False

        # create attributes
        obj._l, obj._m, obj._lo, obj._up, obj._s = obj._init_attributes()
        obj._fx = obj._create_attr("fx")
        obj._prior = obj._create_attr("prior")
        obj._stage = obj._create_attr("stage")

        return obj

    def __new__(
        cls,
        container: Container,
        name: str,
        type: str = "free",
        domain: list[str | Set] | None = None,
        records: Any | None = None,
        domain_forwarding: bool = False,
        description: str = "",
        uels_on_axes: bool = False,
    ):
        if not isinstance(container, gp.Container):
            raise TypeError(
                f"Container must of type `Container` but found {container}"
            )

        if not isinstance(name, str):
            raise TypeError(f"Name must of type `str` but found {name}")

        try:
            symbol = container[name]
            if isinstance(symbol, cls):
                return symbol

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
        domain: list[str | Set] | None = None,
        records: Any | None = None,
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
            if self.type != type.casefold():
                raise TypeError(
                    "Cannot overwrite symbol in container unless variable"
                    f" types are equal: `{self.type}` !="
                    f" `{type.casefold()}`"
                )

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
            type = cast_type(type)
            self._is_dirty = False
            self._is_frozen = False
            name = validation.validate_name(name)

            super().__init__(
                container,
                name,
                type,
                domain,
                domain_forwarding=domain_forwarding,
                description=description,
                uels_on_axes=uels_on_axes,
            )

            validation.validate_container(self, self.domain)
            self.where = condition.Condition(self)
            self.container._add_statement(self)

            # create attributes
            self._l, self._m, self._lo, self._up, self._s = (
                self._init_attributes()
            )
            self._fx = self._create_attr("fx")
            self._prior = self._create_attr("prior")
            self._stage = self._create_attr("stage")

            if records is not None:
                self.setRecords(records, uels_on_axes=uels_on_axes)
            else:
                self.container._run()

    def __getitem__(self, indices: tuple | str) -> implicits.ImplicitVariable:
        domain = validation.validate_domain(self, indices)

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

    @l.setter
    def l(self, value: int | float | Expression):
        self._l[...] = value

    @property
    def m(self):
        """
        Marginal

        Returns
        -------
        ImplicitParameter
        """
        return self._m

    @m.setter
    def m(self, value: int | float | Expression):
        self._m[...] = value

    @property
    def lo(self):
        """
        Lower bound

        Returns
        -------
        ImplicitParameter
        """
        return self._lo

    @lo.setter
    def lo(self, value: int | float | Expression):
        self._lo[...] = value

    @property
    def up(self):
        """
        Upper bound

        Returns
        -------
        ImplicitParameter
        """
        return self._up

    @up.setter
    def up(self, value: int | float | Expression):
        self._up[...] = value

    @property
    def scale(self):
        """
        Scale

        Returns
        -------
        ImplicitParameter
        """
        return self._s

    @scale.setter
    def scale(self, value: int | float | Expression):
        self._s[...] = value

    @property
    def fx(self):
        """
        Fx

        Returns
        -------
        ImplicitParameter
        """
        return self._fx

    @fx.setter
    def fx(self, value: int | float | Expression):
        self._fx[...] = value

    @property
    def prior(self):
        """
        Prior

        Returns
        -------
        ImplicitParameter
        """
        return self._prior

    @prior.setter
    def prior(self, value: int | float | Expression):
        self._prior[...] = value

    @property
    def stage(self):
        """
        Stage

        Returns
        -------
        ImplicitParameter
        """
        return self._stage

    @stage.setter
    def stage(self, value: int | float | Expression):
        self._stage[...] = value

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
                for _, symbol in self.container.data.items():
                    symbol._requires_state_check = True

    def setRecords(self, records: Any, uels_on_axes: bool = False) -> None:
        super().setRecords(records, uels_on_axes)
        self.container._run()

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, var_type: str | VariableType):
        """
        The type of variable; [binary, integer, positive, negative, free, sos1, sos2, semicont, semiint]

        Parameters
        ----------
        var_type : str
            The type of variable
        """
        given_type = cast_type(var_type)
        gt.Variable.type.fset(self, given_type)

    def gamsRepr(self) -> str:
        """
        Representation of this Variable in GAMS language.

        Returns
        -------
        str
        """
        return self.name

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


def cast_type(type: str | VariableType) -> str:
    if isinstance(type, str) and type.lower() not in VariableType.values():
        raise ValueError(
            f"Allowed variable types: {VariableType.values()} but"
            f" found {type}."
        )

    if isinstance(type, VariableType):
        type = type.value

    return type.lower()
