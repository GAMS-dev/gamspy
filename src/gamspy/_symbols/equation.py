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
from gams.core.gdx import GMS_DT_EQU
from gams.transfer._internals import EQU_TYPE
from gams.transfer._internals import TRANSFER_TO_GAMS_EQUATION_SUBTYPES

import gamspy as gp
import gamspy._algebra.condition as condition
import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
import gamspy._symbols.implicits as implicits
import gamspy._validation as validation
from gamspy._symbols.symbol import Symbol

if TYPE_CHECKING:
    from gamspy import Set, Variable, Container
    from gamspy._algebra.operation import Operation
    from gamspy._algebra.expression import Expression


eq_types = ["=e=", "=l=", "=g="]

non_regular_map = {
    "nonbinding": "=n=",
    "external": "=x=",
    "cone": "=c=",
    "boolean": "=b=",
}


class EquationType(Enum):
    REGULAR = "REGULAR"
    NONBINDING = "NONBINDING"
    EXTERNAL = "EXTERNAL"
    CONE = "CONE"
    BOOLEAN = "BOOLEAN"

    @classmethod
    def values(cls):
        return list(cls._value2member_map_.keys())

    def __str__(self) -> str:
        return self.value


class Equation(gt.Equation, operable.Operable, Symbol):
    """
    Represents an Equation symbol in GAMS.
    https://www.gams.com/latest/docs/UG_Equations.html

    Parameters
    ----------
    container : Container
    name : str
    type : str
    domain : List[Set | str], optional
    definition: Expression, optional
    records : Any, optional
    domain_forwarding : bool, optional
    description : str, optional
    uels_on_axes: bool
    definition_domain: list, optional

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=['i1','i2'])
    >>> a = gp.Parameter(m, "a", [i], records=[['i1',1],['i2',2]])
    >>> v = gp.Variable(m, "v", domain=[i])
    >>> e = gp.Equation(m, "e", domain=[i])
    >>> e[i] = a[i] <= v[i]

    """

    @classmethod
    def _constructor_bypass(
        cls,
        container: Container,
        name: str,
        type: str | EquationType = "regular",
        domain: list[Set | str] | None = None,
        records: Any | None = None,
        description: str = "",
    ):
        # create new symbol object
        obj = Equation.__new__(
            cls,
            container,
            name,
            type,
            domain,
            records=records,
            description=description,
        )

        # set private properties directly
        type = cast_type(type)
        obj.type = EQU_TYPE[type]
        obj._gams_type = GMS_DT_EQU
        obj._gams_subtype = TRANSFER_TO_GAMS_EQUATION_SUBTYPES[type]
        obj._requires_state_check = False
        obj._container = container
        container._requires_state_check = True
        obj._name = name
        obj._domain = domain
        obj._domain_forwarding = False
        obj._description = description
        obj._records = None
        obj._modified = True

        # add to container
        container.data.update({name: obj})

        # gamspy attributes
        obj._is_dirty = False
        obj._is_frozen = False
        obj.where = condition.Condition(obj)
        obj.container._add_statement(obj)

        # create attributes
        obj._l, obj._m, obj._lo, obj._up, obj._s = obj._init_attributes()
        obj._stage = obj._create_attr("stage")
        obj._range = obj._create_attr("range")
        obj._slacklo = obj._create_attr("slacklo")
        obj._slackup = obj._create_attr("slackup")
        obj._slack = obj._create_attr("slack")
        obj._infeas = obj._create_attr("infeas")

        return obj

    def __new__(
        cls,
        container: Container,
        name: str,
        type: str | EquationType = "regular",
        domain: list[Set | str] | None = None,
        definition: Variable | Operation | Expression | None = None,
        records: Any | None = None,
        domain_forwarding: bool = False,
        description: str = "",
        uels_on_axes: bool = False,
        definition_domain: list | None = None,
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
                    f"Cannot overwrite symbol `{name}` in container"
                    " because it is not an Equation object)"
                )
        except KeyError:
            return object.__new__(cls)

    def __init__(
        self,
        container: Container,
        name: str,
        type: str | EquationType = "regular",
        domain: list[Set | str] | None = None,
        definition: Variable | Operation | Expression | None = None,
        records: Any | None = None,
        domain_forwarding: bool = False,
        description: str = "",
        uels_on_axes: bool = False,
        definition_domain: list | None = None,
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
            try:
                type = cast_type(type)
                if self.type != type.casefold():
                    raise TypeError(
                        "Cannot overwrite symbol in container unless equation"
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

            except ValueError as err:
                raise ValueError(err)

            except TypeError as err:
                raise TypeError(err)

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
                records,
                domain_forwarding,
                description,
                uels_on_axes,
            )

            validation.validate_container(self, self.domain)

            self.where = condition.Condition(self)
            self.container._add_statement(self)
            self._definition_domain = definition_domain
            self._init_definition(definition)

            # create attributes
            self._l, self._m, self._lo, self._up, self._s = (
                self._init_attributes()
            )
            self._stage = self._create_attr("stage")
            self._range = self._create_attr("range")
            self._slacklo = self._create_attr("slacklo")
            self._slackup = self._create_attr("slackup")
            self._slack = self._create_attr("slack")
            self._infeas = self._create_attr("infeas")

            self.container._run()

    def __hash__(self):
        return id(self)

    def __getitem__(self, indices: tuple | str):
        domain = validation.validate_domain(self, indices)

        return implicits.ImplicitEquation(
            self, name=self.name, type=self.type, domain=domain  # type: ignore  # noqa: E501
        )

    def __setitem__(
        self,
        indices: tuple | str | implicits.ImplicitSet,
        assignment: Expression,
    ):
        domain = validation.validate_domain(self, indices)

        self._set_definition(assignment, domain)
        self._is_dirty = True

        if not self.container.delayed_execution:
            self.container._run()

    def __eq__(self, other):  # type: ignore
        return expression.Expression(self, "=e=", other)

    def _init_attributes(self) -> tuple:
        level = self._create_attr("l")
        marginal = self._create_attr("m")
        lower = self._create_attr("lo")
        upper = self._create_attr("up")
        scale = self._create_attr("scale")
        return level, marginal, lower, upper, scale

    def _create_attr(self, attr_name):
        return implicits.ImplicitParameter(
            self,
            name=f"{self.name}.{attr_name}",
            records=self.records,
            domain=self.domain,
        )

    def _init_definition(
        self,
        assignment: Variable | Operation | Expression | None = None,
    ) -> None:
        if assignment is None:
            self._definition = None  # type: ignore
            return None

        domain = (
            self._definition_domain if self._definition_domain else self.domain
        )
        self._set_definition(assignment, domain)

    def _set_definition(self, assignment, domain):
        # In case of an MCP equation without any equality, add the equality
        if not any(eq_type in assignment.gamsRepr() for eq_type in eq_types):
            assignment = assignment == 0

        if self.type in non_regular_map.keys():
            assignment.replace_operator(non_regular_map[self.type])

        statement = expression.Expression(
            implicits.ImplicitEquation(
                self,
                name=self.name,
                type=self.type,
                domain=domain,
            ),
            "..",
            assignment,
        )

        self.container._add_statement(statement)
        self._definition = statement

    @property
    def l(self):  # noqa: E741, E743
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
    def range(self):
        """
        Range

        Returns
        -------
        ImplicitParameter
        """
        return self._range

    @property
    def slacklo(self):
        """
        Slack lower bound

        Returns
        -------
        ImplicitParameter
        """
        return self._slacklo

    @property
    def slackup(self):
        """
        Slack upper bound

        Returns
        -------
        ImplicitParameter
        """
        return self._slackup

    @property
    def slack(self):
        """
        Slack

        Returns
        -------
        ImplicitParameter
        """
        return self._slack

    @property
    def infeas(self):
        """
        Infeasability

        Returns
        -------
        ImplicitParameter
        """
        return self._infeas

    @property
    def records(self):
        """
        Records of the Equation

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
        Representation of this Equation in GAMS language.

        Returns
        -------
        str
        """
        return self.name

    def getStatement(self) -> str:
        """
        Statement of the Equation declaration

        Returns
        -------
        str
        """
        output = f"Equation {self.name}"

        if self.domain:
            output += self._get_domain_str()

        if self.description:
            output += ' "' + self.description + '"'

        output += ";"
        return output


def cast_type(type: str | EquationType) -> str:
    if isinstance(type, str):
        if type.upper() not in EquationType.values():
            raise ValueError(
                "Allowed equation types:"
                f" {EquationType.values()} but found {type}."
            )

        # assign eq by default
        if type.upper() == "REGULAR":
            type = "eq"

    elif isinstance(type, EquationType):
        # assign eq by default
        type = "eq" if EquationType.REGULAR else str(type)

    return type
