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
from enum import Enum
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
from gamspy.exceptions import GamspyException

if TYPE_CHECKING:
    from gamspy import Set, Variable, Container
    from gamspy._algebra.operation import Operation
    from gamspy._algebra.expression import Expression


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
            return object.__new__(Equation)
        else:
            if isinstance(symobj, Equation):
                return symobj
            else:
                raise TypeError(
                    f"Cannot overwrite symbol `{symobj.name}` in container"
                    " because it is not an Equation object)"
                )

    def __init__(
        self,
        container: "Container",
        name: str,
        type: Union[str, EquationType] = "regular",
        domain: Optional[List[Union["Set", str]]] = None,
        definition: Optional[
            Union["Variable", "Operation", "Expression"]
        ] = None,
        records: Optional[Any] = None,
        domain_forwarding: bool = False,
        description: str = "",
        uels_on_axes: bool = False,
        definition_domain: Optional[list] = None,
    ):
        type = cast_type(type)

        # enable load on demand
        self._is_dirty = False

        # allow freezing
        self._is_frozen = False

        # check if the name is a reserved word
        name = utils._reservedCheck(name)

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

        # allow conditions
        self.where = condition.Condition(self)

        # add statement
        self.container._addStatement(self)

        # add defition if exists
        self._definition_domain = definition_domain
        self._add_definition(definition)

        # create attributes
        self._l, self._m, self._lo, self._up, self._s = self._init_attributes()
        self._stage = self._create_attr("stage")
        self._range = self._create_attr("range")
        self._slacklo = self._create_attr("slacklo")
        self._slackup = self._create_attr("slackup")
        self._slack = self._create_attr("slack")
        self._infeas = self._create_attr("infeas")

    def __hash__(self):
        return id(self)

    def __getitem__(self, indices: Union[tuple, str]):
        domain = self.domain if indices == ... else utils._toList(indices)
        return implicits.ImplicitEquation(
            self, name=self.name, type=self.type, domain=domain  # type: ignore  # noqa: E501
        )

    def __setitem__(
        self,
        indices: Union[tuple, str, implicits.ImplicitSet],
        assignment: "Expression",
    ):
        domain = self.domain if indices == ... else utils._toList(indices)

        non_regular_map = {
            "nonbinding": "=n=",
            "external": "=x=",
            "cone": "=c=",
            "boolean": "=b=",
        }

        regular_map = {
            "=e=": "eq",
            "=g=": "geq",
            "=l=": "leq",
        }

        eq_types = ["=e=", "=l=", "=g="]

        # In case of an MCP equation without any equality, add the equality
        if not any(eq_type in assignment.gamsRepr() for eq_type in eq_types):
            assignment = assignment == 0

        if self.type in non_regular_map.keys():  # type: ignore
            assignment._op_type = non_regular_map[self.type]  # type: ignore
        else:
            self.type = regular_map[assignment._op_type]

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

        self.container._addStatement(statement)
        self._definition = statement

        if self.container.delayed_execution:
            self._is_dirty = True

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
        )

    def _add_definition(
        self,
        assignment: Optional[
            Union["Variable", "Operation", "Expression"]
        ] = None,
    ) -> None:
        """
        Needed for scalar equations
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> wh = gp.Set(m, "wh", records=['i1', 'i2'])
        >>> build = gp.Parameter(m, "build", domain=[wh], records=[('i1',5), ('i2', 5)])
        >>> eq = gp.Equation(m, "eq")
        >>> eq[...] = gp.Sum(wh, build[wh]) <= 1

        """
        if assignment is None:
            self._definition = assignment
            return

        eq_types = ["=e=", "=l=", "=g="]

        # In case of an MCP equation without any equality, add the equality
        if not any(eq_type in assignment.gamsRepr() for eq_type in eq_types):
            assignment = assignment == 0

        non_regular_map = {
            "nonbinding": "=n=",
            "external": "=x=",
            "cone": "=c=",
            "boolean": "=b=",
        }

        regular_map = {
            "=e=": "eq",
            "=g=": "geq",
            "=l=": "leq",
        }

        if self.type in non_regular_map.keys():
            assignment._op_type = non_regular_map[self.type]  # type: ignore
        else:
            self.type = regular_map[assignment._op_type]  # type: ignore

        domain = self._definition_domain if self._definition_domain else []
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

        self.container._addStatement(statement)
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
    def stage(self):
        """
        Stage

        Returns
        -------
        ImplicitParameter
        """
        return self._stage

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
                for symnam, symobj in self.container.data.items():
                    symobj._requires_state_check = True

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
            domain_names = [set.name for set in self.domain]  # type: ignore
            domain_str = "(" + ",".join(domain_names) + ")"
            output += domain_str

        if self.description:
            output += ' "' + self.description + '"'

        output += ";"
        return output


def cast_type(type: Union[str, EquationType]) -> str:
    if isinstance(type, str):
        if type.upper() not in EquationType.values():
            raise ValueError(
                "Allowed equation types:"
                f" {EquationType.values()} but found {type}."
            )

        # assign eq by default
        if type.upper() == "REGULAR":
            type = "EQ"

    elif isinstance(type, EquationType):
        # assign eq by default
        type = "EQ" if EquationType.REGULAR else str(type)

    return type
