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
    expr: Expression, optional
    records : Any, optional
    domain_forwarding : bool, optional
    description : str, optional
    uels_on_axes: bool
    expr_domain: list, optional

    Examples
    --------
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=['i1','i2'])
    >>> a = gp.Parameter(m, "a", [i], records=[['i1',1],['i2',2]])
    >>> e = gp.Equation(m, "e", [i])
    >>> e[i] = gp.Sum(i, a[i]) == a
    """

    def __init__(
        self,
        container: "Container",
        name: str,
        type: Union[str, EquationType] = "regular",
        domain: Optional[List[Union["Set", str]]] = None,
        expr: Optional[Union["Variable", "Operation", "Expression"]] = None,
        records: Optional[Any] = None,
        domain_forwarding: bool = False,
        description: str = "",
        uels_on_axes: bool = False,
        expr_domain: Optional[list] = None,
    ):
        type = self._cast_type(type)

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

        # enable load on demand
        self._is_dirty = False

        # allow freezing
        self._is_frozen = False

        # allow conditions
        self.where = condition.Condition(self)

        # add statement
        self.container._addStatement(self)

        # add defition if exists
        self._expr_domain = expr_domain
        self.expr = expr

        # create attributes
        self._l, self._m, self._lo, self._up, self._s = self._init_attributes()
        self._stage = self._create_attr("stage")
        self._range = self._create_attr("range")
        self._slacklo = self._create_attr("slacklo")
        self._slackup = self._create_attr("slackup")
        self._slack = self._create_attr("slack")
        self._infeas = self._create_attr("infeas")

    def _cast_type(self, type: Union[str, EquationType]) -> str:
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

    def __hash__(self):
        return id(self)

    def __getitem__(self, indices: Union[tuple, str]):
        domain = utils._toList(indices)
        return implicits.ImplicitEquation(
            self, name=self.name, type=self.type, domain=domain  # type: ignore  # noqa: E501
        )

    def __setitem__(
        self,
        indices: Union[tuple, str, implicits.ImplicitSet],
        assignment: "Expression",
    ):
        domain = utils._toList(indices)

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

        self._is_dirty = True
        if self.container.debug:
            self.container._loadOnDemand()

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

    @property
    def l(self):  # noqa: E741, E743
        return self._l

    @property
    def m(self):
        return self._m

    @property
    def lo(self):
        return self._lo

    @property
    def up(self):
        return self._up

    @property
    def scale(self):
        return self._s

    @property
    def stage(self):
        return self._stage

    @property
    def range(self):
        return self._range

    @property
    def slacklo(self):
        return self._slacklo

    @property
    def slackup(self):
        return self._slackup

    @property
    def slack(self):
        return self._slack

    @property
    def infeas(self):
        return self._infeas

    @property
    def expr(self) -> Optional[Union["Variable", "Operation", "Expression"]]:
        return self._expr

    @expr.setter
    def expr(
        self,
        assignment: Optional[
            Union["Variable", "Operation", "Expression"]
        ] = None,
    ) -> None:
        """
        Needed for scalar equations
        >>> eq..  sum(wh,build(wh)) =l= 1;
        >>> eq.expr = Sum(wh, build[wh]) <= 1
        """
        if assignment is None:
            self._expr = assignment
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

        domain = self._expr_domain if self._expr_domain else self.domain
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
        self._expr = statement

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
