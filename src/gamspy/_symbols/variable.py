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

from typing import Any, List, Optional, Union, TYPE_CHECKING
import gams.transfer as gt
import pandas as pd
import gamspy._algebra._expression as expression
import gamspy._algebra._operable as operable
import gamspy._algebra._condition as condition
import gamspy._symbols._implicits as implicits
import gamspy.utils as utils

if TYPE_CHECKING:
    from gamspy import Set, Container


class Variable(gt.Variable, operable.Operable):
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
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=['i1','i2'])
    >>> v = gp.Variable(m, "a", [i])
    """

    def __init__(
        self,
        container: "Container",
        name: str,
        type: str = "free",
        domain: Optional[List[Union[str, "Set"]]] = None,
        records: Optional[Any] = None,
        domain_forwarding: bool = False,
        description: str = "",
        uels_on_axes: bool = False,
    ):
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

        # allow conditions
        self.where = condition.Condition(self)

        # add statement
        self.ref_container._addStatement(self)

        # create attributes
        self._l, self._m, self._lo, self._up, self._s = self._init_attributes()
        self._fx = self._create_attr("fx")
        self._prior = self._create_attr("prior")
        self._stage = self._create_attr("stage")

    def __getitem__(
        self, indices: Union[list, str]
    ) -> implicits.ImplicitVariable:
        domain = utils._toList(indices)
        return implicits.ImplicitVariable(
            self.ref_container, name=self.name, domain=domain
        )

    def __neg__(self):
        return implicits.ImplicitVariable(
            self.ref_container, name=f"-{self.name}", domain=self.domain
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
        return implicits.ImplicitParameter(
            self.ref_container,
            name=f"{self.name}.{attr_name}",
            records=self.records,
        )

    @property
    def l(self):  # noqa: E741,E743
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
    def fx(self):
        return self._fx

    @property
    def prior(self):
        return self._prior

    @property
    def stage(self):
        return self._stage

    @property
    def records(self):
        if not self._is_dirty:
            return self._records

        self.ref_container._loadOnDemand()

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

        self.ref_container._requires_state_check = True
        self.ref_container.modified = True

        if self._records is not None:
            if self.domain_forwarding:  # pragma: no cover
                self._domainForwarding()

                # reset state check flags for all symbols in the container
                for symnam, symobj in self.ref_container.data.items():
                    symobj._requires_state_check = True

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
            statement_name += utils._getDomainStr(self.domain)

        output += f"Variable {statement_name}"

        if self.description:
            output += ' "' + self.description + '"'

        records_str = " / "
        if self._records is not None:
            col_mapping = {
                "level": "L",
                "marginal": "M",
                "lower": "LO",
                "upper": "UP",
                "scale": "scale",
            }

            if len(self.domain) == 0:
                for col, value in zip(
                    self._records.columns.tolist(),
                    self._records.values.tolist()[0],
                ):
                    # Discrete variables cannot have scale
                    if (
                        self.type.lower() in ["binary", "integer"]
                        and col.lower() == "scale"
                    ):
                        continue
                    else:
                        records_str += f"{col_mapping[col.lower()]} {value},"
                records_str = records_str[:-1]
            else:
                # domain, level, marginal, lower, upper, scale
                for _, row in self._records.iterrows():
                    row_as_list = row.tolist()
                    label_str = ".".join(row_as_list[: len(self.domain)])

                    for column_name in row.index.tolist():
                        # Discrete variables cannot have scale
                        if (
                            isinstance(column_name, str)
                            and column_name.lower() == "scale"
                            and self.type.lower()
                            in [
                                "binary",
                                "integer",
                            ]
                        ):
                            continue
                        else:
                            if column_name in col_mapping.keys():
                                records_str += f"\n{label_str}.{col_mapping[column_name.lower()]} {row[column_name]}"  # noqa: E501

        records_str += "/"

        output += records_str

        output += ";"

        return output
