from typing import Any, List, Optional, Union, TYPE_CHECKING
import gams.transfer as gt
import pandas as pd
import gamspy._algebra._expression as expression
import gamspy._algebra._operable as operable
import gamspy._algebra._condition as condition
import gamspy.symbols._implicits as implicits
import gamspy.utils as utils

if TYPE_CHECKING:
    from gamspy import Set, Container


class Variable(gt.Variable, operable.OperableMixin):
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

    def __iter__(self):
        assert self._records is not None, (
            f"Variable {self.name} does not contain any records. Cannot"
            " iterate over a Variable with no records"
        )

        if self._records is not None:
            return self._records.iterrows()

    def __getitem__(self, indices: Union[list, str]) -> implicits.ImplicitVariable:
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

    @property
    def records(self):
        if not self._is_dirty:
            return self._records

        updated_records = self.ref_container._loadOnDemand(self.name)
        if updated_records is not None:
            self.records = updated_records.copy()
            self.domain_labels = self.domain_names
        else:
            self.records = updated_records

        self._is_dirty = False

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
            if self.domain_forwarding:
                self._domainForwarding()

                # reset state check flags for all symbols in the container
                for symnam, symobj in self.ref_container.data.items():
                    symobj._requires_state_check = True

    def gamsRepr(self) -> str:
        """Representation of this Variable in GAMS language.

        Returns
        -------
        str
        """
        representation = self.name
        if self.domain:
            representation += utils._getDomainStr(self.domain)

        return representation

    def getStatement(self) -> str:
        """Statement of the Variable definition

        Returns
        -------
        str
        """
        output = self.type + " "

        output += f"Variable {self.gamsRepr()}"

        if self.description:
            output += ' "' + self.description + '"'

        if self._records is not None:
            records_str = " / "
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
