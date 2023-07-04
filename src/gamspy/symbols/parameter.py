from typing import Any, Optional, Union, TYPE_CHECKING
import gams.transfer as gt
import gamspy._algebra._expression as expression
import gamspy._algebra._operable as operable
import gamspy._algebra._condition as condition
import gamspy.symbols._implicits as implicits
import gamspy.utils as utils
import pandas as pd

if TYPE_CHECKING:
    from gamspy import Container


class Parameter(gt.Parameter, operable.OperableMixin):
    def __init__(
        self,
        container: "Container",
        name: str,
        domain: Optional[list] = None,
        records: Optional[Any] = None,
        domain_forwarding: bool = False,
        description: str = "",
        uels_on_axes: bool = False,
    ):
        super().__init__(
            container,
            name,
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

    @property
    def assign(self):
        return self._assignment

    @assign.setter
    def assign(self, assignment):
        self._assignment = assignment

        self._is_dirty = True
        statement = expression.Expression(
            implicits.ImplicitParameter(self.ref_container, name=self.name),
            "=",
            assignment,
        )

        self.ref_container._addStatement(statement)

    def __iter__(self):
        assert self._records is not None, (
            f"Parameter {self.name} does not contain any records. Cannot"
            " iterate over a Parameter with no records"
        )

        if self._records is not None:
            return self._records.iterrows()

    def __getitem__(
        self, indices: Union[list, str]
    ) -> implicits.ImplicitParameter:
        domain = utils._toList(indices)
        return implicits.ImplicitParameter(
            self.ref_container, name=self.name, domain=domain
        )

    def __setitem__(
        self, indices: Union[list, str], assignment: expression.Expression
    ) -> None:
        domain = utils._toList(indices)

        statement = expression.Expression(
            implicits.ImplicitParameter(
                self.ref_container, name=self.name, domain=domain
            ),
            "=",
            assignment,
        )

        self.ref_container._addStatement(statement)

        self._is_dirty = True

    def __eq__(self, other):  # type: ignore
        return expression.Expression(self, "==", other)

    def __neg__(self):
        return implicits.ImplicitParameter(
            self.ref_container, name=f"-{self.name}", domain=self._domain
        )

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
        """Representation of this Parameter in GAMS language.

        Returns
        -------
        str
        """
        representation = self.name
        if self.domain:
            representation += utils._getDomainStr(self.domain)

        return representation

    def getStatement(self) -> str:
        """Statement of the Parameter definition

        Returns
        -------
        str
        """
        output = f"Parameter {self.gamsRepr()}"

        if self.description:
            output += ' "' + self.description + '"'

        records_str = " / "

        if self._records is not None:
            if self.is_scalar:
                # Parameter a(i) / 5.0 /;
                value = (
                    0 if self._records.empty else self._records.values[0][0]
                )
                records_str += str(value)
            else:
                # Parameter a(i) / i1 1\n i2 2\n /;
                for _, row in self._records.iterrows():
                    row_as_list = row.tolist()
                    label_str = ".".join(row_as_list[:-1])
                    records_str += "\n" + f"{label_str} {row_as_list[-1]}"

        records_str += " /"

        output += records_str + ";"

        return output
