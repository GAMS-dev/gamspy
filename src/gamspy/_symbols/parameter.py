from typing import Any, Optional, Union, TYPE_CHECKING
import gams.transfer as gt
import gamspy._algebra._expression as expression
import gamspy._algebra._operable as operable
import gamspy._algebra._condition as condition
import gamspy._symbols._implicits as implicits
import gamspy.utils as utils
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from gamspy import Container


class Parameter(gt.Parameter, operable.OperableMixin):
    """
    Represents a parameter symbol in GAMS.
    https://www.gams.com/latest/docs/UG_DataEntry.html#UG_DataEntry_Parameters

    Parameters
    ----------
    container : Container
    name : str
    domain : list, optional
    records : Any, optional
    domain_forwarding : bool, optional
    description : str, optional
    uels_on_axes : bool

    Example
    ----------
    >>> m = gt.Container()
    >>> i = gt.Set(m, "i", records=['i1','i2'])
    >>> a = gt.Parameter(m, "a", [i], records=[['i1',1],['i2',2]])
    """

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
        self,
        indices: Union[tuple, str, implicits.ImplicitSet],
        assignment: expression.Expression,
    ) -> None:
        if len(self._domain) == 0:
            raise Exception(
                "Cannot perform an indexed assignment over a scalar Parameter."
                " Specify the domain of the parameter to perform an indexed"
                " asssignment"
            )

        if isinstance(indices, (tuple, str)):
            if isinstance(indices, str):
                indices = [indices]  # type: ignore

            if len(self._domain) != len(indices):
                raise Exception(
                    "Dimension of the symbol domain and the dimension of the"
                    " assignment indices must be the same!\nEquation"
                    f" dimension: {len(self._domain)}\nIndexed assignment"
                    f" dimension: {len(indices)}"
                )
        else:
            if len(self._domain) != len(indices.domain):
                raise Exception(
                    "Dimension of the symbol domain and the dimension of the"
                    " assignment indices must be the same!\nEquation"
                    f" dimension: {len(self._domain)}\nIndexed assignment"
                    f" dimension: {len(indices.domain)}"
                )

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
            if self.domain_forwarding:  # pragma: no cover
                self._domainForwarding()

                # reset state check flags for all symbols in the container
                for symnam, symobj in self.ref_container.data.items():
                    symobj._requires_state_check = True

    def gamsRepr(self) -> str:
        """
        Representation of this Parameter in GAMS language.

        Returns
        -------
        str
        """
        return self.name

    def getStatement(self) -> str:
        """
        Statement of the Parameter definition

        Returns
        -------
        str
        """
        statement_name = self.name
        if self.domain:
            statement_name += utils._getDomainStr(self.domain)

        output = f"Parameter {statement_name}"

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
