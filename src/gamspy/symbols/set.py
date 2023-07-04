from typing import Any, List, Optional, Union, TYPE_CHECKING
import gams.transfer as gt
import pandas as pd
import gamspy._algebra._expression as expression
import gamspy._algebra._operable as operable
import gamspy._algebra._condition as condition
import gamspy.symbols._implicits as implicits
import gamspy.utils as utils

if TYPE_CHECKING:
    from gamspy import Container


class Set(gt.Set, operable.OperableMixin):
    def __init__(
        self,
        container: "Container",
        name: str,
        domain: Optional[List[Union[gt.Set, str]]] = None,
        is_singleton: bool = False,
        records: Optional[Any] = None,
        domain_forwarding: bool = False,
        description: str = "",
        uels_on_axes: bool = False,
    ):
        super().__init__(
            container,
            name,
            domain,
            is_singleton,
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

    # Set Attributes
    @property
    def pos(self):
        return expression.Expression(f"{self.name}", ".", "pos")

    @property
    def ord(self):
        return expression.Expression(f"{self.name}", ".", "ord")

    @property
    def off(self):
        return expression.Expression(f"{self.name}", ".", "off")

    @property
    def rev(self):
        return expression.Expression(f"{self.name}", ".", "rev")

    @property
    def uel(self):
        return expression.Expression(f"{self.name}", ".", "uel")

    @property
    def len(self):
        return expression.Expression(f"{self.name}", ".", "len")

    @property
    def tlen(self):
        return expression.Expression(f"{self.name}", ".", "tlen")

    @property
    def val(self):
        return expression.Expression(f"{self.name}", ".", "val")

    @property
    def tval(self):
        return expression.Expression(f"{self.name}", ".", "tval")

    @property
    def first(self):
        return expression.Expression(f"{self.name}", ".", "first")

    @property
    def last(self):
        return expression.Expression(f"{self.name}", ".", "last")

    def __iter__(self):
        assert self._records is not None, (
            f"Set {self.name} does not contain any records. Cannot iterate"
            " over a Set with no records"
        )

        if self._records is not None:
            return self._records.iterrows()

    def __getitem__(self, indices: Union[list, str]) -> implicits.ImplicitSet:
        domain = utils._toList(indices)
        return implicits.ImplicitSet(self.ref_container, name=self.name, domain=domain)

    def __setitem__(
        self,
        indices: Union[list, str],
        assignment,
    ):
        domain = utils._toList(indices)

        if isinstance(assignment, bool):
            assignment = "yes" if assignment is True else "no"  # type: ignore

        statement = expression.Expression(
            implicits.ImplicitSet(self.ref_container, name=self.name, domain=domain),
            "=",
            assignment,
        )

        self.ref_container._addStatement(statement)
        self._is_dirty = True

    def lag(self, n: int, type: str = "linear"):
        """Lag operation shifts the values of a Set or Alias by one to the left

        Parameters
        ----------
        n : int
        type : 'linear' or 'circular', optional

        Returns
        -------
        ImplicitSet

        Raises
        ------
        ValueError
            When type is not circular or linear
        """
        if type == "circular":
            return implicits.ImplicitSet(self.ref_container, name=f"{self.name} -- {n}")
        elif type == "linear":
            return implicits.ImplicitSet(self.ref_container, name=f"{self.name} - {n}")

        raise ValueError("Lag type must be linear or circular")

    def lead(self, n: int, type: str = "linear"):
        """Lead shifts the values of a Set or Alias by one to the right

        Parameters
        ----------
        n : int
        type : 'linear' or 'circular', optional

        Returns
        -------
        ImplicitSet

        Raises
        ------
        ValueError
            When type is not circular or linear
        """
        if type == "circular":
            return implicits.ImplicitSet(self.ref_container, name=f"{self.name} ++ {n}")
        elif type == "linear":
            return implicits.ImplicitSet(self.ref_container, name=f"{self.name} + {n}")

        raise ValueError("Lead type must be linear or circular")

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
        """Representation of this Set in GAMS language.

        Returns
        -------
        str
        """
        representation = self.name

        domain = []
        for set in self.domain:
            if isinstance(set, str):
                if set != "*":
                    domain.append('"' + set + '"')
            else:
                domain.append(set.name)

        if domain:
            domain_str = "(" + ",".join(domain) + ")"
            representation += domain_str

        return representation

    def getStatement(self) -> str:
        """Statement of the Set definition

        Returns
        -------
        str
        """
        output = f"Set {self.name}"

        if self._is_singleton:
            output = f"Singleton {output}"

        domain_str = ",".join(
            [set if isinstance(set, str) else set.name for set in self.domain]
        )
        output += f"({domain_str})"

        if self.description:
            output += f' "{self.description}"'

        records_str = " / "
        if self._records is not None:
            if self.domain is None or len(self.domain) <= 1:
                records_str += ",".join(self._records.iloc[:, 0].values.tolist())
            else:
                strings = (
                    self._records.to_string(index=False, header=False)
                    .strip()
                    .split("\n")
                )
                for string in strings:
                    records_str += "\n" + ".".join(string.split())

        output += records_str + " /"

        output += ";"

        return output
