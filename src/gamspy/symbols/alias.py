import gams.transfer as gt
import gamspy._algebra._operable as operable
import gamspy._algebra._condition as condition
import gamspy.symbols._implicits as implicits
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gamspy import Set, Container


class Alias(gt.Alias, operable.OperableMixin):
    def __init__(self, container: "Container", name: str, alias_with: "Set"):
        super().__init__(container, name, alias_with)

        # enable load on demand
        self._is_dirty = False

        # allow conditions
        self.where = condition.Condition(self)

        # add statement
        self.ref_container._addStatement(self)

    def __iter__(self):
        assert self.alias_with._records is not None, (
            f"Alias {self.name} does not contain any records. Cannot iterate"
            " over an Alias with no records"
        )

        if self.alias_with._records is not None:
            return self.alias_with._records.iterrows()

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
            return implicits.ImplicitSet(
                self.ref_container, name=f"{self.name} -- {n}"
            )
        elif type == "linear":
            return implicits.ImplicitSet(
                self.ref_container, name=f"{self.name} - {n}"
            )

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
            return implicits.ImplicitSet(
                self.ref_container, name=f"{self.name} ++ {n}"
            )
        elif type == "linear":
            return implicits.ImplicitSet(
                self.ref_container, name=f"{self.name} + {n}"
            )

        raise ValueError("Lead type must be linear or circular")

    def gamsRepr(self) -> str:
        """Representation of this Alias in GAMS language.

        Returns
        -------
        str
        """
        representation = self.name

        domain = []
        for set in self.alias_with.domain:
            if isinstance(set, str):
                if set != "*":
                    domain.append(set)
            else:
                domain.append(set.name)

        if domain:
            domain_str = "(" + ",".join(domain) + ")"
            representation += f"{domain_str}"

        return representation

    def getStatement(self) -> str:
        """Statement of the Alias definition

        Returns
        -------
        str
        """
        return f"Alias({self.alias_with.name},{self.name});"
