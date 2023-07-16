import gams.transfer as gt
import gamspy._algebra._operable as operable
import gamspy._algebra._condition as condition
import gamspy._algebra._expression as expression
import gamspy._symbols._implicits as implicits
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:  # pragma: no cover
    from gamspy import Set, Container
    from gamspy._algebra._operable import Operable
    from gamspy._algebra._expression import Expression


class Alias(gt.Alias, operable.Operable):
    """
    Represents an Alias symbol in GAMS.
    https://www.gams.com/latest/docs/UG_SetDefinition.html#UG_SetDefinition_TheAliasStatementMultipleNamesForASet

    Parameters
    ----------
    container : Container
    name : str
    alias_with : Set

    Example
    ----------
    >>> m = gt.Container()
    >>> i = gt.Set(m, "i")
    >>> j = gt.Alias(m, "j", i)
    """

    def __init__(self, container: "Container", name: str, alias_with: "Set"):
        super().__init__(container, name, alias_with)

        # enable load on demand
        self._is_dirty = False

        # allow conditions
        self.where = condition.Condition(self)

        # add statement
        self.ref_container._addStatement(self)

        # iterator index
        self._current_index = 0

    def __len__(self):
        if self.records is not None:
            return len(self.records.index)

        return 0

    def __next__(self):
        if self._current_index < len(self):
            row = self.records.iloc[self._current_index]
            self._current_index += 1
            return row

        self._current_index = 0
        raise StopIteration

    def __iter__(self):
        return self

    def lag(self, n: Union[int, "Operable"], type: str = "linear"):
        """Lag operation shifts the values of a Set or Alias by one to the left

        Parameters
        ----------
        n : int | Operable
        type : 'linear' or 'circular', optional

        Returns
        -------
        ImplicitSet

        Raises
        ------
        ValueError
            When type is not circular or linear
        """
        jump = n if isinstance(n, int) else n.gamsRepr()  # type: ignore

        if type == "circular":
            return implicits.ImplicitSet(
                self.ref_container, name=f"{self.name} -- {jump}"
            )
        elif type == "linear":
            return implicits.ImplicitSet(
                self.ref_container, name=f"{self.name} - {jump}"
            )

        raise ValueError("Lag type must be linear or circular")

    def lead(self, n: Union[int, "Operable"], type: str = "linear"):
        """Lead shifts the values of a Set or Alias by one to the right

        Parameters
        ----------
        n : int | Operable
        type : 'linear' or 'circular', optional

        Returns
        -------
        ImplicitSet

        Raises
        ------
        ValueError
            When type is not circular or linear
        """
        jump = n if isinstance(n, int) else n.gamsRepr()  # type: ignore

        if type == "circular":
            return implicits.ImplicitSet(
                self.ref_container, name=f"{self.name} ++ {jump}"
            )
        elif type == "linear":
            return implicits.ImplicitSet(
                self.ref_container, name=f"{self.name} + {jump}"
            )

        raise ValueError("Lead type must be linear or circular")

    def sameAs(self, other: Union["Set", "Alias"]) -> "Expression":
        return expression.Expression(
            "sameAs(", ",".join([self.gamsRepr(), other.gamsRepr()]), ")"
        )

    def gamsRepr(self) -> str:
        """
        Representation of this Alias in GAMS language.

        Returns
        -------
        str
        """
        return self.name

    def getStatement(self) -> str:
        """
        Statement of the Alias definition

        Returns
        -------
        str
        """
        return f"Alias({self.alias_with.name},{self.name});"
