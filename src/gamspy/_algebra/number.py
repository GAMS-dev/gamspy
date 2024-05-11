from __future__ import annotations

import gamspy._algebra.condition as condition
import gamspy._algebra.operable as operable


class Number(operable.Operable):
    """
    Needed for conditions on numbers.

    Parameters
    ----------
    value : int | float

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> k = gp.Set(m, "k", records=["1964-i","1964-ii","1964-iii","1964-iv"])
    >>> ki = gp.Set(m, name="ki", domain=[k], description="initial period")
    >>> ki[k] = gp.Number(1).where[gp.Ord(k) == 1]

    """

    def __init__(self, value: int | float):
        self._value = value
        self.where = condition.Condition(self)

    def gamsRepr(self) -> str:
        """
        Representation of this Number in GAMS language.

        Returns
        -------
        str
        """
        return f"{self._value}"
