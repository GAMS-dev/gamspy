from __future__ import annotations

import gamspy._algebra.condition as condition
import gamspy._algebra.operable as operable
from gamspy._algebra import expression
from gamspy._special_values import SpecialValues


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
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"Number should be of type int or float but given value has type `{type(value)}`"
            )

        self._value: int | float | str = value
        self.domain: list = []
        self.where = condition.Condition(self)

    def __eq__(self, other):
        return expression.Expression(self, "=e=", other)

    def __ne__(self, other):
        return expression.Expression(self, "ne", other)

    def __hash__(self):
        return id(self)

    def __repr__(self) -> str:
        return f"Number(value={self._value})"

    def gamsRepr(self) -> str:
        """
        Representation of this Number in GAMS language.

        Returns
        -------
        str

        Examples
        --------
        >>> import gamspy as gp
        >>> gp.Number(2.0).gamsRepr()
        '2.0'

        """
        if isinstance(self._value, (int, float)):
            if SpecialValues.isEps(self._value):
                return "EPS"
            elif SpecialValues.isNA(self._value):
                return "NA"
            elif SpecialValues.isPosInf(self._value):
                return "INF"
            elif SpecialValues.isNegInf(self._value):
                return "(-INF)"
            elif SpecialValues.isUndef(self._value):
                return "UNDF"

            if self._value < 0:
                return f"({self._value})"

        return f"{self._value}"

    def latexRepr(self) -> str:
        """
        Representation of this Number in Latex.

        Returns
        -------
        str

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m)
        >>> print(i.latexRepr())
        i

        """
        return f"{self._value}"
