from __future__ import annotations

from typing import TYPE_CHECKING

import gamspy._symbols as syms
import gamspy._symbols.alias as alias
import gamspy._symbols.implicits as implicits
import gamspy._symbols.set as gams_set
import gamspy.utils as utils
from gamspy._symbols.implicits.implicit_symbol import ImplicitSymbol

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from gamspy import Equation, Set
    from gamspy._algebra.expression import Expression


class ImplicitEquation(ImplicitSymbol):
    def __init__(
        self,
        parent: Equation,
        name: str,
        type: str,
        domain: list[Set | str],
    ) -> None:
        """Implicit Equation

        Parameters
        ----------
        parent : Equation
        name : str
        domain : List[Set | str]
        """
        super().__init__(parent, name, domain)
        self.type = type

    def __repr__(self) -> str:
        return f"ImplicitEquation(parent={self.parent}, name='{self.name}', domain={self.domain}, type={self.type})"

    @property
    def l(self):
        return implicits.ImplicitParameter(
            self.parent,
            name=self.parent.l.name,
            domain=self.domain,
            scalar_domains=self._scalar_domains,
        )

    @l.setter
    def l(self, value: int | float | Expression):
        # b[t].l = 30 -> b.l[t] = 30
        self.parent.l[self.domain] = value

    @property
    def m(self):
        return implicits.ImplicitParameter(
            self.parent,
            name=self.parent.m.name,
            domain=self.domain,
            scalar_domains=self._scalar_domains,
        )

    @m.setter
    def m(self, value: int | float | Expression):
        # b[t].m = 30 -> b.m[t] = 30
        self.parent.m[self.domain] = value

    @property
    def lo(self):
        return implicits.ImplicitParameter(
            self.parent,
            name=self.parent.lo.name,
            domain=self.domain,
            scalar_domains=self._scalar_domains,
        )

    @lo.setter
    def lo(self, value: int | float | Expression):
        # b[t].lo = 30 -> b.lo[t] = 30
        self.parent.lo[self.domain] = value

    @property
    def up(self):
        return implicits.ImplicitParameter(
            self.parent,
            name=self.parent.up.name,
            domain=self.domain,
            scalar_domains=self._scalar_domains,
        )

    @up.setter
    def up(self, value: int | float | Expression):
        # b[t].up = 30 -> b.up[t] = 30
        self.parent.up[self.domain] = value

    @property
    def scale(self):
        return implicits.ImplicitParameter(
            self.parent,
            name=self.parent.scale.name,
            domain=self.domain,
            scalar_domains=self._scalar_domains,
        )

    @scale.setter
    def scale(self, value: int | float | Expression):
        # b[t].scale = 30 -> b.scale[t] = 30
        self.parent.scale[self.domain] = value

    @property
    def stage(self):
        return implicits.ImplicitParameter(
            self.parent,
            name=self.parent.stage.name,
            domain=self.domain,
            scalar_domains=self._scalar_domains,
        )

    @stage.setter
    def stage(self, value: int | float | Expression):
        # b[t].stage = 30 -> b.stage[t] = 30
        self.parent.stage[self.domain] = value

    @property
    def range(self):
        return self.parent.range

    @property
    def slacklo(self):
        return self.parent.slacklo

    @property
    def slackup(self):
        return self.parent.slackup

    @property
    def slack(self):
        return self.parent.slack

    @property
    def records(self) -> pd.DataFrame | None:
        if self.parent.records is None:
            return None

        temp_name = "autotemp" + utils._get_unique_name()
        temp_param = syms.Parameter._constructor_bypass(
            self.container, temp_name, self.parent.domain
        )
        domain = list(self.domain)
        for i, d in self._scalar_domains:
            domain.insert(i, d)

        temp_param[domain] = self.l
        del self.container._data[temp_name]

        recs = temp_param.records
        if recs is not None:
            columns = recs.columns.to_list()
            columns[columns.index("value")] = "level"
            recs.columns = columns

        return recs

    def toDense(self, column: str = "level") -> np.ndarray:
        """
        Converts the records to a dense numpy.array format.

        Parameters
        ----------
        column : str, optional
            The attribute to convert, by default "level". One of "level",
            "marginal", "lower", "upper" or "scale".

        Returns
        -------
        np.ndarray
            A numpy array with the records. An array of zeros if the parent
            symbol has no records.

        Examples
        --------
        >>> import numpy as np
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=["i1", "i2"])
        >>> j = gp.Set(m, "j", records=["j1", "j2", "j3"])
        >>> p = gp.Parameter(m, "p", domain=[i, j], records=np.array([[1, 2, 3], [4, 5, 6]]))
        >>> v = gp.Variable(m, "v", domain=[i, j])
        >>> e = gp.Equation(m, "e", domain=[i, j])
        >>> e[i, j] = v[i, j] <= p[i, j]
        >>> e.l[i, j] = p[i, j]
        >>> print(e[i, j].toDense())
        [[1. 2. 3.]
         [4. 5. 6.]]
        >>> print(e[i, "j2"].toDense())
        [2. 5.]

        """
        if not isinstance(column, str):
            raise TypeError("Argument 'column' must be type str")

        columns = {
            "level": "l",
            "marginal": "m",
            "lower": "lo",
            "upper": "up",
            "scale": "scale",
        }

        if column not in columns:
            raise TypeError(
                f"Argument 'column' must be one of the following: {list(columns)}"
            )

        return getattr(self, columns[column]).toDense()

    def toValue(self, column: str = "level") -> float:
        """
        Returns the numerical value of a scalar (fully indexed) implicit
        equation attribute.

        Parameters
        ----------
        column : str, optional
            The attribute to convert, by default "level". One of "level",
            "marginal", "lower", "upper" or "scale".

        Returns
        -------
        float

        Raises
        ------
        TypeError
            If the implicit equation is not scalar (all indices must be
            literals).

        Examples
        --------
        >>> import numpy as np
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=["i1", "i2"])
        >>> j = gp.Set(m, "j", records=["j1", "j2", "j3"])
        >>> p = gp.Parameter(m, "p", domain=[i, j], records=np.array([[1, 2, 3], [4, 5, 6]]))
        >>> v = gp.Variable(m, "v", domain=[i, j])
        >>> e = gp.Equation(m, "e", domain=[i, j])
        >>> e[i, j] = v[i, j] <= p[i, j]
        >>> e.l[i, j] = p[i, j]
        >>> e["i1", "j2"].toValue()
        np.float64(2.0)

        """
        if not isinstance(column, str):
            raise TypeError("Argument 'column' must be type str")

        columns = {
            "level": "l",
            "marginal": "m",
            "lower": "lo",
            "upper": "up",
            "scale": "scale",
        }

        if column not in columns:
            raise TypeError(
                f"Argument 'column' must be one of the following: {list(columns)}"
            )

        return getattr(self, columns[column]).toValue()

    def toList(self, columns: str | list[str] | None = None) -> list:
        """
        Converts the specified attributes of the implicit equation to a Python
        list.

        Parameters
        ----------
        columns : str | list[str] | None, optional
            The attribute column(s) to include (e.g., "level", "marginal",
            "lower", "upper", "scale"). If None, defaults to "level".

        Returns
        -------
        list
            A list of the requested attribute values. For scalars, a list of
            the attribute values. For multi-dimensional implicit equations, a
            list of tuples where domain indices are followed by the requested
            attributes. An empty list if the parent symbol has no records.

        Examples
        --------
        >>> import numpy as np
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=["i1", "i2"])
        >>> j = gp.Set(m, "j", records=["j1", "j2", "j3"])
        >>> p = gp.Parameter(m, "p", domain=[i, j], records=np.array([[1, 2, 3], [4, 5, 6]]))
        >>> v = gp.Variable(m, "v", domain=[i, j])
        >>> e = gp.Equation(m, "e", domain=[i, j])
        >>> e[i, j] = v[i, j] <= p[i, j]
        >>> e.l[i, j] = p[i, j]
        >>> e[i, "j2"].toList()
        [('i1', 2.0), ('i2', 5.0)]

        """
        if self.parent.records is None:
            return []

        domain = list(self.domain)
        temp_name = "autotemp" + utils._get_unique_name()
        temp_eqn = syms.Equation._constructor_bypass(
            self.container, temp_name, domain=domain
        )

        try:
            index = domain if domain else [...]
            temp_eqn.l[index] = self.l
            temp_eqn.m[index] = self.m
            temp_eqn.lo[index] = self.lo
            temp_eqn.up[index] = self.up
            temp_eqn.scale[index] = self.scale
            return temp_eqn.toList(columns)
        finally:
            del self.container._data[temp_name]

    def gamsRepr(self) -> str:
        representation = f"{self.name}"
        domain = list(self.domain)
        for i, d in self._scalar_domains:
            domain.insert(i, d)

        if len(domain):
            set_strs = []
            for set in domain:
                if isinstance(set, (gams_set.Set, alias.Alias, implicits.ImplicitSet)):
                    set_strs.append(set.gamsRepr())
                elif isinstance(set, str):
                    if set == "*":
                        set_strs.append(set)
                    else:
                        set_strs.append('"' + set + '"')

            domain_str = "(" + ",".join(set_strs) + ")"

            representation += domain_str

        return representation
