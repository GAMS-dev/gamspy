from __future__ import annotations

from typing import TYPE_CHECKING

import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
import gamspy._symbols as syms
import gamspy._validation as validation
import gamspy.utils as utils
from gamspy._symbols.implicits.implicit_symbol import ImplicitSymbol
from gamspy._universe import UNIVERSE, Universe, is_universe, is_universe_domain

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import EllipsisType

    import numpy as np
    import pandas as pd

    from gamspy import Alias, Set
    from gamspy._algebra.expression import Expression


class ImplicitSet(ImplicitSymbol, operable.Operable):
    """
    Implicit Set

    Parameters
    ----------
    container : Container
    name : str
    domain : list[Set | str], optional
    """

    def __init__(
        self,
        parent: Set | Alias,
        name: str,
        domain: list[Set | Universe | str] | None = None,
        scalar_domains: list[tuple[int, Set]] | None = None,
    ) -> None:
        self.parent = parent
        if domain is None:
            domain = [UNIVERSE]

        super().__init__(name, domain, parent_scalar_domains=scalar_domains)

    def __ge__(self, other) -> Expression:
        return expression.Expression(self, ">=", other)

    def __le__(self, other) -> Expression:
        return expression.Expression(self, "<=", other)

    def __repr__(self) -> str:
        return f"ImplicitSet(parent={self.parent}, name='{self.name}', domain={self.domain}, parent_scalar_domains={self.parent_scalar_domains})"

    def __getitem__(
        self, indices: Sequence | str | EllipsisType | slice
    ) -> ImplicitSet:
        domain = validation.validate_domain(self, indices)
        return ImplicitSet(
            parent=self.parent,
            name=self.name,
            domain=domain,
            scalar_domains=self._scalar_domains,
        )

    @property
    def records(self) -> pd.DataFrame | None:
        if self.parent.records is None:
            return None

        temp_name = "autotemp" + utils._get_unique_name()
        temp_param = syms.Set._constructor_bypass(
            self.container, temp_name, self.parent.domain
        )
        domain = list(self.domain)
        for i, d in self._scalar_domains:
            domain.insert(i, d)

        temp_param[domain] = self
        del self.container._data[temp_name]
        return temp_param.records

    def toDense(self) -> np.ndarray:
        """
        Converts set membership to a dense numpy.array format.

        Members of the set are represented as ``1.0`` and non-members as
        ``0.0``. The shape of the returned array follows the domain of this
        implicit set: literal indices reduce the dimensionality accordingly.

        Returns
        -------
        np.ndarray
            A numpy array with the membership indicators. An array of zeros if
            the parent symbol has no records.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=["i1", "i2", "i3"])
        >>> j = gp.Set(m, "j", records=["j1", "j2"])
        >>> k = gp.Set(m, "k", domain=[i, j], records=[("i1", "j1"), ("i2", "j2")])
        >>> print(k[i, j].toDense())
        [[1. 0.]
         [0. 1.]
         [0. 0.]]
        >>> print(k[i, "j2"].toDense())
        [0. 1. 0.]

        """
        domain = list(self.domain)
        temp_name = "autotemp" + utils._get_unique_name()
        temp_param = syms.Parameter._constructor_bypass(
            self.container, temp_name, domain
        )

        try:
            temp_param[domain if domain else [...]] = self
            return temp_param.toDense()
        finally:
            del self.container._data[temp_name]

    def toValue(self) -> float:
        """
        Returns the membership of a scalar (fully indexed) set as a float.

        Returns ``1.0`` if the element is a member of the parent set and
        ``0.0`` otherwise.

        Returns
        -------
        float

        Raises
        ------
        TypeError
            If the implicit set is not scalar (all indices must be literals).

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=["i1", "i2", "i3"])
        >>> j = gp.Set(m, "j", records=["j1", "j2"])
        >>> k = gp.Set(m, "k", domain=[i, j], records=[("i1", "j1"), ("i2", "j2")])
        >>> k["i1", "j1"].toValue()
        np.float64(1.0)
        >>> k["i1", "j2"].toValue()
        np.float64(0.0)

        """
        domain = list(self.domain)
        temp_name = "autotemp" + utils._get_unique_name()
        temp_param = syms.Parameter._constructor_bypass(
            self.container, temp_name, domain
        )

        try:
            temp_param[domain if domain else [...]] = self
            return temp_param.toValue()
        finally:
            del self.container._data[temp_name]

    def toList(self, *, include_element_text: bool = False) -> list:
        """
        Converts the records of the implicit set to a Python list.

        Parameters
        ----------
        include_element_text : bool, optional
            If True, includes the element explanatory text in the output.
            Defaults to False.

        Returns
        -------
        list
            A list of the set elements. If the set has dimension > 1, the
            elements are returned as tuples. An empty list if the parent
            symbol has no records.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=["i1", "i2", "i3"])
        >>> j = gp.Set(m, "j", records=["j1", "j2"])
        >>> k = gp.Set(m, "k", domain=[i, j], records=[("i1", "j1"), ("i2", "j2")])
        >>> k[i, j].toList()
        [('i1', 'j1'), ('i2', 'j2')]
        >>> k[i, "j2"].toList()
        [('i2', 'j2')]

        """
        if self.parent.records is None:
            return []

        temp_name = "autotemp" + utils._get_unique_name()
        temp_set = syms.Set._constructor_bypass(
            self.container, temp_name, self.parent.domain
        )
        domain = list(self.domain)
        for i, d in self._scalar_domains:
            domain.insert(i, d)

        try:
            temp_set[domain] = self
            return temp_set.toList(include_element_text=include_element_text)
        finally:
            del self.container._data[temp_name]

    def latexRepr(self):
        name = self._latex_name
        representation = name

        domain = list(self.domain)

        for i, d in self._scalar_domains:
            domain.insert(i, d)

        if not is_universe_domain(domain):
            set_strs = []
            for elem in domain:
                if isinstance(elem, utils._get_domain_element_types()):
                    set_strs.append(elem.latexRepr())
                elif is_universe(elem):
                    set_strs.append(r"\text{`*'}")
                elif isinstance(elem, str):
                    elem = elem.replace("_", r"\_")
                    set_strs.append(f"\\text{{`{elem}'}}")

            domain_str = "{" + ",".join(set_strs) + "}"
            representation = f"{representation}_{domain_str}"

        return representation

    def gamsRepr(self) -> str:
        representation = self.name

        if not is_universe_domain(self.domain):
            domain = list(self.domain)
            for i, d in self._scalar_domains:
                domain.insert(i, d)

            representation += utils._get_domain_str(domain)

        return representation
