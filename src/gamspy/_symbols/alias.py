from __future__ import annotations

import os
import threading
import weakref
from typing import TYPE_CHECKING, Any, cast

from gams.core.gdx import GMS_DT_ALIAS

import gamspy as gp
import gamspy._algebra.condition as condition
import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
import gamspy._symbols.implicits as implicits
import gamspy._validation as validation
import gamspy.utils as utils
from gamspy._symbols.base import BaseSymbol
from gamspy._symbols.set import SetMixin
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    import pandas as pd

    from gamspy import Alias, Container, Set
    from gamspy._algebra.expression import Expression
    from gamspy._algebra.operation import Operation
    from gamspy._symbols.implicits import ImplicitSet
    from gamspy._types import IndexType, NormalizedDomainType, SetRecordsType


class Alias(operable.Operable, BaseSymbol, SetMixin):
    """
    Represents an Alias symbol in GAMS.
    https://gamspy.readthedocs.io/en/latest/user/basics/alias.html

    Parameters
    ----------
    container : Container
        Container of the alias.
    name : str, optional
        Name of the alias.
    alias_with : Set
        Alias set object.

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i")
    >>> j = gp.Alias(m, "j", i)

    """

    @classmethod
    def _constructor_bypass(
        cls, container: Container, name: str, alias_with: Set | Alias
    ):
        # create new symbol object
        obj = object.__new__(cls)

        # legacy gtp attributes
        ## set private properties directly

        obj._container = cast(
            "Container",
            weakref.proxy(container)
            if not isinstance(container, weakref.ProxyType)
            else container,
        )
        obj.name = name
        obj._alias_with = alias_with

        ## typing
        obj._gams_type = GMS_DT_ALIAS
        obj._gams_subtype = 1

        ## add to container
        obj._container._data.update({name: obj})

        # gamspy attributes
        obj.where = condition.Condition(obj)
        obj._latex_name = name.replace("_", r"\_")
        obj.container._add_statement(obj)
        obj._metadata = {}

        return obj

    def __new__(
        cls,
        container: Container | None = None,
        name: str | None = None,
        alias_with: Set | Alias | None = None,
    ):
        if container is not None and not isinstance(container, gp.Container):
            raise TypeError(
                f"Container must of type `Container` but found {type(container)}"
            )

        if name is None:
            return object.__new__(cls)
        else:
            if not isinstance(name, str):
                raise TypeError(f"Name must of type `str` but found {type(name)}")
            try:
                if not container:
                    container = gp._ctx_managers[
                        (os.getpid(), threading.get_native_id())
                    ]

                symbol = container._data[name]
            except KeyError:
                return object.__new__(cls)

            if isinstance(symbol, cls):
                if id(symbol.alias_with) != id(alias_with):
                    raise ValueError(
                        "Redefinition of an Alias symbol with a different `alias_with` object is not allowed!"
                    )
                return symbol

            raise TypeError(
                f"Cannot overwrite symbol `{name}` in container"
                " because it is not an Alias object)"
            )

    def __init__(
        self,
        container: Container | None = None,
        name: str | None = None,
        alias_with: Set | Alias = None,  # type: ignore
    ):
        self._metadata: dict[str, Any] = {}
        self._assignment: Expression | None = None

        if container is None:
            try:
                container = gp._ctx_managers[(os.getpid(), threading.get_native_id())]
            except KeyError as e:
                raise ValidationError("Alias requires a container.") from e

        self._container = cast("Container", weakref.proxy(container))

        if name is not None:
            name = validation.validate_name(name)
        else:
            name = container._get_symbol_name(prefix="a")

        self.name = name

        # gtp attributes
        self._alias_with = self._validate_alias_with(alias_with)
        self._gams_type = GMS_DT_ALIAS
        self._gams_subtype = 1
        self._container._data.update({name: self})

        # gamspy attributes
        self._latex_name = self.name.replace("_", r"\_")
        self.where = condition.Condition(self)
        self._container._add_statement(self)
        self._container._synch_with_gams()

    @property
    def _should_unload_to_gams(self) -> bool:
        return self.alias_with._should_unload_to_gams

    @_should_unload_to_gams.setter
    def _should_unload_to_gams(self, value: bool) -> None:
        self.alias_with._should_unload_to_gams = value

    @property
    def _should_load_from_gams(self) -> bool:
        return self.alias_with._should_load_from_gams

    @_should_load_from_gams.setter
    def _should_load_from_gams(self, value: bool) -> None:
        self.alias_with._should_load_from_gams = value

    def _serialize(self) -> dict:
        info: dict[str, Any] = {"_metadata": self._metadata}
        if self._assignment is not None:
            info["_assignment"] = self._assignment.getDeclaration()

        return info

    def _deserialize(self, info: dict) -> None:
        for key, value in info.items():
            if key == "_assignment":
                left, right = value.split(" = ")
                value = expression.Expression(left, "=", right)

            setattr(self, key, value)

    def __bool__(self):
        raise ValidationError(
            "Alias cannot be used as a truth value. Use len(<symbol>.records) instead."
        )

    def __repr__(self) -> str:
        return f"Alias(name='{self.name}', alias_with={self.alias_with})"

    def __getitem__(self, indices: IndexType) -> ImplicitSet:
        domain = validation.validate_domain(self, indices)

        return implicits.ImplicitSet(self, name=self.name, domain=domain)

    def __setitem__(self, indices: IndexType, rhs: Expression | Operation | bool | str):
        # self[domain] = rhs
        domain = validation.validate_domain(self, indices)

        if isinstance(rhs, bool):
            rhs = "yes" if rhs is True else "no"

        statement = expression.Expression(
            implicits.ImplicitSet(self, name=self.name, domain=domain),
            "=",
            rhs,
        )

        # Cannot validate definition if we are in a gp.Loop since the control indices can be provided by the gp.Loop
        if not self._container._in_loop:
            statement._validate_definition(utils._unpack(domain))

        self._container._add_statement(statement)
        self._assignment = statement

        self.container._synch_with_gams()
        self._should_load_from_gams = True

    @property
    def records(self) -> pd.DataFrame | None:
        """
        Returns the main symbol records

        Returns
        -------
        DataFrame | None
            The main symbol records, None if no records were set
        """
        return self.alias_with.records

    @records.setter
    def records(self, records):
        self.alias_with.records = records

    def equals(
        self,
        other: Set | Alias,
        check_element_text: bool = True,
        check_meta_data: bool = True,
    ) -> bool:
        """
        Used to compare the symbol to another symbol.

        Parameters
        ----------
        other : Set or Alias
            The other symbol (Set or Alias) to compare with the current alias.
        check_element_text : bool, optional
            If True, check that all set elements have the same descriptive element text, otherwise skip.
        check_meta_data : bool, optional
            If True, check that symbol name and description are the same, otherwise skip.

        Returns
        -------
        bool
            True if the two symbols are equal in the specified aspects; False if they are not equal.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i")
        >>> j = gp.Alias(m, "j", i)
        >>> print(i.equals(j))  # Compare the Set 'i' with the Alias 'j'
        True

        """
        return self.alias_with.equals(
            other,
            check_element_text=check_element_text,
            check_meta_data=check_meta_data,
        )

    def toList(self, include_element_text: bool = False) -> list:
        """
        Convenience method to return symbol records as a python list

        Parameters
        ----------
        include_element_text : bool, optional
            If True, include the element text as tuples (record, element text).
            If False, return a list of records only.

        Returns
        -------
        list
            A list containing the records of the symbol.

        Examples
        --------
        >> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=["new-york", "chicago", "topeka"])
        >>> j = gp.Alias(m, "j", i)
        >>> print(j.toList())
        ['new-york', 'chicago', 'topeka']

        """
        return self.alias_with.toList(include_element_text=include_element_text)

    def pivot(
        self,
        index: list[str] | str | None = None,
        columns: list[str] | str | None = None,
        fill_value: int | float | None = None,
    ) -> pd.DataFrame | None:
        """
        Convenience function to pivot records into a new shape (only symbols with > 1D can be pivoted).

        Parameters
        ----------
        index : list[str] | str, optional
            If index is None then it is set to dimensions [0..dimension-1]
        columns : list[str] | str, optional
            If columns is None then it is set to the last dimension.
        fill_value : int | float, optional
            Missing values in the pivot will take the value provided by fill_value

        Returns
        -------
        DataFrame
            A new DataFrame containing the pivoted data.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=["seattle", "san-diego"])
        >>> j = gp.Set(m, "j", records=["new-york", "chicago", "topeka"])
        >>> ij = gp.Set(m, "ij", [i,j], records=[("seattle", "chicago"), ("seattle", "topeka"), ("san-diego", "new-york")])
        >>> routes = gp.Alias(m, name="routes", alias_with=ij)
        >>> print(routes.pivot(fill_value=""))  # doctest: +NORMALIZE_WHITESPACE
                  chicago topeka new-york
        seattle      True   True
        san-diego                    True

        """
        return self.alias_with.pivot(index, columns, fill_value)

    def getSparsity(self) -> float | None:
        """
        Gets the sparsity of the symbol w.r.t the cardinality

        Returns
        -------
        float | None
            Sparsity of an alias
        """
        return self.alias_with.getSparsity()

    @property
    def is_singleton(self) -> bool:
        """
        if symbol is a singleton set

        Returns
        -------
        bool
            True if the alias is singleton; False otherwise
        """
        return self.alias_with.is_singleton

    def _getUELCodes(self, dimension, ignore_unused=False):
        return self.alias_with._getUELCodes(dimension, ignore_unused=ignore_unused)

    def _getUELs(
        self,
        dimensions: list[int] | int | None = None,
        ignore_unused: bool = False,
    ) -> list[str]:
        return self.alias_with._getUELs(
            dimensions=dimensions, ignore_unused=ignore_unused
        )

    def _removeUELs(
        self,
        uels: list[str] | str | None = None,
        dimensions: list[int] | int | None = None,
    ) -> None:
        return self.alias_with._removeUELs(uels=uels, dimensions=dimensions)

    def _assert_valid_records(self):
        self.alias_with._assert_valid_records()

    @property
    def alias_with(self) -> Set:
        """
        Returns the aliased object

        Returns
        -------
        Set
            The aliased Set
        """
        return self._alias_with

    def _validate_alias_with(self, alias_with: Set | Alias) -> Set:
        from gamspy._symbols import Set, UniverseAlias

        if alias_with is None:
            raise ValueError("`alias_With` cannot be None.")

        if isinstance(alias_with, UniverseAlias):
            raise TypeError(
                "Cannot create an Alias to a UniverseAlias, create a new UniverseAlias symbol instead."
            )

        if not isinstance(alias_with, (Set, Alias)):
            raise TypeError("Symbol 'alias_with' must be type Set or Alias")

        if isinstance(alias_with, Alias):
            parent = alias_with
            while not isinstance(parent, Set):
                parent = parent.alias_with
            alias_with = parent

        return alias_with

    @property
    def domain_names(self) -> list[str]:
        """
        Returns the string version of domain names

        Returns
        -------
        list[str]
            A list of string version of domain names
        """
        return self.alias_with.domain_names

    @property
    def domain(self) -> NormalizedDomainType:
        """
        Returns list of domains given either as string (* for universe set) or as reference to the Set/Alias object

        Returns
        -------
        list[Set | str]
            A list of domains given either as string (* for universe set) or as reference to the Set/Alias object
        """
        return self.alias_with.domain

    @property
    def domain_type(self) -> str | None:
        """
        Returns the state of domain links

        Returns
        -------
        str
            none, relaxed or regular
        """
        return self.alias_with.domain_type

    @property
    def description(self) -> str:
        """
        Returns description of symbol

        Returns
        -------
        str
            Description of symbol
        """
        return self.alias_with.description

    @property
    def dimension(self) -> int:
        """
        Returns the dimension of symbol

        Returns
        -------
        int
            Dimension of symbol
        """
        return self.alias_with.dimension

    @property
    def number_records(self) -> int:
        """
        Returns the number of symbol records

        Returns
        -------
        int
            Number of symbol records
        """
        return self.alias_with.number_records

    @property
    def domain_labels(self) -> list[str] | None:
        """
        Returns the column headings for the records DataFrame

        Returns
        -------
        list[str] | None
            Column headings for the records DataFrame
        """
        return self.alias_with.domain_labels

    @domain_labels.setter
    def domain_labels(self, labels):
        self.alias_with.domain_labels = labels

    @property
    def summary(self) -> dict:
        """
        Returns a dict of only the metadata

        Returns
        -------
        dict
            Outputs a dict of only the metadata
        """
        return {
            "name": self.name,
            "description": self.description,
            "alias_with": self.alias_with.name,
            "is_singleton": self.is_singleton,
            "domain": self.domain_names,
            "domain_type": self.domain_type,
            "dimension": self.dimension,
            "number_records": self.number_records,
        }

    def _setRecords(
        self, records: SetRecordsType, *, uels_on_axes: bool = False
    ) -> None:
        self.alias_with.setRecords(records, uels_on_axes=uels_on_axes)

    def setRecords(self, records: SetRecordsType, uels_on_axes: bool = False) -> None:
        """
        Sets the records of the Set that is aliased.

        This is a convenience method to load data into the set. It handles various
        input formats like lists and pandas DataFrames.

        Parameters
        ----------
        records : pd.DataFrame | pd.Series | Sequence
            The data to load. Common formats:


            - List of strings: `['i1', 'i2']`
            - List of tuples (for multi-dimensional sets): `[('a', '1'), ('b', '2')]`
            - pandas DataFrame.
        uels_on_axes : bool, optional
            If True, assumes that the domain information is located in the axes
            (index/columns) of the `records` object rather than the data values.
            Use this when passing a DataFrame where the indices represent the set elements.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i")
        >>> j = gp.Alias(m, name="j", alias_with=i)
        >>> j.setRecords(["seattle", "san-diego"])
        >>> j.records.values.tolist()
        [['seattle', ''], ['san-diego', '']]

        """
        self.alias_with.setRecords(records, uels_on_axes)

    def gamsRepr(self) -> str:
        """
        Returns the string representation of this Alias in the GAMS language.

        (e.g., 'j').

        Returns
        -------
        str
            The GAMS string representation.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", domain=["*"], records=['i1','i2'])
        >>> j = gp.Alias(m, "j", i)
        >>> j.gamsRepr()
        'j'

        """
        return self.name

    def getDeclaration(self) -> str:
        """
        Returns the GAMS declaration statement for this Alias.

        (e.g., 'Alias(i, j);').

        Returns
        -------
        str
            The GAMS declaration string.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=['i1','i2'])
        >>> j = gp.Alias(m, "j", i)
        >>> j.getDeclaration()
        'Alias(i,j);'

        """
        return f"Alias({self.alias_with.name},{self.name});"

    def getAssignment(self) -> str:
        """
        Latest assignment to the Set in GAMS

        Returns
        -------
        str

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=['i1','i2'])
        >>> j = gp.Alias(m, "j", alias_with=i)
        >>> j['i1'] = False
        >>> j.getAssignment()
        'j("i1") = no;'

        """
        if self._assignment is None:
            raise ValidationError("Set was not assigned!")

        return self._assignment.getDeclaration()
