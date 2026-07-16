from __future__ import annotations

import os
import threading
import weakref
from typing import TYPE_CHECKING, Literal, cast

import pandas as pd
from gams.core import gdx
from gams.core.gdx import GMS_DT_ALIAS

import gamspy as gp
import gamspy._algebra.condition as condition
import gamspy._gdx as gdxio
import gamspy._validation as validation
from gamspy._symbols.base import BaseSymbol
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gamspy import Container

unique_name = gp.utils._get_unique_name()
TEMP_ALIAS_NAME = "autotemp_a" + unique_name
TEMP_SET_NAME = "autotemp_s" + unique_name
TEMP_GDX_OUT_NAME = "_" + unique_name + ".gdx"


class UniverseAlias(BaseSymbol):
    """
    Represents a UniverseAlias symbol in GAMS.

    Parameters
    ----------
    container : Container
    name : str

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> universe = gp.UniverseAlias(m)
    >>> universe.records
    Empty DataFrame
    Columns: [uni]
    Index: []
    >>> i = gp.Set(m, "i", records=['i1', 'i2'])
    >>> universe.records
      uni
    0  i1
    1  i2

    """

    @classmethod
    def _constructor_bypass(cls, container: Container, name: str) -> UniverseAlias:
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

        ## typing
        obj._gams_type = GMS_DT_ALIAS
        obj._gams_subtype = 0

        ## add to container
        obj._container._data.update({name: obj})

        # gamspy attributes
        obj.where = condition.Condition(obj)
        obj._latex_name = name.replace("_", r"\_")
        obj.container._add_statement(obj)

        return obj

    def __new__(cls, container: Container | None = None, name: str = "universe"):
        if container is not None and not isinstance(container, gp.Container):
            raise TypeError(
                f"Container must of type `Container` but found {type(container)}"
            )

        if not isinstance(name, str):
            raise TypeError(f"Name must of type `str` but found {type(name)}")

        try:
            if not container:
                container = gp._ctx_managers[(os.getpid(), threading.get_native_id())]

            symbol = container._data[name]
        except KeyError:
            return object.__new__(cls)

        if isinstance(symbol, cls):
            return symbol

        raise TypeError(
            f"Cannot overwrite symbol `{name}` in container"
            " because it is not a UniverseAlias object)"
        )

    def __init__(self, container: Container | None = None, name: str = "universe"):
        if name is not None:
            name = validation.validate_name(name)
        else:
            name = container._get_symbol_name(prefix="u")

        self.name = name

        if container is None:
            try:
                container = gp._ctx_managers[(os.getpid(), threading.get_native_id())]
            except KeyError as e:
                raise ValidationError("UniverseAlias requires a container.") from e

        self._container = cast("Container", weakref.proxy(container))

        # gtp attributes
        self._gams_type = gdx.GMS_DT_ALIAS
        self._gams_subtype = 0
        self._container._data.update({name: self})

        # gamspy attributes
        self._latex_name = self.name.replace("_", r"\_")
        self.where = condition.Condition(self)

        self._container._add_statement(self)

    @property
    def _should_unload_to_gams(self) -> bool:
        return False

    @_should_unload_to_gams.setter
    def _should_unload_to_gams(self, value: bool) -> None: ...

    @property
    def _should_load_from_gams(self) -> bool:
        return True

    @_should_load_from_gams.setter
    def _should_load_from_gams(self, value: bool) -> None: ...

    def _serialize(self) -> dict:
        return {}

    def _deserialize(self, info: dict) -> None: ...

    def __repr__(self) -> str:
        return f"UniverseAlias(name='{self.name}')"

    @property
    def is_singleton(self) -> bool:
        """
        Whether a symbol is a singleton set

        Returns
        -------
        bool
            Always False
        """
        return False

    @property
    def alias_with(self) -> str:
        """
        Returns aliased object

        Returns
        -------
        str
            Always "*"
        """
        return "*"

    @property
    def domain_names(self) -> list[str]:
        """
        Always ["*"] for universe alias

        Returns
        -------
        list[str]
            Always ["*"]
        """
        return ["*"]

    @property
    def domain_labels(self) -> list[str]:
        """
        Always ["uni"] for universe alias

        Returns
        -------
        list[str]
            Always ["uni"]
        """
        return ["uni"]

    @domain_labels.setter
    def domain_labels(self, value) -> None: ...

    @property
    def domain(self) -> Literal["*"]:
        """
        Always "*" for universe alias

        Returns
        -------
        Literal
            Always "*"
        """
        return "*"

    @property
    def description(self) -> str:
        """
        Always 'Aliased with *' for universe alias

        Returns
        -------
        str
            Always 'Aliased with *'
        """
        return "Aliased with *"

    @property
    def dimension(self) -> int:
        """
        Always 1 for universe alias

        Returns
        -------
        int
            Always 1
        """
        return 1

    def toList(self) -> list[str]:
        """
        Convenience method to return symbol records as a python list

        Returns
        -------
        list[str] | None
            A list of symbol records

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=["seattle", "san-diego"])
        >>> j = gp.Set(m, "j", records=["new-york", "chicago", "topeka"])
        >>> ij = gp.UniverseAlias(m, "ij")
        >>> print(ij.toList())
        ['seattle', 'san-diego', 'new-york', 'chicago', 'topeka']

        """
        records = self.records
        return records.set_index(records.columns[0]).index.to_list()

    @property
    def number_records(self) -> int:
        """
        Number of symbol records

        Returns
        -------
        int
            Number of symbol records
        """
        return len(self)

    @property
    def domain_type(self) -> str:
        """
        Always none for universe alias

        Returns
        -------
        str
            Always 'none'
        """
        return "none"

    def _getUELs(self, ignore_unused: bool = False) -> list[str]:
        """
        Gets UELs from the Container. Returns only UELs in the data if ignore_unused=True, otherwise return all UELs.

        Parameters
        ----------
        ignore_unused : bool, optional
            Whether to get all UELs or only used ones, by default False

        Returns
        -------
        list | None
            A list of UELs if the symbol is valid, otherwise None.
        """
        return self._container._getUELs(ignore_unused=ignore_unused)

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
            "alias_with": self.alias_with,
        }

    def getSparsity(self) -> float:
        """
        Get the sparsity of the symbol w.r.t the cardinality

        Returns
        -------
        float
            Always 0
        """
        return 0.0

    def addElements(self, elements: Iterable[str]) -> None:
        """
        This function enables adding elements to the universe set (*).

        Parameters
        ----------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> _ = gp.Set(m, "i", records=[f"i{idx}" for idx in range(1, 5)])
        >>> uni = gp.UniverseAlias(m, "uni")
        >>> print(uni.toList())
        ['i1', 'i2', 'i3', 'i4']
        >>> uni.addElements(["i5", "i6"])
        >>> print(uni.toList())
        ['i1', 'i2', 'i3', 'i4', 'i5', 'i6']

        """
        element_str = ", ".join(elements)
        self._container._add_statement(f"Set {TEMP_SET_NAME} / {element_str} /;")
        self._container._synch_with_gams()

    @property
    def records(self) -> pd.DataFrame:
        """
        Records of the UniverseAlias

        Returns
        -------
        list[str]

        Examples
        --------
        >>> import gamspy as gp
        >>> import numpy as np
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["seattle", "san-diego"])
        >>> uni = gp.UniverseAlias(m)
        >>> uni.records
                 uni
        0    seattle
        1  san-diego

        """
        global TEMP_ALIAS_NAME
        global TEMP_GDX_OUT_NAME

        temp_path = os.path.join(self._container.working_directory, TEMP_GDX_OUT_NAME)
        self._container._add_statement(f"Alias (*, {TEMP_ALIAS_NAME});")
        self._container._add_statement(
            f"execute_unload '{temp_path}' {TEMP_ALIAS_NAME};"
        )
        self._container._synch_with_gams()

        with gdxio.open_gdx(self._container.system_directory, temp_path) as handle:
            uels: list[str] = self._container._gams2np.gdxGetUelList(handle)

        return pd.DataFrame(uels[1:], columns=["uni"])

    @records.setter
    def records(self, value) -> None: ...

    def gamsRepr(self) -> str:
        """
        Representation of the UniverseAlias in GAMS language.

        Returns
        -------
        str

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.UniverseAlias(m, name="universe")
        >>> i.gamsRepr()
        'universe'

        """
        return self.name

    def latexRepr(self) -> str:
        """
        Representation of the UniverseAlias in LaTeX.

        Returns
        -------
        str

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.UniverseAlias(m, name="universe")
        >>> i.latexRepr()
        'universe'

        """
        return self._latex_name

    def getDeclaration(self) -> str:
        """
        Declaration of the UniverseAlias in GAMS

        Returns
        -------
        str

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.UniverseAlias(m, name="universe")
        >>> i.getDeclaration()
        'Alias(universe,*);'

        """
        return f"Alias({self.name},*);"
