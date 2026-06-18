from __future__ import annotations

import atexit
import copy
import os
import platform
import re
import signal
import sys
import tempfile
import threading
import traceback
import warnings
import weakref
from collections.abc import Iterable
from difflib import get_close_matches
from pathlib import Path
from typing import TYPE_CHECKING, TextIO, cast, no_type_check

import gams.transfer as gt
import pandas as pd

import gamspy as gp
import gamspy._gdx as gdxio
import gamspy._miro as miro
import gamspy._validation as validation
import gamspy.utils as utils
from gamspy._backend.backend import backend_factory
from gamspy._communication import close_connection, get_connection, open_connection
from gamspy._config import get_option
from gamspy._extrinsic import ExtrinsicLibrary
from gamspy._internals import (
    ATTR_PREFIX,
    EQU_TYPE,
    TRANSFER_TO_GAMS_EQUATION_SUBTYPES,
    TRANSFER_TO_GAMS_VARIABLE_SUBTYPES,
    CasePreservingDict,
)
from gamspy._miro import MiroJSONEncoder
from gamspy._model import Problem, Sense
from gamspy._options import write_solver_options
from gamspy._workspace import Workspace
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any, Literal

    from pandas import DataFrame

    from gamspy import (
        Alias,
        Equation,
        EquationType,
        Model,
        Parameter,
        Set,
        UniverseAlias,
        Variable,
    )
    from gamspy._algebra.expression import Expression
    from gamspy._algebra.operation import Operation
    from gamspy._options import Options
    from gamspy._symbols.implicits import ImplicitVariable
    from gamspy._types import (
        DomainType,
        ParameterRecordsType,
        SetRecordsType,
        SymbolType,
        VarEquRecordsType,
    )

LOOPBACK = "127.0.0.1"
IS_MIRO_INIT = int(os.getenv("MIRO", 0))
MIRO_GDX_IN = os.getenv("GAMS_IDC_GDX_INPUT", None)
MIRO_GDX_OUT = os.getenv("GAMS_IDC_GDX_OUTPUT", None)
is_windows = platform.system() == "Windows"


def add_sysdir_to_path(system_directory: str) -> None:
    if is_windows:
        if "PATH" in os.environ:
            if not os.environ["PATH"].startswith(system_directory + os.pathsep):
                os.environ["PATH"] = system_directory + os.pathsep + os.environ["PATH"]
        else:
            os.environ["PATH"] = system_directory


def get_system_directory(system_directory: str | os.PathLike | None) -> str:
    if isinstance(system_directory, os.PathLike):
        system_directory = os.fspath(system_directory)

    if system_directory is not None:
        gams_path = os.path.join(system_directory, "gams")
        if is_windows:
            gams_path = f"{gams_path}.exe"

        if not os.path.exists(gams_path):
            raise ValidationError(
                f"`{system_directory}` is not a valid GAMS system directory."
            )
        return system_directory

    return get_option("GAMS_SYSDIR")


class Container:
    """
    Central workspace for building, modifying, executing, and exchanging data
    with GAMS models.

    https://gamspy.readthedocs.io/en/latest/reference/gamspy._container.html


    A :class:`Container` owns all GAMSPy symbols (sets, parameters, variables,
    equations, models) and manages synchronization with the GAMS execution
    engine.


    A container can:


    * Create and store symbols
    * Load symbols and records from GDX/G00 files
    * Synchronize modified symbols with GAMS
    * Generate and persist GAMS code and data files
    * Act as a context manager.

    Parameters
    ----------
    load_from : str, os.PathLike, Container, gt.Container, optional
        Source to initialize the container from:

        * ``.gdx`` file: loads symbols and records from a GDX file.
        * ``.g00`` file: restarts from a GAMS save file.
        * Container: Copies symbols from the given container into the new container.
    system_directory : str, os.PathLike, optional
        Path to the directory that holds the GAMS installation, by default None
    working_directory : str, os.PathLike, optional
        Directory used for temporary files (``.gms``, ``.lst``, ``.gdx``).
        If omitted, a temporary directory is created.
    debugging_level : {"keep", "keep_on_error", "delete"}, optional
        Controls whether temporary files are retained:


        * ``"keep"``: Keep files and generated GAMS code.
        * ``"keep_on_error"``: Keep files only on errors (default).
        * ``"delete"``: Always clean up.
    options : Options, optional
        Global options for the overall execution
    output : TextIO, optional
        Stream to which GAMS output is written.

    Examples
    --------
    Basic usage


    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=["i1", "i2"])


    Loading from an existing GDX


    >>> m.write("data.gdx")
    >>> m2 = gp.Container(load_from="data.gdx")
    >>> list(m2.data.keys()) == list(m.data.keys())
    True


    Using as a context manager


    >>> with gp.Container() as m:
    ...     i = gp.Set(m, "i")
    ...     j = gp.Set(m, "j", domain=i)

    """

    def __init__(
        self,
        load_from: str | os.PathLike | Container | gt.Container | None = None,
        system_directory: str | os.PathLike | None = None,
        working_directory: str | os.PathLike | None = None,
        debugging_level: Literal["keep", "keep_on_error", "delete"] = "keep_on_error",
        options: Options | None = None,
        output: TextIO | None = None,
    ):
        import gams.core.numpy as gnp

        self._comm_pair_id = utils._get_unique_name()
        self.output = output
        self._gams_string = ""
        self._in_loop: int = 0
        self._last_control_flow: (
            Literal["loop", "for", "while", "if", "elseif", "else"] | None
        ) = None
        self._arbitrary_code_executed: bool = False
        self.models: dict[str, Model] = {}
        self._mpsge_models: list[str] = []
        if IS_MIRO_INIT:
            atexit.register(self._write_miro_files)

        system_directory = get_system_directory(system_directory)
        add_sysdir_to_path(system_directory)
        self.system_directory = system_directory
        self._gams2np = gnp.Gams2Numpy._bypass_workspace(system_directory)

        self._unsaved_statements: list = []

        self._data: CasePreservingDict[str, SymbolType] = CasePreservingDict()

        self._options = validation.validate_global_options(options)
        if self._options.license is not None:
            self._license_path = self._options.license
        else:
            self._license_path = utils._get_license_path(self.system_directory)

        self._network_license = self._is_network_license()

        self._debugging_level = debugging_level
        self._workspace = Workspace(debugging_level, working_directory)
        self._working_directory = self._workspace.working_directory
        self._process_directory = tempfile.mkdtemp(dir=self.working_directory)

        self._job, self._gdx_in, self._gdx_out = self._setup_paths()

        # needed for miro
        self._miro_input_symbols: list[str] = []
        self._miro_output_symbols: list[str] = []

        open_connection(self)
        weakref.finalize(self, close_connection, self._comm_pair_id)

        self._is_restarted = False
        if load_from is not None:
            if isinstance(load_from, os.PathLike):
                load_from = os.fspath(load_from)

            if not isinstance(load_from, (str, gt.Container, Container)):
                raise ValidationError(
                    f"`load_from` must be of type str or Container but found {type(load_from)}"
                )

            if isinstance(load_from, str):
                if load_from.endswith(".gdx"):
                    self.loadRecordsFromGdx(load_from)
                elif load_from.endswith(".g00"):
                    self._options._set_extra_options(
                        {
                            "restart": load_from,
                            "gdx": self._gdx_out,
                            "gdxSymbols": "allNoData",
                        }
                    )
                    self._synch_with_gams()
                    self._options._set_extra_options({})
                    symbol_names = gdxio._get_symbol_names_from_gdx(
                        self.system_directory, self._gdx_out
                    )
                    gdxio.load_missing_symbols(
                        self, self._gdx_out, symbol_names, declare_in_gams=False
                    )
                    self._should_load_from_gams(symbol_names, value=True)
                    self._is_restarted = True
                else:
                    raise ValidationError(
                        f"`load_from` must end with .gdx or .g00 but found {load_from}"
                    )
            else:
                self._read(load_from)
                self._synch_with_gams()

    def __enter__(self) -> Container:
        pid = os.getpid()
        tid = threading.get_native_id()
        gp._ctx_managers[(pid, tid)] = self

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pid = os.getpid()
        tid = threading.get_native_id()

        try:
            del gp._ctx_managers[(pid, tid)]
        except KeyError:
            ...

    def __getitem__(self, symbol_name: str) -> SymbolType:
        try:
            return self._data[symbol_name]
        except KeyError as e:
            error_message = f"`{symbol_name}` does not exist in the Container."
            matches = get_close_matches(
                word=symbol_name, possibilities=self._data.keys(), n=1
            )
            if matches:
                error_message += f" Did you mean `{matches[0]}`?"
            raise KeyError(error_message) from e

    def __iter__(self):
        return iter(self._data.items())

    def __repr__(self) -> str:
        return f"Container(system_directory='{self.system_directory}', working_directory='{self.working_directory}', debugging_level='{self._debugging_level}')"

    def __str__(self):
        if len(self):
            return f"<Container ({hex(id(self))}) with {len(self)} symbols: {self._data.keys()}>"

        return f"<Empty Container ({hex(id(self))})>"

    def __contains__(self, sym) -> bool:
        return sym in self._data

    def __len__(self) -> int:
        return len(self._data)

    @property
    def data(self) -> dict[str, SymbolType]:
        """
        The dictionary that contains all symbols in the Container. Keys are symbol names and values are the symbols themselves.

        Returns
        -------
        dict[str, SymbolType]

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i")
        >>> j = gp.Set(m, "j")
        >>> m.data
        {'i': Set(name='i', domain=['*']), 'j': Set(name='j', domain=['*'])}

        """
        return self._data

    def _resolve_symbols(self, symbols: str | list[str] | None = None) -> list[str]:
        """Validates and sanitizes the `symbols` argument to avoid repetitive logic."""
        if not isinstance(symbols, (str, list, type(None))):
            raise TypeError("Argument 'symbols' must be type list, str or NoneType")

        if symbols is None:
            return list(self._data.keys())

        if isinstance(symbols, str):
            return [symbols]

        if any(not isinstance(i, str) for i in symbols):
            raise TypeError("Argument 'symbols' must contain only type str")

        return symbols

    def _build_describe_dataframe(
        self,
        symbols: str | list[str] | None,
        default_symbols: list[str],
        row_extractor: Callable,
    ) -> DataFrame | None:
        """Helper to simplify DataFrame construction for describe() methods."""
        # Override None behavior for describe functions (they default to specific typed symbol lists, not all container keys)
        resolved_symbols = (
            self._resolve_symbols(symbols) if symbols is not None else default_symbols
        )

        if not resolved_symbols:
            return None

        rows = []
        for sym in self.getSymbols(resolved_symbols):
            row = row_extractor(sym)
            rows.append(row)

        if not rows:
            return None

        df = pd.DataFrame(rows)
        return df.round(3).sort_values(by="name", ignore_index=True)

    def _getUELs(
        self,
        symbols: str | list[str] | None = None,
        ignore_unused: bool = False,
    ) -> list[str]:
        AnyContainerAlias = (gp.Alias, gp.UniverseAlias)

        # Use helper for validation, but respect the custom `None` default behavior of this function
        symbols = (
            self._resolve_symbols(symbols)
            if symbols is not None
            else self.listSymbols()
        )

        uni = {}
        for symobj in self.getSymbols(symbols):
            if not isinstance(symobj, AnyContainerAlias) and symobj.records is not None:
                uni.update(dict.fromkeys(symobj._getUELs(ignore_unused=ignore_unused)))

        return list(uni.keys())

    def _assert_valid_records(self, symbols=None):
        symbols = self._resolve_symbols(symbols)
        for symobj in self.getSymbols(symbols):
            if not isinstance(symobj, gp.UniverseAlias):
                symobj._assert_valid_records()

    def hasSymbols(self, symbols: list[str] | str) -> list[bool] | bool:
        """
        Checks if the specified symbol or symbols exist in the Container.

        Parameters
        ----------
        symbols : str | list[str]
            Name of the symbol or a list of symbol names to check for existence.

        Returns
        -------
        bool | list[bool]
            A boolean value if a single string is provided, or a list of boolean values
            corresponding to the list of strings provided, indicating whether each symbol
            is present in the Container.

        Raises
        ------
        TypeError
            If the 'symbols' argument is not of type str or list.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = m.addSet("i")
        >>> m.hasSymbols("i")
        True
        >>> m.hasSymbols(["i", "j"])
        [True, False]

        """
        if isinstance(symbols, str):
            return symbols in self
        elif isinstance(symbols, list):
            return [sym in self for sym in symbols]

        raise TypeError("Argument 'symbols' must be type str or list")

    def listSymbols(self) -> list[str]:
        """
        Lists the names of all symbols present in the Container.

        Returns
        -------
        list[str]
            A list containing the names of all symbols.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = m.addSet("i")
        >>> v = m.addVariable("v")
        >>> m.listSymbols()
        ['i', 'v']

        """
        return list(self._data.keys())

    def listParameters(self) -> list[str]:
        """
        Lists the names of all Parameter symbols in the Container.

        Returns
        -------
        list[str]
            A list containing the names of all parameters.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> p = m.addParameter("p")
        >>> m.listParameters()
        ['p']

        """
        return [
            s.name
            for s in self.getSymbols(self.listSymbols())
            if isinstance(s, gp.Parameter)
        ]

    def listSets(self) -> list[str]:
        """
        Lists the names of all Set symbols in the Container.

        Returns
        -------
        list[str]
            A list containing the names of all sets.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = m.addSet("i")
        >>> m.listSets()
        ['i']

        """
        return [
            s.name for s in self.getSymbols(self.listSymbols()) if isinstance(s, gp.Set)
        ]

    def listAliases(self) -> list[str]:
        """
        Lists the names of all Alias and UniverseAlias symbols in the Container.

        Returns
        -------
        list[str]
            A list containing the names of all aliases.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = m.addSet("i")
        >>> a = m.addAlias("a", i)
        >>> m.listAliases()
        ['a']

        """
        return [
            s.name
            for s in self.getSymbols(self.listSymbols())
            if isinstance(s, (gp.Alias, gp.UniverseAlias))
        ]

    def listVariables(self, types: str | list[str] | None = None) -> list[str]:
        """
        Lists the names of all variables in the Container, optionally filtering by variable type.

        Parameters
        ----------
        types : str | list[str] | None, optional
            The variable type(s) to filter by (e.g., "free", "positive", "binary").
            If None, all variables are listed.

        Returns
        -------
        list[str]
            A list containing the names of the filtered variables.

        Raises
        ------
        TypeError
            If the 'types' argument is not of type str, list, or None.
        ValueError
            If an unrecognized variable type is provided.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> v1 = m.addVariable("v1", type="free")
        >>> v2 = m.addVariable("v2", type="binary")
        >>> m.listVariables(types="binary")
        ['v2']

        """
        if not isinstance(types, (str, list, type(None))):
            raise TypeError("Argument 'types' must be type str, list, or NoneType")

        syms = [
            s for s in self.getSymbols(self.listSymbols()) if isinstance(s, gp.Variable)
        ]
        if types is None:
            return [s.name for s in syms]

        types = [types] if isinstance(types, str) else types
        types = [i.casefold() for i in types]

        if any(i not in TRANSFER_TO_GAMS_VARIABLE_SUBTYPES for i in types):
            raise ValueError(
                "User input unrecognized variable type, "
                f"variable types can only take: {list(TRANSFER_TO_GAMS_VARIABLE_SUBTYPES.keys())}"
            )

        return [s.name for s in syms if s.type in types]

    def listEquations(self, types: str | list[str] | None = None) -> list[str]:
        """
        Lists the names of all equations in the Container, optionally filtering by equation type.

        Parameters
        ----------
        types : str | list[str] | None, optional
            The equation type(s) to filter by (e.g., "regular").
            If None, all equations are listed.

        Returns
        -------
        list[str]
            A list containing the names of the filtered equations.

        Raises
        ------
        TypeError
            If the 'types' argument is not of type str, list, or None.
        ValueError
            If an unrecognized equation type is provided.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> e = m.addEquation("e", type="regular")
        >>> m.listEquations()
        ['e']

        """
        if not isinstance(types, (str, list, type(None))):
            raise TypeError("Argument 'types' must be type str, list, or NoneType")

        syms = [
            s for s in self.getSymbols(self.listSymbols()) if isinstance(s, gp.Equation)
        ]
        if types is None:
            return [s.name for s in syms]

        types = [types] if isinstance(types, str) else types
        types = [EQU_TYPE[i.casefold()] for i in types]

        if any(i not in TRANSFER_TO_GAMS_EQUATION_SUBTYPES for i in types):
            raise ValueError(
                "User input unrecognized variable type, "
                f"variable types can only take: {list(TRANSFER_TO_GAMS_EQUATION_SUBTYPES.keys())}"
            )

        return [s.name for s in syms if s.type in types]

    def describeSets(self, symbols: str | list[str] | None = None) -> DataFrame | None:
        """
        Provides a structural summary of the specified Set symbols as a pandas DataFrame.

        Parameters
        ----------
        symbols : str | list[str] | None, optional
            Name or list of names of the Set symbols to describe.
            If None, describes all Sets in the Container.

        Returns
        -------
        DataFrame | None
            A pandas DataFrame describing the Sets (e.g., dimension, domain, sparsity, number of records),
            or None if no matching Sets are found.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = m.addSet("i", records=["a", "b"])
        >>> df = m.describeSets()

        """

        def extractor(sym):
            is_alias = isinstance(sym, (gp.Alias, gp.UniverseAlias))
            alias_with = None
            if is_alias:
                alias_with = (
                    sym.alias_with.name
                    if hasattr(sym.alias_with, "name")
                    else sym.alias_with
                )

            return {
                "name": sym.name,
                "is_singleton": sym.is_singleton,
                "is_alias": is_alias,
                "alias_with": alias_with,
                "domain": sym.domain_names,
                "domain_type": sym.domain_type,
                "dimension": sym.dimension,
                "number_records": sym.number_records,
                "sparsity": sym.getSparsity(),
            }

        df = self._build_describe_dataframe(symbols, self.listSets(), extractor)
        if df is not None and not df["is_alias"].any():
            df = df.drop(columns=["is_alias", "alias_with"])
        return df

    def describeAliases(
        self, symbols: str | list[str] | None = None
    ) -> DataFrame | None:
        """
        Provides a structural summary of the specified Alias symbols as a pandas DataFrame.

        Parameters
        ----------
        symbols : str | list[str] | None, optional
            Name or list of names of the Alias symbols to describe.
            If None, describes all Aliases in the Container.

        Returns
        -------
        DataFrame | None
            A pandas DataFrame describing the Aliases (e.g., alias_with, domain, number of records),
            or None if no matching Aliases are found.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = m.addSet("i", records=["a", "b"])
        >>> a = m.addAlias("a", i)
        >>> df = m.describeAliases()

        """

        def extractor(sym):
            if isinstance(sym, gp.Alias):
                alias_parent = sym.alias_with.name
            elif isinstance(sym, gp.UniverseAlias):
                alias_parent = sym.alias_with

            return {
                "name": sym.name,
                "alias_with": alias_parent,
                "is_singleton": sym.is_singleton,
                "domain": sym.domain_names,
                "domain_type": sym.domain_type,
                "dimension": sym.dimension,
                "number_records": sym.number_records,
                "sparsity": sym.getSparsity(),
            }

        return self._build_describe_dataframe(symbols, self.listAliases(), extractor)

    def describeParameters(
        self, symbols: str | list[str] | None = None
    ) -> DataFrame | None:
        """
        Provides a statistical summary of the specified Parameter symbols as a pandas DataFrame.

        Parameters
        ----------
        symbols : str | list[str] | None, optional
            Name or list of names of the Parameter symbols to describe.
            If None, describes all Parameters in the Container.

        Returns
        -------
        DataFrame | None
            A pandas DataFrame describing the Parameters (e.g., domain, min/mean/max values, sparsity),
            or None if no matching Parameters are found.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> p = m.addParameter("p", records=5)
        >>> df = m.describeParameters()

        """

        def extractor(sym):
            return {
                "name": sym.name,
                "domain": sym.domain_names,
                "domain_type": sym.domain_type,
                "dimension": sym.dimension,
                "number_records": sym.number_records,
                "min": sym.getMinValue(),
                "mean": sym.getMeanValue(),
                "max": sym.getMaxValue(),
                "where_min": sym.whereMin(),
                "where_max": sym.whereMax(),
                "sparsity": sym.getSparsity(),
            }

        return self._build_describe_dataframe(symbols, self.listParameters(), extractor)

    def describeVariables(
        self, symbols: str | list[str] | None = None
    ) -> DataFrame | None:
        """
        Provides a statistical and structural summary of the specified Variable symbols as a pandas DataFrame.

        Parameters
        ----------
        symbols : str | list[str] | None, optional
            Name or list of names of the Variable symbols to describe.
            If None, describes all Variables in the Container.

        Returns
        -------
        DataFrame | None
            A pandas DataFrame describing the Variables (e.g., type, min/max level, sparsity, dimension),
            or None if no matching Variables are found.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> v = m.addVariable("v")
        >>> df = m.describeVariables()

        """

        def extractor(sym):
            return {
                "name": sym.name,
                "type": sym.type,
                "domain": sym.domain_names,
                "domain_type": sym.domain_type,
                "dimension": sym.dimension,
                "number_records": sym.number_records,
                "sparsity": sym.getSparsity(),
                "min_level": sym.getMinValue("level"),
                "mean_level": sym.getMeanValue("level"),
                "max_level": sym.getMaxValue("level"),
                "where_max_abs_level": sym.whereMaxAbs("level"),
            }

        return self._build_describe_dataframe(symbols, self.listVariables(), extractor)

    def describeEquations(
        self, symbols: str | list[str] | None = None
    ) -> DataFrame | None:
        """
        Provides a statistical and structural summary of the specified Equation symbols as a pandas DataFrame.

        Parameters
        ----------
        symbols : str | list[str] | None, optional
            Name or list of names of the Equation symbols to describe.
            If None, describes all Equations in the Container.

        Returns
        -------
        DataFrame | None
            A pandas DataFrame describing the Equations (e.g., type, min/max level, sparsity, dimension),
            or None if no matching Equations are found.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> e = m.addEquation("e")
        >>> df = m.describeEquations()

        """

        def extractor(sym):
            return {
                "name": sym.name,
                "type": sym.type,
                "domain": sym.domain_names,
                "domain_type": sym.domain_type,
                "dimension": sym.dimension,
                "number_records": sym.number_records,
                "sparsity": sym.getSparsity(),
                "min_level": sym.getMinValue("level"),
                "mean_level": sym.getMeanValue("level"),
                "max_level": sym.getMaxValue("level"),
                "where_max_abs_level": sym.whereMaxAbs("level"),
            }

        return self._build_describe_dataframe(symbols, self.listEquations(), extractor)

    def getSets(self) -> list[Set]:
        """
        Retrieves all Set symbols present in the Container.

        Returns
        -------
        list[Set]
            A list containing all Set objects in the Container.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = m.addSet("i")
        >>> sets = m.getSets()

        """
        return cast("list[Set]", self.getSymbols(self.listSets()))

    def getAliases(self) -> list[Alias | UniverseAlias]:
        """
        Retrieves all Alias and UniverseAlias symbols present in the Container.

        Returns
        -------
        list[Alias | UniverseAlias]
            A list containing all Alias and UniverseAlias objects in the Container.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = m.addSet("i")
        >>> a = m.addAlias("a", i)
        >>> aliases = m.getAliases()

        """
        return cast("list[Alias | UniverseAlias]", self.getSymbols(self.listAliases()))

    def getParameters(self) -> list[Parameter]:
        """
        Retrieves all Parameter symbols present in the Container.

        Returns
        -------
        list[Parameter]
            A list containing all Parameter objects in the Container.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> p = m.addParameter("p")
        >>> parameters = m.getParameters()

        """
        return cast("list[Parameter]", self.getSymbols(self.listParameters()))

    def getVariables(self, types: str | list[str] | None = None) -> list[Variable]:
        """
        Retrieves all Variable symbols in the Container, optionally filtering by variable type.

        Parameters
        ----------
        types : str | list[str] | None, optional
            The variable type(s) to filter by (e.g., "free", "positive", "binary").
            If None, retrieves all variables.

        Returns
        -------
        list[Variable]
            A list containing the filtered Variable objects.

        Raises
        ------
        TypeError
            If the 'types' argument is not of type str, list, or None.
        ValueError
            If an unrecognized variable type is provided.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> v = m.addVariable("v", type="free")
        >>> variables = m.getVariables()

        """
        return cast("list[Variable]", self.getSymbols(self.listVariables(types=types)))

    def getEquations(self) -> list[Equation]:
        """
        Returns all equation symbols in the Container.

        Returns
        -------
        list[Equation]

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> eq1 = gp.Equation(m, name="eq1")
        >>> eq2 = gp.Equation(m, name="eq2")
        >>> equation_objects = m.getEquations()

        """
        equations = [
            equation
            for equation in self.listEquations()
            if not equation.startswith(ATTR_PREFIX)
        ]
        return cast("list[Equation]", self.getSymbols(equations))

    def getSymbols(
        self, symbols: str | Iterable[str] | None = None
    ) -> list[SymbolType]:
        """
        Retrieves specific symbols from the Container by their names.

        Parameters
        ----------
        symbols : str | Iterable[str] | None, optional
            Name or iterable of names of the symbols to retrieve.
            If None, retrieves all symbols in the Container.

        Returns
        -------
        list[SymbolType]
            A list containing the requested symbol objects.

        Raises
        ------
        TypeError
            If the 'symbols' argument is not a str, Iterable, or None.
        KeyError
            If a specified symbol name does not exist in the Container.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = m.addSet("i")
        >>> p = m.addParameter("p")
        >>> syms = m.getSymbols(["i", "p"])

        """
        if symbols is None:
            return list(self._data.values())

        if isinstance(symbols, str):
            symbols = [symbols]
        elif not isinstance(symbols, Iterable):
            raise TypeError("Argument 'symbols' must be type str or other iterable")

        obj: list[SymbolType] = []
        for symname in symbols:
            try:
                obj.append(self._data[symname])
            except KeyError as err:
                raise KeyError(
                    f"Symbol `{symname}` does not appear in the Container"
                ) from err
        return obj

    @property
    def working_directory(self) -> str:
        """
        Absolute path of the working directory used by this container.
        The directory contains generated GAMS input/output files such as
        ``.gms``, ``.lst``, ``.gdx`` and temporary scratch files.

        Returns
        -------
        str

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> m.working_directory # doctest: +ELLIPSIS
        '...'

        """
        return self._working_directory

    @property
    def in_miro(self) -> bool:
        """
        Indicates whether the container is executed inside a GAMS MIRO context.
        When running under MIRO, input data is typically provided by MIRO itself.
        This flag can be used to skip expensive or redundant data-loading steps
        (e.g., reading Excel files).

        Returns
        -------
        bool

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> if not m.in_miro:
        ...     pass # e.g. load data from files

        """
        return MIRO_GDX_IN is not None

    def _is_network_license(self) -> bool:
        with open(self._license_path, encoding="utf-8") as file:
            lines = file.readlines()

        return bool(lines[0][54] == "+" and lines[4][47] == "N")

    def _get_symbol_name(self, prefix: str) -> str:
        use_py_var_name = get_option("USE_PY_VAR_NAME")
        if use_py_var_name == "no":
            name = prefix + utils._get_unique_name() + "gpauto"
        elif use_py_var_name == "yes":
            name = utils._get_name_from_stack()
        elif use_py_var_name == "yes-or-autogenerate":
            try:
                name = utils._get_name_from_stack()
                # if a symbol with the same name exists, autogenerate.
                try:
                    _ = self._data[name]
                    name = prefix + utils._get_unique_name() + "gpauto"
                except KeyError:
                    ...
            except ValidationError:
                name = prefix + utils._get_unique_name() + "gpauto"
        else:
            raise ValidationError(
                f'Invalid value `{use_py_var_name}` for `USE_PY_VAR_NAME`. Possible values are "no", "yes", "yes-or-autogenerate"'
            )

        return name

    @no_type_check
    def _interrupt(self) -> None:
        _, process = get_connection(self._comm_pair_id)
        if platform.system() == "Windows":
            os.kill(process.pid, signal.CTRL_C_EVENT)
        else:
            os.kill(process.pid, signal.SIGINT)

    def _write_miro_files(self):
        # create conf_<model>/<model>_io.json
        try:
            encoder = MiroJSONEncoder(self)
            encoder.write_json()
        except Exception:
            close_connection(self._comm_pair_id)
            traceback.print_exc(file=sys.stderr)
            os._exit(1)

    def _add_statement(self, statement) -> None:
        self._unsaved_statements.append(statement)

    def _delete_autogenerated_symbols(self) -> None:
        """
        Removes autogenerated model attributes, objective variable and equation from
        the container
        """
        autogenerated_symbol_names = self._get_autogenerated_symbol_names()

        for name in autogenerated_symbol_names:
            if name in self._data:
                del self._data[name]

    def _setup_paths(self) -> tuple[str, str, str]:
        suffix = "_" + utils._get_unique_name()
        job = os.path.join(self.working_directory, suffix)
        gdx_in = f"{job}in.gdx"
        gdx_out = f"{job}out.gdx"

        return job, gdx_in, gdx_out

    def _get_autogenerated_symbol_names(self) -> list[str]:
        names = []
        for name in self._data:
            if name.startswith(ATTR_PREFIX):
                names.append(name)

        return names

    def _symbols_to_unload(self) -> list[str]:
        symbol_names: list[str] = []
        seen: set[str] = set()

        # Needed to avoid attribute lookup.
        append = symbol_names.append
        seen_add = seen.add

        for name, symbol in self._data.items():
            if not symbol._should_unload_to_gams:
                continue

            if type(symbol) is gp.Alias:
                alias_name = symbol._alias_with.name

                if alias_name not in seen:
                    append(alias_name)
                    seen_add(alias_name)

            append(name)
            seen_add(name)

        return symbol_names

    def _should_unload_to_gams(self, value: bool = False) -> None:
        for symbol in self._data.values():
            symbol._should_unload_to_gams = value

    def _should_load_from_gams(
        self, symbol_names: Iterable[str], value: bool = True
    ) -> None:
        for name in symbol_names:
            symbol = self._data[name]
            symbol._should_load_from_gams = value
            if not isinstance(symbol, (gp.Alias, gp.UniverseAlias)):
                symbol._handle_domain_forwarding()

    def _synch_with_gams(self) -> DataFrame | None:
        if self._in_loop:
            return None

        runner = backend_factory(self, self._options, output=self.output)
        summary = runner.run()

        if self._options and self._options.seed is not None:
            # Required for correct seeding. Seed can only be set in the first run.
            self._options.seed = None

        if IS_MIRO_INIT:
            miro._write_default_gdx_miro(self)

        return summary

    def _generate_gams_string(self, gdx_in: str, symbol_names: list[str]) -> str:
        LOADABLE = (gp.Set, gp.Parameter, gp.Variable, gp.Equation)
        MIRO_INPUT_TYPES = (gp.Set, gp.Parameter)
        assume_suffix = int(get_option("ASSUME_VARIABLE_SUFFIX"))

        strings = ["$onMultiR", "$onUNDF"]
        if assume_suffix == 1:
            strings.append("$onDotL")
        elif assume_suffix == 2:
            strings.append("$onDotScale")

        for statement in self._unsaved_statements:
            if type(statement) is str:
                strings.append(statement)
            else:
                strings.append(statement.getDeclaration())

        if symbol_names:
            loadables = []
            for name in symbol_names:
                symbol = self._data[name]
                if type(symbol) in LOADABLE and not name.startswith(ATTR_PREFIX):
                    loadables.append(symbol)

            if loadables:
                strings.append(f"$gdxIn {gdx_in}")
                for loadable in loadables:
                    if (
                        type(loadable) in MIRO_INPUT_TYPES
                        and loadable._is_miro_input
                        and not IS_MIRO_INIT
                        and MIRO_GDX_IN
                    ):
                        miro_load = miro.get_load_input_str(loadable.name, gdx_in)
                        strings.append(miro_load)
                    else:
                        strings.append(f"$loadDC {loadable.name}")

                strings.append("$gdxIn")

        if assume_suffix == 1:
            strings.append("$offDotL")
        elif assume_suffix == 2:
            strings.append("$offDotScale")

        strings.extend(["$offUNDF", "$offMulti"])

        if not IS_MIRO_INIT and MIRO_GDX_OUT:
            if len(self._miro_output_symbols) == 0:
                self._write(MIRO_GDX_OUT, symbol_names=[])
            else:
                strings.append(miro.get_unload_output_str(self))

        gams_string = "\n".join(strings)

        if self._debugging_level == "keep":
            self._gams_string += gams_string + "\n"

        return gams_string

    def _validate_load_symbols(self, symbol_names: dict[str, str]) -> dict[str, str]:
        for gamspy_name in symbol_names.values():
            if gamspy_name not in self._data:
                raise ValidationError(
                    f"Invalid renaming. `{gamspy_name}` does not exist in the container."
                )

        return symbol_names

    def _load_records_with_rename(self, load_from: str, names: dict[str, str]) -> None:
        records_dict = gdxio.get_records(self, load_from, names)
        original_state = self._options.miro_protect
        self._options.miro_protect = False

        for gamspy_name in names.values():
            updated_records = records_dict[gamspy_name]
            symbol = self._data[gamspy_name]
            symbol.records = updated_records
            symbol.domain_labels = symbol.domain_names
            symbol._should_unload_to_gams = False

        self._options.miro_protect = original_state

    # TODO: Legacy function from GTP. Pay the technical debt.
    @no_type_check
    def _read_from_container(
        self, load_from: gt.Container | Container, symbols: list[str] | None
    ):
        read_symbols = symbols if symbols is not None else list(load_from.data.keys())

        for i in read_symbols:
            if i not in load_from:
                raise ValidationError(
                    f"User specified to read symbol `{i}`, but it does not exist in source Container."
                )

        for symname in read_symbols:
            if symname in self._data:
                raise ValidationError(
                    f"Attempting to create a new symbol (through a read operation) named `{symname}` "
                    "but an object with this name already exists in the Container. "
                )

        AnyContainerDomainSymbol = (
            gp.Set,
            gp.Alias,
            gp.UniverseAlias,
            gt.Set,
            gt.Alias,
            gt.UniverseAlias,
        )

        cf_read_symbols = [s.casefold() for s in read_symbols]
        link_domains = []

        for symname, symobj in load_from.data.items():
            if symname.casefold() not in cf_read_symbols:
                continue

            if any(
                isinstance(domobj, AnyContainerDomainSymbol) for domobj in symobj.domain
            ):
                link_domains.append((symname, symobj.domain_names))

            if isinstance(symobj, (gp.Set, gt.Set)):
                symbol = gp.Set._constructor_bypass(
                    self,
                    symname,
                    symobj.domain_names,
                    is_singleton=symobj.is_singleton,
                    records=copy.deepcopy(symobj.records),
                    description=symobj.description,
                )
                symbol._should_unload_to_gams = True
            elif isinstance(symobj, (gp.Parameter, gt.Parameter)):
                symbol = gp.Parameter._constructor_bypass(
                    self,
                    symname,
                    symobj.domain_names,
                    records=copy.deepcopy(symobj.records),
                    description=symobj.description,
                )
                symbol._should_unload_to_gams = True
            elif isinstance(symobj, (gp.Variable, gt.Variable)):
                symbol = gp.Variable._constructor_bypass(
                    self,
                    symname,
                    symobj.type,
                    symobj.domain_names,
                    records=copy.deepcopy(symobj.records),
                    description=symobj.description,
                )
                symbol._should_unload_to_gams = True
            elif isinstance(symobj, (gp.Equation, gt.Equation)):
                symbol = gp.Equation._constructor_bypass(
                    self,
                    symname,
                    symobj.type,
                    symobj.domain_names,
                    records=copy.deepcopy(symobj.records),
                    description=symobj.description,
                )
                symbol._should_unload_to_gams = True
            elif isinstance(symobj, (gp.Alias, gt.Alias)):
                alias_with = cast("Set | Alias", self._data[symobj.alias_with.name])
                gp.Alias._constructor_bypass(self, symname, alias_with)
            elif isinstance(symobj, (gp.UniverseAlias, gt.UniverseAlias)):
                gp.UniverseAlias._constructor_bypass(self, symname)

        for symname, domain in link_domains:
            domain = [
                self._data[i] if i in self and i in cf_read_symbols else i
                for i in domain
            ]
            self._data[symname]._domain = domain

        self._synch_with_gams()

    def _read(
        self,
        load_from: str | os.PathLike | Container | gt.Container,
        symbol_names: list[str] | None = None,
        encoding: str | None = None,
        *,
        load_records: bool = True,
    ) -> None:
        symbols = (
            self._resolve_symbols(symbol_names) if symbol_names is not None else None
        )

        if not isinstance(encoding, (str, type(None))):
            raise TypeError("Argument 'encoding' must be type str or NoneType")

        if isinstance(load_from, (os.PathLike, str)):
            fpath = Path(load_from).expanduser().resolve()  # ty: ignore[invalid-argument-type]
            if not fpath.exists():
                raise ValueError(
                    f"GDX file '{os.fspath(fpath)}' does not exist, "
                    "check filename spelling or path specification"
                )
            if not os.fspath(fpath).casefold().endswith(".gdx"):
                raise ValueError(
                    "Unexpected file type passed to 'load_from' argument -- expected file extension '.gdx'"
                )
            load_from = os.fspath(fpath)
            gdxio.read(self, load_from, symbols, load_records, encoding)
        elif isinstance(load_from, (gt.Container, Container)):
            self._read_from_container(load_from, symbols)
        else:
            raise ValueError(
                "Argument 'load_from' expects "
                "type str or PathLike (i.e., a path to a GDX file) "
                ", a valid gmdHandle (or GamsDatabase instance) "
                ", an instance of another Container "
                ", User passed: "
                f"'{type(load_from)}'."
            )

    def read(
        self,
        load_from: str | os.PathLike | Container | gt.Container,
        symbol_names: list[str] | None = None,
        encoding: str | None = None,
        *,
        load_records: bool = True,
    ) -> None:
        """
        Read symbols and records from a GDX file or another container.


        Parameters
        ----------
        load_from : str | os.PathLike | Container | gt.Container
            Source to read from.


        symbol_names : list[str], optional
            Names of symbols to read. If omitted, all symbols are read.


        mode : str, optional
            GDX read mode ("category", or "string", default: "category").


        encoding : str, optional
            Text encoding for symbol metadata.


        load_records : bool, optional
            Whether to load symbol records (default: True).


        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=["a", "b"])
        >>> m.write("example.gdx")


        >>> m2 = gp.Container()
        >>> m2.read("example.gdx")
        >>> "i" in m2.data
        True


        Reading selected symbols only


        >>> m2 = gp.Container()
        >>> m2.read("example.gdx", symbol_names=["i"])

        """
        if isinstance(load_from, os.PathLike):
            load_from = os.fspath(load_from)

        self._read(load_from, symbol_names, encoding, load_records=load_records)
        self._synch_with_gams()

    def setRecords(
        self, records: dict[SymbolType, Any], *, uels_on_axes: bool | list[bool] = False
    ) -> None:
        """
        Set records for multiple symbols in a single batch operation.


        This is functionally equivalent to calling ``symbol.setRecords`` for
        each symbol, but triggers only one synchronization with GAMS.


        Parameters
        ----------
        records : dict[SymbolType, Any]
            Mapping from symbols to their new records.


        uels_on_axes : bool | list[bool], optional
            Whether domain labels (UELs) are stored on pandas axes. Either a single
            boolean applied to all symbols, or a list matching ``records`` order.


        Raises
        ------
        ValidationError
            If ``uels_on_axes`` length does not match ``records`` length.


        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i")
        >>> j = gp.Set(m, "j")
        >>> m.setRecords({i: ["a", "b"], j: [1, 2, 3]})

        """
        if not isinstance(uels_on_axes, list):
            uels_on_axes = [uels_on_axes] * len(records)

        if len(uels_on_axes) != len(records):
            raise ValidationError(
                f"Length of `records` and `uels_on_axes` must match. Size of records: {len(records)}, size of uels_on_axes: {len(uels_on_axes)}"
            )

        for item, uels_on_axe in zip(records.items(), uels_on_axes, strict=False):
            symbol, record = item
            if isinstance(symbol, gp.UniverseAlias):
                continue

            symbol._setRecords(record, uels_on_axes=uels_on_axe)

        self._synch_with_gams()

    def _write(
        self,
        write_to: str | os.PathLike,
        symbol_names: list[str] | None = None,
        *,
        compress: bool = False,
        eps_to_zero: bool = True,
    ):
        symbols = (
            self._resolve_symbols(symbol_names) if symbol_names is not None else None
        )

        if isinstance(write_to, (os.PathLike, str)):
            fpath = Path(write_to).expanduser().resolve()
            if not os.fspath(fpath).casefold().endswith(".gdx"):
                raise ValueError(
                    "Unexpected file type passed to 'write_to' argument -- expected file extension '.gdx'"
                )
            write_to = os.fspath(fpath)
            gdxio.write(
                self, write_to, symbols, compress=compress, eps_to_zero=eps_to_zero
            )
        else:
            raise TypeError(
                "Argument 'write_to' expects type str/Pathlike (.gdx), a valid gmdHandle, or GamsDatabase."
            )

    def write(
        self,
        write_to: Path | str,
        symbol_names: list[str] | None = None,
        mode: str | None = None,
        *,
        compress: bool | None = None,
        eps_to_zero: bool = True,
    ) -> None:
        """
        Write symbols and records to a GDX file.

        Parameters
        ----------
        write_to : Path | str
            Target GDX file path.

        symbol_names : list[str], optional
            Symbols to write. If omitted, all symbols are written.

        compress : bool, optional
            Whether to compress the GDX file.

        mode : str, optional
            Write mode (passed to GAMS Transfer).

        eps_to_zero : bool, optional
            Convert EPS values to zero before writing.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=["x", "y"])
        >>> m.write("out.gdx")

        """
        self._synch_with_gams()

        if isinstance(write_to, Path):
            write_to = str(write_to.resolve())

        if compress is not None:
            # Store the old value of `compress` in OLDVALUE_GDXCOMPRESS
            self._add_statement(rf"""
$if setEnv GDXCOMPRESS $setLocal OLDVALUE_GDXCOMPRESS %sysEnv.GDXCOMPRESS%
$setEnv GDXCOMPRESS {int(compress)}
""")

        if mode is not None:
            warnings.warn(
                "'mode' is deprecated and will be removed in a future version. ",
                DeprecationWarning,
                stacklevel=2,
            )

        if eps_to_zero:
            self._add_statement("$onEpsToZero")

        if symbol_names is None:
            symbols_str = ""
        elif isinstance(symbol_names, list):
            symbols_str = " ".join(symbol_names)
        elif isinstance(symbol_names, dict):
            symbols_str = " ".join(
                [f"{key}={value}" for key, value in symbol_names.items()]
            )
        else:
            raise TypeError(
                f"`symbol_names` must be of type list, dict, or None but given {type(symbol_names)}"
            )

        self._add_statement(rf"$gdxUnload {write_to} {symbols_str}")

        if eps_to_zero:
            self._add_statement("$offEpsToZero")

        if compress is not None:
            # Restore the `compress`` value if there was one set, otherwise we drop the variable.
            self._add_statement(r"""
$ifThen setLocal OLDVALUE_GDXCOMPRESS
$  setEnv GDXCOMPRESS %OLDVALUE_GDXCOMPRESS%
$else
$  dropEnv GDXCOMPRESS
$endIf
""")

        self._synch_with_gams()

    def writeSolverOptions(
        self, solver: str, solver_options: dict | str | Path, file_number: int = 1
    ) -> None:
        """
        Writes solver options of the specified solver to the working directory.

        Parameters
        ----------
        solver : str
            Name of the solver.
        solver_options : dict | str | Path
            Options of the specified solver or path to an existing solver options file.
        file_number : int
            Solver option file number. Equivalent to optfile option of GAMS. See https://gams.com/latest/docs/UG_GamsCall.html#GAMSAOoptfile for more details.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> m.writeSolverOptions("conopt", solver_options={"rtmaxv": "1.e12"})

        """
        write_solver_options(self, solver, solver_options, file_number)

    def generateGamsString(
        self, path: str | None = None, *, show_raw: bool = False
    ) -> str:
        """
        Return the generated GAMS code executed by this container.


        Only available when ``debugging_level='keep'`` was specified at
        container creation time.


        Parameters
        ----------
        path : str, optional
            File path to write the generated GAMS code to.


        show_raw : bool, optional
            If True, strips data-loading and auxiliary statements, leaving
            the core model formulation.


        Returns
        -------
        str
            Generated GAMS code.


        Raises
        ------
        ValidationError
            If debugging level is not ``"keep"``.


        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container(debugging_level="keep")
        >>> i = gp.Set(m, "i")
        >>> gams = m.generateGamsString()

        """
        if self._debugging_level != "keep":
            raise ValidationError(
                "`debugging_level` argument of the container must be set to 'keep' to use this function."
            )

        gams_string = self._gams_string
        if show_raw:
            gams_string = utils._filter_gams_string(self._gams_string)

        if path:
            with open(path, "w") as file:
                file.write(gams_string)

        return gams_string

    def loadRecordsFromGdx(
        self,
        load_from: str | Path,
        symbol_names: list[str] | dict[str, str] | None = None,
    ) -> None:
        """
        Loads data of the given symbols from a GDX file. If no
        symbol names are given, data of all symbols are loaded.

        Parameters
        ----------
        load_from : str
            Path to the GDX file
        symbol_names : list[str], dict[str, str], optional
            Symbol names whose data will be load from GDX, by default None.
            Default option loads records of all symbols in the GDX file.
            If given as a dict, keys are the symbol names in the GDX file, and
            values are the names of the GAMSPy symbols.

        Examples
        --------
        >>> from gamspy import Container, Set
        >>> m = Container()
        >>> i = Set(m, "i", records=["i1", "i2"])
        >>> m.write("test.gdx")
        >>> m2 = Container()
        >>> i = Set(m2, "i")
        >>> m2.loadRecordsFromGdx("test.gdx")
        >>> print(i.records.equals(m2["i"].records))
        True

        """
        if isinstance(load_from, Path):
            load_from = str(load_from.resolve())

        if not os.path.exists(load_from):
            raise ValidationError(f"`{load_from}` does not exist.")

        if symbol_names is None:
            # If no symbol names are given, all records in the gdx should be loaded
            symbol_names = gdxio._get_symbol_names_from_gdx(
                self.system_directory, load_from
            )
            gdxio.load_missing_symbols(
                self, load_from, symbol_names, declare_in_gams=False
            )
            self._add_statement(f"$declareAndLoad {load_from}")
            self._synch_with_gams()
            self._should_load_from_gams(symbol_names)
            return

        if not isinstance(symbol_names, (dict, list)):
            raise TypeError("`symbol_names` must be either a list or a dictionary.")

        if isinstance(symbol_names, dict):
            symbol_names = self._validate_load_symbols(symbol_names)
            gdxio.load_missing_symbols(self, load_from, symbol_names)
            symbol_str = " ".join(
                f"{value}={key}" for key, value in symbol_names.items()
            )
            self._add_statement(f"$gdxLoad {load_from} {symbol_str}")
            self._synch_with_gams()
            self._should_load_from_gams(symbol_names.values())
        else:
            symbol_str = " ".join(symbol_names)
            gdxio.load_missing_symbols(self, load_from, symbol_names)
            self._add_statement(f"$gdxLoad {load_from} {symbol_str}")
            self._synch_with_gams()
            self._should_load_from_gams(symbol_names)

    def addGamsCode(self, gams_code: str) -> None:
        """
        Adds an arbitrary GAMS code to the generate .gms file.
        Using addGAMSCode might result in a license error if no GAMSpy++ license is used.

        Parameters
        ----------
        gams_code : str
            GAMS code that you want to insert.

        Examples
        --------
        >>> from gamspy import Container
        >>> m = Container()
        >>> m.addGamsCode("scalar piHalf / [pi/2] /;")
        >>> m["piHalf"].toValue()
        np.float64(1.5707963267948966)

        """
        self._add_statement(gams_code)
        self._options._set_extra_options(
            {"gdx": self._gdx_out, "gdxSymbols": "newOrChangedNoData"}
        )
        self._synch_with_gams()
        symbol_names = gdxio._get_symbol_names_from_gdx(
            self.system_directory, self._gdx_out
        )
        gdxio.load_missing_symbols(
            self, self._gdx_out, symbol_names, declare_in_gams=False
        )
        self._options._set_extra_options({})
        self._should_load_from_gams(symbol_names, value=True)

        # Unfortunately MPSGE requires a dirty trick
        pattern = re.compile(r"^\$sysInclude\s+mpsgeset\s+(\w+)\s*$", re.MULTILINE)
        match = pattern.search(gams_code)
        if match:
            model_name = match.group(1)
            self._mpsge_models.append(model_name.lower())

        self._unsaved_statements = []
        self._arbitrary_code_executed = True

    def close(self) -> None:
        """
        Close the connection to the GAMS execution engine and release resources.

        After calling this method, the container must not be used for further
        model execution or data synchronization. Symbol data remains accessible
        for read-only inspection.


        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()  # Starts running the GAMS execution engine.
        >>> m.close()           # Closes the connection to the execution engine.

        """
        close_connection(self._comm_pair_id)

    def addAlias(
        self,
        name: str | None = None,
        alias_with: Set | Alias = None,  # type: ignore
    ) -> Alias:
        """
        Creates a new Alias and adds it to the container

        Parameters
        ----------
        name : str, optional
            Name of the alias.
        alias_with : Set | Alias
            Alias set object.

        Returns
        -------
        Alias

        Raises
        ------
        TypeError
            In case the alias_with is different than a Set or an Alias
        ValueError
            If there is symbol with same name but different type in the
            Container

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = m.addSet("i")
        >>> a = m.addAlias("a", i)

        """
        if name is None:
            name = self._get_symbol_name(prefix="a")

        return gp.Alias(self, name, alias_with)

    def addUniverseAlias(self, name: str | None = None) -> UniverseAlias:
        """
        Creates a new UniverseAlias and adds it to the container

        Parameters
        ----------
        name : str, optional
            Name of the universe alias.

        Returns
        -------
        UniverseAlias

        Raises
        ------
        ValueError
            If there is symbol with same name but different type in the
            Container

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> a = m.addUniverseAlias("a")

        """
        if name is None:
            name = self._get_symbol_name(prefix="u")

        return gp.UniverseAlias(self, name)

    def addSet(
        self,
        name: str | None = None,
        domain: DomainType | None = None,
        is_singleton: bool = False,
        records: SetRecordsType | None = None,
        domain_forwarding: bool | list[bool] = False,
        description: str = "",
        uels_on_axes: bool = False,
        is_miro_input: bool = False,
        is_miro_output: bool = False,
    ) -> Set:
        """
        Creates a Set and adds it to the container.

        Parameters
        ----------
        name : str, optional
            Name of the set. If omitted, a unique name is generated.
        domain : Sequence[Set | Alias | str] | Set | Alias | str, optional
            Domain over which the set is defined.
        is_singleton : bool, optional
            If True, the set may contain at most one element.
        records : pd.DataFrame | np.ndarray | list, optional
            Initial elements of the set.
        domain_forwarding : bool | list[bool], optional
            Enable domain forwarding.
        description : str, optional
            Human-readable description.
        uels_on_axes : bool
            Assume that symbol domain information is contained in the axes of the given records.
        is_miro_input : bool
            Whether the symbol is a GAMS MIRO input symbol. See: https://gams.com/miro/tutorial.html
        is_miro_output : bool
            Whether the symbol is a GAMS MIRO output symbol. See: https://gams.com/miro/tutorial.html

        Returns
        -------
        Set

        Raises
        ------
        err
            In case arguments are not valid
        ValueError
            When there is symbol with same name in the
            Container

        Examples
        --------
        Simple set:


        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = m.addSet("i", records=["a", "b"])


        Indexed set:


        >>> j = m.addSet("j", domain=i)


        Singleton set:


        >>> s = m.addSet("s", is_singleton=True, records=["s1"])

        """
        if name is None:
            name = self._get_symbol_name(prefix="s")

        return gp.Set(
            self,
            name,
            domain,
            is_singleton,
            records,
            domain_forwarding,
            description,
            uels_on_axes,
            is_miro_input=is_miro_input,
            is_miro_output=is_miro_output,
        )

    def addParameter(
        self,
        name: str | None = None,
        domain: DomainType | None = None,
        records: ParameterRecordsType | None = None,
        domain_forwarding: bool | list[bool] = False,
        description: str = "",
        uels_on_axes: bool = False,
        is_miro_input: bool = False,
        is_miro_output: bool = False,
        is_miro_table: bool = False,
    ) -> Parameter:
        """
        Creates a Parameter and adds it to the Container

        Parameters
        ----------
        name : str, optional
            Name of the parameter. If omitted, a unique name is generated.
        domain : Sequence[Set | Alias | str] | Set | Alias | Dim | str, optional
            Domain over which the parameter is defined.
        records : int | float | pd.DataFrame | np.ndarray | list, optional
            Records of the parameter.
        domain_forwarding : bool | list[bool], optional
            Whether the parameter forwards the domain.
        description : str, optional
            Human-readable description.
        uels_on_axes : bool
            Assume that symbol domain information is contained in the axes of the given records.
        is_miro_input : bool
            Whether the symbol is a GAMS MIRO input symbol. See: https://gams.com/miro/tutorial.html
        is_miro_output : bool
            Whether the symbol is a GAMS MIRO output symbol. See: https://gams.com/miro/tutorial.html
        is_miro_table : bool
            Whether the symbol is a GAMS MIRO table symbol. See: https://gams.com/miro/tutorial.html

        Returns
        -------
        Parameter

        Raises
        ------
        err
            In case arguments are not valid
        ValueError
            If there is symbol with same name but different type in the
            Container

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> a = m.addParameter("a")

        """
        if name is None:
            name = self._get_symbol_name(prefix="p")

        return gp.Parameter(
            self,
            name,
            domain,
            records,
            domain_forwarding,
            description,
            uels_on_axes,
            is_miro_input=is_miro_input,
            is_miro_output=is_miro_output,
            is_miro_table=is_miro_table,
        )

    def addVariable(
        self,
        name: str | None = None,
        type: str = "free",
        domain: DomainType | None = None,
        records: VarEquRecordsType | None = None,
        domain_forwarding: bool | list[bool] = False,
        description: str = "",
        uels_on_axes: bool = False,
        is_miro_output: bool = False,
    ) -> Variable:
        """
        Creates a Variable and adds it to the Container

        Parameters
        ----------
        name : str, optional
            Name of the variable. If omitted, a unique name is generated.
        type : str, optional
            Type of the variable. "free" by default.
        domain : Sequence[Set | Alias | str] | Set | Alias | Dim | str, optional
            Domain of the variable.
        records : Sequence | np.ndarray | int | float | pd.DataFrame | pd.Series | dict, optional
            Records of the variable.
        domain_forwarding : bool | list[bool], optional
            Whether the variable forwards the domain.
        description : str, optional
            Description of the variable.
        is_miro_output : bool
            Whether the symbol is a GAMS MIRO output symbol. See: https://gams.com/miro/tutorial.html

        Returns
        -------
        Variable

        Raises
        ------
        err
            In case arguments are not valid
        ValueError
            If there is symbol with same name but different type in the
            Container

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> v = m.addVariable("v")

        """
        if name is None:
            name = self._get_symbol_name(prefix="v")

        return gp.Variable(
            self,
            name,
            type,
            domain,
            records,
            domain_forwarding,
            description,
            uels_on_axes,
            is_miro_output=is_miro_output,
        )

    def addEquation(
        self,
        name: str | None = None,
        type: str | EquationType = "regular",
        domain: DomainType | None = None,
        definition: Variable | Operation | Expression | None = None,
        records: VarEquRecordsType | None = None,
        domain_forwarding: bool | list[bool] = False,
        description: str = "",
        uels_on_axes: bool = False,
        is_miro_output: bool = False,
        definition_domain: list | None = None,
    ) -> Equation:
        """
        Creates an Equation and adds it to the Container

        Parameters
        ----------
        name : str, optional
            Name of the equation. If omitted, a unique name is generated.
        type : str
            Type of the equation. "regular" by default.
        domain : Sequence[Set | Alias] | Set | Alias, optional
            Domain of the variable.
        definition: Expression, optional
            Definition of the equation.
        records : Sequence | np.ndarray | int | float | pd.DataFrame | pd.Series | dict, optional
            Records of the equation.
        domain_forwarding : bool | list[bool], optional
            Whether the equation forwards the domain.
        description : str, optional
            Description of the equation.
        uels_on_axes: bool
            Assume that symbol domain information is contained in the axes of the given records.
        definition_domain: list, optional
            Definiton domain of the equation.
        is_miro_output : bool
            Whether the symbol is a GAMS MIRO output symbol. See: https://gams.com/miro/tutorial.html

        Returns
        -------
        Equation

        Raises
        ------
        err
            In case arguments are not valid
        ValueError
            If there is symbol with same name but different type in the
            Container

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = m.addEquation("i")

        """
        if name is None:
            name = self._get_symbol_name(prefix="e")

        return gp.Equation(
            self,
            name,
            type,
            domain,
            definition,
            records,
            domain_forwarding,
            description,
            uels_on_axes,
            is_miro_output,
            definition_domain,
        )

    def addModel(
        self,
        name: str | None = None,
        description: str = "",
        problem: Problem | str = Problem.MIP,
        equations: Sequence[Equation] = [],
        sense: Sense | str = Sense.FEASIBILITY,
        objective: Variable | Expression | None = None,
        matches: dict[
            Equation | Sequence[Equation],
            Variable | Sequence[Variable],
        ]
        | None = None,
        limited_variables: Sequence[ImplicitVariable] | None = None,
        external_module: str | None = None,
    ) -> Model:
        """
        Creates a Model and adds it to the Container

        Parameters
        ----------
        name : str, optional
            Name of the model. If omitted, a unique name is generated.
        description : str, optional
            Description of the model.
        equations : Sequence[Equation]
            Sequence of Equation objects.
        problem : Problem or str, optional
            'LP', 'NLP', 'QCP', 'DNLP', 'MIP', 'RMIP', 'MINLP', 'RMINLP', 'MIQCP', 'RMIQCP', 'MCP', 'CNS', 'MPEC', 'RMPEC', 'EMP', or 'MPSGE',
            by default Problem.LP.
        sense : Sense, optional
            "MIN", "MAX", or "FEASIBILITY".
        objective : Variable | Expression, optional
            Objective variable to minimize or maximize or objective itself.
        matches : dict[Equation | Sequence[Equation], Variable | Sequence[Variable]], optional
            Equation - Variable matches for MCP models.
        limited_variables : Sequence[ImplicitVariable], optional
            Allows limiting the domain of variables used in a model.
        external_module: str, optional
            The name of the external module in which the external equations are implemented

        Returns
        -------
        Model

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> x = gp.Variable(m, "x")
        >>> e = gp.Equation(m, "e", definition=x >= 1)
        >>> model = m.addModel(name="demo", equations=[e], problem="LP", sense="MIN", objective=x)

        """
        if name is None:
            name = self._get_symbol_name(prefix="m")

        return gp.Model(
            self,
            name,
            description,
            problem,
            equations,
            sense,
            objective,
            matches,
            limited_variables,
            external_module=external_module,
        )

    def copy(self, working_directory: str | None = None) -> Container:
        """
        Creates a copy of the Container. Should not be invoked after
        creating the model.

        Parameters
        ----------
        working_directory : str, optional
            Working directory of the new Container, by default None

        Returns
        -------
        Container

        Raises
        ------
        ValidationError

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i")
        >>> new_cont = m.copy(working_directory="test")
        >>> new_cont.data.keys() == m.data.keys()
        True

        """
        if working_directory is None:
            working_directory = tempfile.mkdtemp()

        os.makedirs(working_directory, exist_ok=True)
        if working_directory == self.working_directory:
            raise ValidationError(
                "Copy of a container cannot have the same working directory"
                " with the original container."
            )

        save_file_path = os.path.join(working_directory, "save.g00")
        self._options._set_extra_options({"save": save_file_path})
        self._synch_with_gams()
        self._options._set_extra_options({})

        m = Container(load_from=save_file_path, working_directory=working_directory)

        # if already defined equations exist, add them to .gms file
        for equation in self.getEquations():
            if equation._definition is not None:
                m._add_statement(equation._definition)
                symbol = cast("Equation", m[equation.name])
                symbol._definition = equation._definition

        m._synch_with_gams()

        return m

    def serialize(self, path: str) -> None:
        """
        Serializes the Container into a zip file.

        Parameters
        ----------
        path : str
            Path to the zip file.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=range(3))
        >>> gp.serialize(m, "serialization_path.zip")

        """
        gp.serialize(self, path)

    def importExtrinsicLibrary(
        self, lib_path: str, functions: dict[str, str]
    ) -> ExtrinsicLibrary:
        """
        Imports an extrinsic library to the GAMS environment.

        Parameters
        ----------
        lib_path : str
            Path to the .so, .dylib or .dll file that contains the extrinsic library
        functions : dict[str, str]
            Names of the functions as a dictionary. Key is the desired function name in GAMSPy
            and value is the function name in the extrinsic library.

        Returns
        -------
        ExtrinsicLibrary

        Raises
        ------
        FileNotFoundError
            In case the extrinsic library does not exist in the given path.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> # Assuming 'trilib.so' is a valid library path
        >>> # lib = m.importExtrinsicLibrary("trilib.so", {"my_cos": "Cosine", "my_sin": "Sine"})
        >>> # c = lib.my_cos(0)

        """
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"`{lib_path}` is not a valid path.")

        external_lib = ExtrinsicLibrary(self, lib_path, functions)
        self._add_statement(external_lib)

        return external_lib

    def gamsJobName(self) -> str | None:
        """
        Returns the name of the latest GAMS job that was executed

        Returns
        -------
        str | None

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=["seattle", "san-diego"], description="canning plants")
        >>> gams_file_name = f"{m.gamsJobName()}.gms"

        """
        return self._job

    def gdxInputPath(self) -> str:
        """
        Path to the input GDX file

        Returns
        -------
        str

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=["seattle", "san-diego"], description="canning plants")
        >>> gdx_path = m.gdxInputPath()

        """
        return self._gdx_in

    def gdxOutputPath(self) -> str:
        """
        Path to the output GDX file

        Returns
        -------
        str

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["seattle", "san-diego"], description="canning plants")
        >>> ii = gp.Set(m, name="ii", domain=i, description="seattle plant")
        >>> ii['seattle'] = True
        >>> gdx_path = m.gdxOutputPath()

        """
        return self._gdx_out
