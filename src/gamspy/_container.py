from __future__ import annotations

import atexit
import os
import shutil
import sys
import uuid
from typing import TYPE_CHECKING, Any, Literal

import gams.transfer as gt
from gams import DebugLevel, GamsCheckpoint, GamsJob, GamsWorkspace
from gams.core import gdx

import gamspy as gp
import gamspy._miro as miro
import gamspy.utils as utils
from gamspy._backend.backend import backend_factory
from gamspy._extrinsic import ExtrinsicLibrary
from gamspy._miro import MiroJSONEncoder
from gamspy._options import Options
from gamspy._symbols.symbol import Symbol
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    import pandas as pd

    from gamspy import (
        Alias,
        Equation,
        EquationType,
        Model,
        Parameter,
        Set,
        Variable,
    )
    from gamspy._algebra.expression import Expression
    from gamspy._model import Sense

IS_MIRO_INIT = os.getenv("MIRO", False)
MIRO_GDX_IN = os.getenv("GAMS_IDC_GDX_INPUT", None)
MIRO_GDX_OUT = os.getenv("GAMS_IDC_GDX_OUTPUT", None)
MIRO_INPUT_TYPES = (gt.Set, gt.Parameter)

debugging_map = {
    "delete": DebugLevel.Off,
    "keep_on_error": DebugLevel.KeepFilesOnError,
    "keep": DebugLevel.KeepFiles,
}


class Container(gt.Container):
    """
    A container is an object that holds all symbols and operates on them.

    Parameters
    ----------
    load_from : str, optional
        Path to the GDX file to be loaded from, by default None
    system_directory : str, optional
        Path to the directory that holds the GAMS installation, by default None
    working_directory : str, optional
        Path to the working directory to store temporary files such .lst, .gms,
        .gdx, .g00 files.
    debugging_level : str, optional
        Decides on keeping the temporary files generate by GAMS, by default
        "keep_on_error"
    options : Options, optional
        Global options for the overall execution
    miro_protect : bool, optional
        Protects MIRO input symbol records from being re-assigned, by default True

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i")

    """

    def __init__(
        self,
        load_from: str | None = None,
        system_directory: str | None = None,
        working_directory: str | None = None,
        debugging_level: str = "keep_on_error",
        options: Options | None = None,
        miro_protect: bool = True,
    ):
        self._gams_string = ""
        if IS_MIRO_INIT:
            atexit.register(self._write_miro_files)

        system_directory = (
            system_directory
            if system_directory
            else utils._get_gamspy_base_directory()
        )

        self._unsaved_statements: list = []
        self.miro_protect = miro_protect

        # import symbols from arbitrary gams code
        self._import_symbols: list[str] = []

        super().__init__(system_directory=system_directory)

        self._license_path = utils._get_license_path(self.system_directory)

        self.workspace = GamsWorkspace(
            working_directory,
            self.system_directory,
            self._get_debugging_level(debugging_level),
        )

        self.working_directory = self.workspace.working_directory

        (
            self._save_to,
            self._restart_from,
            self._gdx_in,
            self._gdx_out,
        ) = self._setup_paths()

        self._job: GamsJob | None = None
        self._temp_container = gt.Container(
            system_directory=self.system_directory
        )

        if options is not None and not isinstance(options, Options):
            raise TypeError(
                f"`options` must be of type Option but found {type(options)}"
            )
        self._options = Options() if options is None else options

        # needed for miro
        self._miro_input_symbols: list[str] = []
        self._miro_output_symbols: list[str] = []
        if load_from is not None:
            self.read(load_from)
            self._run()

    def __repr__(self):
        return f"<Container ({hex(id(self))})>"

    def __str__(self):
        if len(self):
            return f"<Container ({hex(id(self))}) with {len(self)} symbols: {self.data.keys()}>"

        return f"<Empty Container ({hex(id(self))})>"

    def _write_miro_files(self):  # pragma: no cover
        if len(self._miro_input_symbols) + len(self._miro_output_symbols) == 0:
            return

        # create conf_<model>/<model>_io.json
        encoder = MiroJSONEncoder(self)
        encoder.write_json()

    def _get_debugging_level(self, debugging_level: str):
        if (
            not isinstance(debugging_level, str)
            or debugging_level not in debugging_map
        ):
            raise ValidationError(
                "Debugging level must be one of 'delete', 'keep',"
                " 'keep_on_error'"
            )

        return debugging_map[debugging_level]

    def _write_default_gdx_miro(self):  # pragma: no cover
        # create data_<model>/default.gdx
        symbols = self._miro_input_symbols + self._miro_output_symbols

        filename = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        directory = os.path.dirname(sys.argv[0])
        data_path = os.path.join(directory, f"data_{filename}")
        try:
            os.mkdir(data_path)
        except FileExistsError:
            pass

        for name in symbols:
            self[name]._is_dirty = False

        super().write(
            os.path.join(data_path, "default.gdx"),
            symbols,
        )

    def addGamsCode(
        self, gams_code: str, import_symbols: list[str] = []
    ) -> None:
        """
        Adds an arbitrary GAMS code to the generate .gms file

        Parameters
        ----------
        gams_code : str
            Gams code that you want to insert.
        import_symbols : list[str], optional
            Symbols to be imported to the container from GAMS.

        Examples
        --------
        >>> from gamspy import Container
        >>> m = Container()
        >>> m.addGamsCode("scalar piHalf / [pi/2] /;", import_symbols=["piHalf"])
        >>> m["piHalf"].toValue()
        1.5707963267948966

        """
        if import_symbols is not None and (
            not isinstance(import_symbols, list)
            or any(not isinstance(symbol, str) for symbol in import_symbols)
        ):
            raise ValidationError("import_symbols must be a list of strings")

        self._import_symbols = import_symbols
        self._add_statement(gams_code)

        self._run()

    def _add_statement(self, statement) -> None:
        self._unsaved_statements.append(statement)

    def _cast_symbols(self, symbol_names: list[str] | None = None) -> None:
        """Casts GTP symbols to GAMSpy symbols"""
        symbol_names = symbol_names if symbol_names else list(self.data.keys())

        for symbol_name in symbol_names:
            gtp_symbol = self.data[symbol_name]
            new_domain = [
                (
                    self.data[member.name]
                    if not isinstance(member, str)
                    else member
                )
                for member in gtp_symbol.domain
            ]

            del self.data[symbol_name]

            if isinstance(gtp_symbol, gt.Alias):
                alias_with = self.data[gtp_symbol.alias_with.name]
                _ = gp.Alias._constructor_bypass(
                    self, gtp_symbol._name, alias_with
                )
            elif isinstance(gtp_symbol, gt.UniverseAlias):
                _ = gp.UniverseAlias._constructor_bypass(
                    self,
                    gtp_symbol._name,
                )
            elif isinstance(gtp_symbol, gt.Set):
                _ = gp.Set._constructor_bypass(
                    self,
                    gtp_symbol._name,
                    new_domain,
                    gtp_symbol._is_singleton,
                    gtp_symbol._records,
                    gtp_symbol._description,
                )
            elif isinstance(gtp_symbol, gt.Parameter):
                _ = gp.Parameter._constructor_bypass(
                    self,
                    gtp_symbol._name,
                    new_domain,
                    gtp_symbol._records,
                    gtp_symbol._description,
                )
            elif isinstance(gtp_symbol, gt.Variable):
                _ = gp.Variable._constructor_bypass(
                    self,
                    gtp_symbol._name,
                    gtp_symbol._type,
                    new_domain,
                    gtp_symbol._records,
                    gtp_symbol._description,
                )
            elif isinstance(gtp_symbol, gt.Equation):
                symbol_type = gtp_symbol.type
                if gtp_symbol.type in ["eq", "leq", "geq"]:
                    symbol_type = "regular"

                _ = gp.Equation._constructor_bypass(
                    self,
                    gtp_symbol._name,
                    symbol_type,
                    new_domain,
                    gtp_symbol._records,
                    gtp_symbol._description,
                )

    def _delete_autogenerated_symbols(self):
        """
        Removes autogenerated model attributes, objective variable and equation from
        the container
        """
        autogenerated_symbol_names = self._get_autogenerated_symbol_names()

        for name in autogenerated_symbol_names:
            if name in self.data:
                del self.data[name]

    def _get_symbol_names_from_gdx(self, load_from: str) -> list[str]:
        gdx_handle = utils._open_gdx_file(self.system_directory, load_from)
        _, symbol_count, _ = gdx.gdxSystemInfo(gdx_handle)

        symbol_names = []
        for i in range(1, symbol_count + 1):
            _, symbol_name, _, _ = gdx.gdxSymbolInfo(gdx_handle, i)
            symbol_names.append(symbol_name)

        utils._close_gdx_handle(gdx_handle)

        return symbol_names

    def _get_symbol_names_to_load(
        self,
        load_from: str,
        symbol_names: list[str] | None = None,
    ) -> list[str]:
        if symbol_names is None:
            symbol_names = self._get_symbol_names_from_gdx(load_from)

        return symbol_names

    def _setup_paths(self) -> tuple[GamsCheckpoint, GamsCheckpoint, str, str]:
        suffix = uuid.uuid4()
        save_to = GamsCheckpoint(self.workspace, f"_save_{suffix}.g00")
        restart_from = GamsCheckpoint(self.workspace, f"_restart_{suffix}.g00")
        gdx_in = self.working_directory + os.sep + f"_gdx_in_{suffix}.gdx"
        gdx_out = self.working_directory + os.sep + f"_gdx_out_{suffix}.gdx"

        return save_to, restart_from, gdx_in, gdx_out

    def _get_autogenerated_symbol_names(self) -> list[str]:
        names = []
        for name in self.data:
            if name.startswith(gp.Model._generate_prefix):
                names.append(name)

        return names

    def _get_touched_symbol_names(self) -> tuple[list[str], list[str]]:
        dirty_names = []
        modified_names = []

        for name, symbol in self:
            if isinstance(symbol, gp.UniverseAlias):
                continue

            if symbol._is_dirty and symbol.synchronize:
                dirty_names.append(name)

            if symbol.modified:
                if (
                    isinstance(symbol, gp.Alias)
                    and symbol.alias_with.name not in modified_names
                ):
                    modified_names.append(symbol.alias_with.name)

                modified_names.append(name)

        return dirty_names, modified_names

    def _run(self, keep_flags: bool = False) -> pd.DataFrame | None:
        runner = backend_factory(self, self._options)
        summary = runner.solve(is_implicit=True, keep_flags=keep_flags)

        if self._options and self._options.seed is not None:
            # Required for correct seeding. Seed can only be set in the first run.
            self._options.seed = None

        if IS_MIRO_INIT:
            self._write_default_gdx_miro()  # pragma: no cover

        return summary

    def _swap_checkpoints(self):
        self._restart_from, self._save_to = self._save_to, self._restart_from

    def _get_unload_symbols_str(
        self, dirty_names: list[str], gdx_out: str
    ) -> str:
        # Write dirty names, import symbols and autogenerated names to gdx
        autogenerated_names = self._get_autogenerated_symbol_names()
        unload_names = dirty_names + autogenerated_names + self._import_symbols

        if len(unload_names) == 0:
            return ""

        unload_str = ",".join(unload_names)
        return f"execute_unload '{gdx_out}' {unload_str}\n"

    def _generate_gams_string(
        self,
        gdx_in: str,
        gdx_out: str,
        dirty_names: list[str],
        modified_names: list[str],
    ) -> str:
        string = "$onMultiR\n$onUNDF\n"
        for statement in self._unsaved_statements:
            if isinstance(statement, str):
                string += statement + "\n"
            elif isinstance(statement, gp.UniverseAlias):
                continue
            else:
                string += statement.getDeclaration() + "\n"

        if modified_names:
            string += f"$gdxIn {gdx_in}\n"

        for symbol_name in modified_names:
            symbol = self[symbol_name]
            if isinstance(symbol, Symbol) and not symbol_name.startswith(
                gp.Model._generate_prefix
            ):
                if (
                    isinstance(symbol, MIRO_INPUT_TYPES)
                    and symbol._is_miro_input
                    and not IS_MIRO_INIT
                    and MIRO_GDX_IN
                ):
                    string += miro.get_load_input_str(symbol_name, gdx_in)
                else:
                    string += f"$loadDC {symbol_name}\n"

        if modified_names:
            string += "$gdxIn\n"

        string += "$offUNDF\n"
        string += self._get_unload_symbols_str(dirty_names, gdx_out)

        if self._miro_output_symbols and not IS_MIRO_INIT and MIRO_GDX_OUT:
            string += miro.get_unload_output_str(self)

        self._gams_string += string

        return string

    def read(
        self,
        load_from: str,
        symbol_names: list[str] | None = None,
        load_records: bool = True,
        mode: str | None = None,
        encoding: str | None = None,
    ):
        """
        Reads specified symbols from the gdx file. If symbol_names are
        not provided, it reads all symbols from the gdx file.

        Parameters
        ----------
        load_from : str
        symbol_names : List[str], optional
        load_records : bool
        mode : str, optional
        encoding : str, optional

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=['i1', 'i2'])
        >>> m.write("test.gdx")
        >>> new_container = gp.Container()
        >>> new_container.read("test.gdx")
        >>> new_container.data.keys() == m.data.keys()
        True

        """
        super().read(load_from, symbol_names, load_records, mode, encoding)
        self._cast_symbols(symbol_names)

    def write(
        self,
        write_to: str,
        symbol_names: list[str] | None = None,
        compress: bool = False,
        mode: str | None = None,
        eps_to_zero: bool = True,
    ):
        """
        Writes specified symbols to the gdx file. If symbol_names are
        not provided, it writes all symbols to the gdx file.

        Parameters
        ----------
        write_to : str
        symbol_names : List[str], optional
        compress : bool
        mode : str, optional
        eps_to_zero : bool

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=['i1', 'i2'])
        >>> m.write("test.gdx")

        """
        dirty_names, _ = self._get_touched_symbol_names()

        if dirty_names:
            self._run(keep_flags=True)

        super().write(
            write_to,
            symbol_names,
            compress,
            mode=mode,
            eps_to_zero=eps_to_zero,
        )

    def generateGamsString(self, show_raw: bool = False) -> str:
        """
        Generates the GAMS code

        Parameters
        ----------
        show_raw : bool, optional
            Shows the raw model without data and other necessary
            GAMS statements, by default False.

        Returns
        -------
        str
        """
        if not show_raw:
            return self._gams_string

        return utils._filter_gams_string(self._gams_string)

    def loadRecordsFromGdx(
        self,
        load_from: str,
        symbol_names: list[str] | None = None,
    ):
        """
        Loads data of the given symbols from a gdx file. If no
        symbol names are given, data of all symbols are loaded.

        Parameters
        ----------
        load_from : str
            Path to the gdx file
        symbols : List[str], optional
            Symbols whose data will be load from gdx, by default None

        Examples
        --------
        >>> from gamspy import Container, Set
        >>> m = Container()
        >>> i = Set(m, "i", records=["i1", "i2"])
        >>> m.write("test.gdx")
        >>> m2 = Container()
        >>> m2.loadRecordsFromGdx("test.gdx")
        >>> print(i.records.equals(m2["i"].records))
        True

        """
        self._load_records_from_gdx(load_from, symbol_names, user_invoked=True)

    def addAlias(self, name: str, alias_with: Set | Alias) -> Alias:
        """
        Creates a new Alias and adds it to the container

        Parameters
        ----------
        name : str
        alias_with : Set | Alias

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
        return gp.Alias(self, name, alias_with)

    def addSet(
        self,
        name: str,
        domain: list[Set | str] | None = None,
        is_singleton: bool = False,
        records: Any | None = None,
        domain_forwarding: bool = False,
        description: str = "",
        uels_on_axes: bool = False,
    ) -> Set:
        """
        Creates a Set and adds it to the container

        Parameters
        ----------
        name : str
        domain : List[Set | str], optional
        is_singleton : bool, optional
        records : Any, optional
        domain_forwarding : bool, optional
        description : str, optional
        uels_on_axes : bool, optional

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
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = m.addSet("i")

        """
        return gp.Set(
            self,
            name,
            domain,
            is_singleton,
            records,
            domain_forwarding,
            description,
            uels_on_axes,
        )

    def addParameter(
        self,
        name: str,
        domain: list[str | Set] | None = None,
        records: Any | None = None,
        domain_forwarding: bool = False,
        description: str = "",
        uels_on_axes: bool = False,
    ) -> Parameter:
        """
        Creates a Parameter and adds it to the Container

        Parameters
        ----------
        name : str
        domain : List[str | Set]], optional
        records : Any, optional
        domain_forwarding : bool, optional
        description : str, optional
        uels_on_axes : bool, optional

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
        return gp.Parameter(
            self,
            name,
            domain,
            records,
            domain_forwarding,
            description,
            uels_on_axes,
        )

    def addVariable(
        self,
        name: str,
        type: str = "free",
        domain: list[str | Set] | None = None,
        records: Any | None = None,
        domain_forwarding: bool = False,
        description: str = "",
        uels_on_axes: bool = False,
    ) -> Variable:
        """
        Creates a Variable and adds it to the Container

        Parameters
        ----------
        name : str
        type : str, optional
        domain : List[str | Set]], optional
        records : Any, optional
        domain_forwarding : bool, optional
        description : str, optional
        uels_on_axes : bool, optional

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
        return gp.Variable(
            self,
            name,
            type,
            domain,
            records,
            domain_forwarding,
            description,
            uels_on_axes,
        )

    def addEquation(
        self,
        name: str,
        type: str | EquationType = "regular",
        domain: list[Set | str] | None = None,
        definition: Expression | None = None,
        records: Any | None = None,
        domain_forwarding: bool = False,
        description: str = "",
        uels_on_axes: bool = False,
        definition_domain: list[Set | str] | None = None,
    ) -> Equation:
        """
        Creates an Equation and adds it to the Container

        Parameters
        ----------
        name : str
        type : str
        domain : List[Set | str], optional
        definition : Definition, optional
        records : Any, optional
        domain_forwarding : bool, optional
        description : str, optional
        uels_on_axes : bool, optional
        definition_domain : List[Set | str], optional

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
            definition_domain,
        )

    def addModel(
        self,
        name: str,
        problem: str,
        equations: list[Equation] = [],
        sense: Literal["MIN", "MAX"] | Sense | None = None,
        objective: Variable | Expression | None = None,
        matches: dict | None = None,
        limited_variables: list | None = None,
    ) -> Model:
        """
        Creates a Model and adds it to the Container

        Parameters
        ----------
        name : str
        equations : List[Equation]
        problem : str
        sense : "MIN", "MAX", optional
        objective : Variable | Expression, optional
        matches : dict, optional
        limited_variables : list, optional

        Returns
        -------
        Model

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> e = gp.Equation(m, "e")
        >>> model = m.addModel("my_model", "LP", [e])

        """
        return gp.Model(
            self,
            name,
            problem,
            equations,
            sense,
            objective,
            matches,
            limited_variables,
        )

    def copy(self, working_directory: str) -> Container:
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
        m = Container(working_directory=working_directory)
        if m.working_directory == self.working_directory:
            raise ValidationError(
                "Copy of a container cannot have the same working directory"
                " with the original container."
            )

        self._run()

        for name, symbol in self:
            new_domain = []
            for elem in symbol.domain:
                if not isinstance(elem, str):
                    new_set = gp.Set._constructor_bypass(
                        m,
                        elem.name,
                        elem.domain,
                        elem.is_singleton,
                        elem.records,
                        elem.description,
                    )
                    new_domain.append(new_set)
                else:
                    new_domain.append(elem)

            if isinstance(symbol, gp.Alias):
                alias_with = gp.Set._constructor_bypass(
                    m,
                    symbol.alias_with.name,
                    symbol.alias_with.domain,
                    symbol.alias_with.is_singleton,
                    symbol.alias_with.records,
                )
                _ = gp.Alias._constructor_bypass(
                    m,
                    name,
                    alias_with,
                )
            elif isinstance(symbol, gp.UniverseAlias):
                _ = gp.UniverseAlias._constructor_bypass(
                    m,
                    name,
                )
            elif isinstance(symbol, gp.Set):
                _ = gp.Set._constructor_bypass(
                    m,
                    name,
                    new_domain,
                    symbol.is_singleton,
                    symbol._records,
                    symbol.description,
                )
            elif isinstance(symbol, gp.Parameter):
                _ = gp.Parameter._constructor_bypass(
                    m,
                    name,
                    new_domain,
                    symbol._records,
                    symbol.description,
                )
            elif isinstance(symbol, gp.Variable):
                _ = gp.Variable._constructor_bypass(
                    m,
                    name,
                    symbol.type,
                    new_domain,
                    symbol._records,
                    symbol.description,
                )
            elif isinstance(symbol, gp.Equation):
                symbol_type = symbol.type
                if symbol.type in ["eq", "leq", "geq"]:
                    symbol_type = "regular"
                _ = gp.Equation._constructor_bypass(
                    container=m,
                    name=name,
                    type=symbol_type,
                    domain=new_domain,
                    records=symbol._records,
                    description=symbol.description,
                )

        try:
            shutil.copy(
                self._save_to._checkpoint_file_name,
                m._save_to._checkpoint_file_name,
            )
        except FileNotFoundError:
            # save_to might not exist and it's fine
            pass

        shutil.copy(
            self._restart_from._checkpoint_file_name,
            m._restart_from._checkpoint_file_name,
        )

        shutil.copy(self._gdx_in, m._gdx_in)
        try:
            shutil.copy(self._gdx_out, m._gdx_out)
        except FileNotFoundError:
            pass

        # if already defined equations exist, add them to .gms file
        for equation in self.getEquations():
            if equation._definition is not None:
                m._add_statement(equation._definition)

        return m

    def getEquations(self) -> list[Equation]:
        """
        Returns all equation symbols in the Container.

        Returns
        -------
        list[Equation]
        """
        equations = [
            equation
            for equation in self.listEquations()
            if not equation.startswith(gp.Model._generate_prefix)
        ]
        return self.getSymbols(equations)

    def _load_records_from_gdx(
        self,
        load_from: str,
        symbol_names: list[str] | None = None,
        user_invoked: bool = False,
    ):
        symbol_names = self._get_symbol_names_to_load(load_from, symbol_names)

        self._temp_container.read(load_from, symbol_names)

        for name in symbol_names:
            if name in self.data:
                updated_records = self._temp_container[name].records

                self[name]._records = updated_records

                if updated_records is not None:
                    self[name]._domain_labels = self[name].domain_names
            else:
                self.read(load_from, [name])

            if user_invoked:
                self[name].modified = True

        self._temp_container.data = {}

        if user_invoked:
            self._run()

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
        """
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"`{lib_path}` is not a valid path.")

        external_lib = ExtrinsicLibrary(lib_path, functions)
        self._add_statement(external_lib)

        return external_lib

    def gamsJobName(self) -> str | None:
        """
        Returns the name of the latest GAMS job that was executed

        Returns
        -------
        str | None
        """
        return self._job.name if self._job is not None else None

    def gdxInputPath(self) -> str:
        """
        Path to the input gdx file

        Returns
        -------
        str
        """
        return self._gdx_in

    def gdxOutputPath(self) -> str:
        """
        Path to the output gdx file

        Returns
        -------
        str
        """
        return self._gdx_out
