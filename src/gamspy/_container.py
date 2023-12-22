#
# GAMS - General Algebraic Modeling System Python API
#
# Copyright (c) 2023 GAMS Development Corp. <support@gams.com>
# Copyright (c) 2023 GAMS Software GmbH <support@gams.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
from __future__ import annotations

import os
import shutil
import sys
import uuid
from typing import Any
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union

import gams.transfer as gt
import pandas as pd
from gams import DebugLevel
from gams import GamsCheckpoint
from gams import GamsJob
from gams import GamsWorkspace
from gams.core import gdx

import gamspy as gp
import gamspy.utils as utils
from gamspy._backend.backend import backend_factory
from gamspy._miro import MiroJSONEncoder
from gamspy._options import _map_options
from gamspy.exceptions import GamspyException

if TYPE_CHECKING:
    from gamspy import (
        Alias,
        Set,
        Parameter,
        Variable,
        Equation,
        EquationType,
        Model,
    )
    from gamspy._algebra.expression import Expression
    from gamspy._options import Options

IS_MIRO_INIT = os.getenv("MIRO", False)

MIRO_GDX_IN = os.getenv("GAMS_IDC_GDX_INPUT", None)
MIRO_GDX_OUT = os.getenv("GAMS_IDC_GDX_OUTPUT", None)


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
    delayed_execution : bool, optional
        Delayed execution mode, by default False
    options : Options
        Global options for the overall execution
    miro_protect : bool
        Protects MIRO input symbol records from being re-assigned

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i")

    """

    def __init__(
        self,
        load_from: Optional[str] = None,
        system_directory: Optional[str] = None,
        working_directory: Optional[str] = None,
        delayed_execution: bool = False,
        options: Optional[Options] = None,
        miro_protect: bool = True,
    ):
        system_directory = (
            system_directory
            if system_directory
            else utils._get_gamspy_base_directory()
        )

        self._delayed_execution = delayed_execution
        self._unsaved_statements: list = []
        self._is_first_run = True
        self.miro_protect = miro_protect

        # import symbols from arbitrary gams code
        self._import_symbols: List[str] = []

        super().__init__(load_from, system_directory)

        self.workspace = GamsWorkspace(
            working_directory, self.system_directory, DebugLevel.KeepFiles
        )

        self.working_directory = self.workspace.working_directory

        (
            self._save_to,
            self._restart_from,
            self._gdx_in,
            self._gdx_out,
        ) = self._setup_paths()

        self._job: Optional[GamsJob] = None
        self._options = options

        # needed for miro
        self._miro_input_symbols: List[str] = []
        self._miro_output_symbols: List[str] = []
        self._first_destruct = True

    def __del__(self):
        if (
            not IS_MIRO_INIT
            or not self._first_destruct
            or len(self._miro_input_symbols) + len(self._miro_output_symbols)
            == 0
        ):
            return

        self._first_destruct = False
        # create conf_<model>/<model>_io.json
        encoder = MiroJSONEncoder(
            self,
            self._miro_input_symbols,
            self._miro_output_symbols,
        )

        encoder.writeJson()

        # create data_<model>/default.gdx
        symbols = list(
            set(self._miro_input_symbols + self._miro_output_symbols)
        )

        filename = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        directory = os.path.dirname(sys.argv[0])
        data_path = os.path.join(directory, f"data_{filename}")
        try:
            os.mkdir(data_path)
        except FileExistsError:
            pass

        super().write(
            os.path.join(data_path, "default.gdx"),
            symbols,
        )

    def _addGamsCode(
        self, gams_code: str, import_symbols: List[str] = []
    ) -> None:
        """
        Adds an arbitrary GAMS code to the generate .gms file

        Parameters
        ----------
        gams_code : str
        import_symbols : List[str], optional
        """
        if import_symbols is not None and (
            not isinstance(import_symbols, list)
            or any(not isinstance(symbol, str) for symbol in import_symbols)
        ):
            raise GamspyException("import_symbols must be a list of strings")

        self._import_symbols = import_symbols
        self._unsaved_statements.append(gams_code)

    def _add_statement(self, statement) -> None:
        self._unsaved_statements.append(statement)

    def _assign_symbol_attributes(
        self,
        gp_symbol: Union["Set", "Parameter", "Variable", "Equation"],
        gtp_symbol: Union[
            "gt.Set", "gt.Parameter", "gt.Variable", "gt.Equation"
        ],
        domain: List[Union[str, "Set", "Alias"]],
    ):
        gp_symbol._domain = domain
        gp_symbol._records = gtp_symbol._records
        gp_symbol._domain_forwarding = gtp_symbol._domain_forwarding
        gp_symbol._description = gtp_symbol._description

    def _cast_symbols(self, symbol_names: Optional[List[str]] = None) -> None:
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
                _ = gp.Alias(
                    self,
                    gtp_symbol.name,
                    alias_with,
                )
            elif isinstance(gtp_symbol, gt.UniverseAlias):
                _ = gp.UniverseAlias(
                    self,
                    gtp_symbol.name,
                )
            elif isinstance(gtp_symbol, gt.Set):
                gp_symbol = gp.Set(
                    self,
                    gtp_symbol.name,
                )

                gp_symbol._is_singleton = gtp_symbol.is_singleton
                self._assign_symbol_attributes(
                    gp_symbol, gtp_symbol, new_domain
                )
            elif isinstance(gtp_symbol, gt.Parameter):
                gp_symbol = gp.Parameter(
                    self,
                    gtp_symbol.name,
                )
                self._assign_symbol_attributes(
                    gp_symbol, gtp_symbol, new_domain
                )
            elif isinstance(gtp_symbol, gt.Variable):
                gp_symbol = gp.Variable(
                    self,
                    gtp_symbol.name,
                    gtp_symbol.type,
                )
                self._assign_symbol_attributes(
                    gp_symbol, gtp_symbol, new_domain
                )
            elif isinstance(gtp_symbol, gt.Equation):
                symbol_type = gtp_symbol.type
                if gtp_symbol.type in ["eq", "leq", "geq"]:
                    symbol_type = "regular"
                gp_symbol = gp.Equation(
                    container=self,
                    name=gtp_symbol.name,
                    type=symbol_type,
                )
                self._assign_symbol_attributes(
                    gp_symbol, gtp_symbol, new_domain
                )

    def _get_symbol_names_from_gdx(self, load_from: str) -> List[str]:
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
        symbol_names: Optional[List[str]] = None,
    ) -> List[str]:
        if symbol_names is None or symbol_names == []:
            symbol_names = self._get_symbol_names_from_gdx(load_from)

        return symbol_names

    def _interrupt(self):
        if self._job:
            self._job.interrupt()
        else:
            raise GamspyException("There is no job initialized.")

    def _setup_paths(
        self,
    ) -> Tuple[GamsCheckpoint, GamsCheckpoint, str, str]:
        suffix = uuid.uuid4()
        save_to = GamsCheckpoint(self.workspace, f"_save_{suffix}.g00")
        restart_from = GamsCheckpoint(self.workspace, f"_restart_{suffix}.g00")
        gdx_in = self.working_directory + os.sep + f"_gdx_in_{suffix}.gdx"
        gdx_out = self.working_directory + os.sep + f"_gdx_out_{suffix}.gdx"

        return save_to, restart_from, gdx_in, gdx_out

    def _get_autogenerated_symbol_names(self) -> List[str]:
        names = []
        for name in self.data.keys():
            if name.startswith(gp.Model._generate_prefix):
                names.append(name)

        return names

    def _get_touched_symbol_names(self) -> Tuple[List[str], List[str]]:
        dirty_names = []
        modified_names = []

        for name, symbol in self:
            if isinstance(symbol, gp.UniverseAlias):
                continue

            if symbol._is_dirty:
                dirty_names.append(name)

            if symbol.modified:
                modified_names.append(name)

            # miro input symbols should always be assigned to catch domain violations
            if (
                isinstance(symbol, (gp.Set, gp.Parameter))
                and symbol._is_miro_input
                and name not in modified_names
            ):
                modified_names.append(name)

        return dirty_names, modified_names

    def _run(self, keep_flags: bool = False) -> Union[pd.DataFrame, None]:
        options = _map_options(
            self.workspace,
            global_options=self._options,
            is_seedable=self._is_first_run,
        )

        runner = backend_factory(self, options)

        summary = runner.solve(is_implicit=True, keep_flags=keep_flags)

        self._is_first_run = False

        return summary

    def _swap_checkpoints(self):
        self._restart_from, self._save_to = self._save_to, self._restart_from

    def _get_unload_symbols_str(
        self, dirty_names: List[str], gdx_out: str
    ) -> str:
        # Write dirty names, import symbols and autogenerated names to gdx
        autogenerated_names = self._get_autogenerated_symbol_names()
        unload_names = dirty_names + autogenerated_names + self._import_symbols

        unload_str = ",".join(unload_names)
        return f"execute_unload '{gdx_out}' {unload_str}\n"

    def _get_load_miro_input_str(self, statement, gdx_in):
        string = "$gdxIn\n"  # close the old one
        string += f"$gdxIn {MIRO_GDX_IN}\n"  # open the new one
        string += f"$load {statement.name}\n"
        string += "$gdxIn\n"  # close the new one
        string += f"$gdxIn {gdx_in}\n"

        return string

    def _get_unload_miro_symbols_str(self):
        unload_str = ",".join(self._miro_output_symbols)
        return f"execute_unload '{MIRO_GDX_OUT}' {unload_str}\n"

    def _generate_gams_string(
        self,
        gdx_in: str,
        gdx_out: str,
        dirty_names: List[str],
        modified_names: List[str],
    ) -> str:
        LOAD_SYMBOL_TYPES = (gp.Set, gp.Parameter, gp.Variable, gp.Equation)
        MIRO_INPUT_TYPES = (gp.Set, gp.Parameter)
        MIRO_OUTPUT_TYPES = LOAD_SYMBOL_TYPES

        string = f"$onMultiR\n$onUNDF\n$gdxIn {gdx_in}\n"
        for statement in self._unsaved_statements:
            if isinstance(statement, str):
                string += statement + "\n"
            elif isinstance(statement, gp.UniverseAlias):
                continue
            else:
                string += statement.getStatement() + "\n"

                if isinstance(statement, LOAD_SYMBOL_TYPES):
                    if (
                        isinstance(statement, MIRO_INPUT_TYPES)
                        and statement._is_miro_input
                    ):
                        if not IS_MIRO_INIT and MIRO_GDX_IN:
                            self.loadRecordsFromGdx(
                                MIRO_GDX_IN, [statement.name]
                            )
                            string += self._get_load_miro_input_str(
                                statement, gdx_in
                            )
                        else:
                            string += f"$load {statement.name}\n"

                        self._miro_input_symbols.append(statement.name)
                    else:
                        string += f"$load {statement.name}\n"

                # add miro output symbol
                if (
                    isinstance(statement, MIRO_OUTPUT_TYPES)
                    and statement._is_miro_output
                ):
                    self._miro_output_symbols.append(statement.name)

        for symbol_name in modified_names:
            if not isinstance(self[symbol_name], gp.Alias) and (
                not hasattr(self[symbol_name], "_is_miro_input")
                or not self[symbol_name]._is_miro_input
            ):
                string += f"$load {symbol_name}\n"

        string += "$offUNDF\n$gdxIn\n"
        string += self._get_unload_symbols_str(dirty_names, gdx_out)

        if self._miro_output_symbols and not IS_MIRO_INIT and MIRO_GDX_OUT:
            string += self._get_unload_miro_symbols_str()

        return string

    @property
    def delayed_execution(self) -> bool:
        """
        Delayed execution mode.

        Returns
        -------
        bool
        """
        return self._delayed_execution

    def gamsJobName(self) -> Union[str, None]:
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

    def addAlias(self, name: str, alias_with: Union[Set, Alias]) -> Alias:
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
        domain: Optional[List[Union[Set, str]]] = None,
        is_singleton: bool = False,
        records: Optional[Any] = None,
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
        domain: Optional[List[Union[str, Set]]] = None,
        records: Optional[Any] = None,
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
        domain: Optional[List[Union[str, Set]]] = None,
        records: Optional[Any] = None,
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
        type: Union[str, EquationType] = "regular",
        domain: Optional[List[Union[Set, str]]] = None,
        definition: Optional[Expression] = None,
        records: Optional[Any] = None,
        domain_forwarding: bool = False,
        description: str = "",
        uels_on_axes: bool = False,
        definition_domain: Optional[List[Union[Set, str]]] = None,
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
        equations: List[Equation],
        sense: Optional[Literal["MIN", "MAX"]] = None,
        objective: Optional[Union[Variable, Expression]] = None,
        matches: Optional[dict] = None,
        limited_variables: Optional[list] = None,
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
        GamspyException

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
            raise GamspyException(
                "Copy of a container cannot have the same working directory"
                " with the original container."
            )

        self._run()

        for name, symbol in self:
            new_domain = []
            for set in symbol.domain:
                if not isinstance(set, str):
                    new_set = gp.Set(
                        m,
                        set.name,
                        set.domain,
                        set.is_singleton,
                        set.records,
                        set.domain_forwarding,
                        set.description,
                    )
                    new_domain.append(new_set)
                else:
                    new_domain.append(set)

            if isinstance(symbol, gt.Alias):
                alias_with = gp.Set(
                    m,
                    symbol.alias_with.name,
                    symbol.alias_with.domain,
                    symbol.alias_with.is_singleton,
                    symbol.alias_with.records,
                )
                _ = gp.Alias(
                    m,
                    name,
                    alias_with,
                )
            elif isinstance(symbol, gt.UniverseAlias):
                _ = gp.UniverseAlias(
                    m,
                    name,
                )
            elif isinstance(symbol, gt.Set):
                _ = gp.Set(
                    m,
                    name,
                    new_domain,
                    symbol.is_singleton,
                    symbol._records,
                    symbol.domain_forwarding,
                    symbol.description,
                )
            elif isinstance(symbol, gt.Parameter):
                _ = gp.Parameter(
                    m,
                    name,
                    new_domain,
                    symbol._records,
                    symbol.domain_forwarding,
                    symbol.description,
                )
            elif isinstance(symbol, gt.Variable):
                _ = gp.Variable(
                    m,
                    name,
                    symbol.type,
                    new_domain,
                    symbol._records,
                    symbol.domain_forwarding,
                    symbol.description,
                )
            elif isinstance(symbol, gt.Equation):
                symbol_type = symbol.type
                if symbol.type in ["eq", "leq", "geq"]:
                    symbol_type = "regular"
                _ = gp.Equation(
                    container=m,
                    name=name,
                    type=symbol_type,
                    domain=new_domain,
                    records=symbol._records,
                    domain_forwarding=symbol.domain_forwarding,
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
        shutil.copy(self._gdx_out, m._gdx_out)

        # if already defined equations exist, add them to .gms file
        for equation in self.getEquations():
            if equation._definition is not None:
                m._add_statement(equation._definition)

        return m

    def generateGamsString(self) -> str:
        """
        Generates the GAMS code

        Returns
        -------
        str
        """
        return self._generate_gams_string(self._gdx_in, self._gdx_out, [], [])

    def loadRecordsFromGdx(
        self,
        load_from: str,
        symbol_names: Optional[List[str]] = None,
    ) -> None:
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
        symbol_names = self._get_symbol_names_to_load(load_from, symbol_names)

        temp_container = gt.Container(system_directory=self.system_directory)
        temp_container.read(load_from, symbol_names)

        for name in symbol_names:
            if name in self.data.keys():
                updated_records = temp_container[name].records

                self[name]._records = updated_records
                if updated_records is not None:
                    self[name]._domain_labels = self[name].domain_names
            else:
                self.read(load_from, [name])

    def read(
        self,
        load_from: str,
        symbol_names: Optional[List[str]] = None,
        load_records: bool = True,
        mode: Optional[str] = None,
        encoding: Optional[str] = None,
    ) -> None:
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
        symbol_names: Optional[List[str]] = None,
        compress: bool = False,
        mode: Optional[str] = None,
        eps_to_zero: bool = True,
    ) -> None:
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

        if len(dirty_names) > 0:
            self._run(keep_flags=True)

        super().write(
            write_to,
            symbol_names,
            compress,
            mode=mode,
            eps_to_zero=eps_to_zero,
        )
