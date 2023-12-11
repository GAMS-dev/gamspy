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

import io
import os
import shutil
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
from gams import GamsOptions
from gams import GamsWorkspace
from gams.control.workspace import GamsException
from gams.control.workspace import GamsExceptionExecution
from gams.core import gdx

import gamspy as gp
import gamspy._engine as engine
import gamspy._neos as neos
import gamspy.utils as utils
from gamspy._model import ModelStatus
from gamspy._options import _map_options
from gamspy.exceptions import customize_exception
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
    from gamspy._engine import EngineConfig
    from gamspy._options import Options


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
    ):
        system_directory = (
            system_directory
            if system_directory
            else utils._get_gamspy_base_directory()
        )

        self._delayed_execution = delayed_execution
        self._unsaved_statements: list = []
        self._is_first_run = True

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

    def _addGamsCode(self, gams_code: str, import_symbols: List[str] = []):
        if import_symbols is not None and (
            not isinstance(import_symbols, list)
            or any(not isinstance(symbol, str) for symbol in import_symbols)
        ):
            raise GamspyException("import_symbols must be a list of strings")

        self._import_symbols = import_symbols
        self._unsaved_statements.append(gams_code)

    def _add_statement(self, statement) -> None:
        self._unsaved_statements.append(statement)

    def _cast_symbols(self, symbol_names: Optional[List[str]] = None) -> None:
        """
        Casts all symbols in the GAMS Transfer container to GAMSpy symbols
        """
        symbol_names = symbol_names if symbol_names else list(self.data.keys())

        for symbol_name in symbol_names:
            symbol = self.data[symbol_name]
            new_domain = [
                self.data[set.name] if not isinstance(set, str) else set
                for set in symbol.domain
            ]

            del self.data[symbol_name]

            if isinstance(symbol, gt.Alias):
                alias_with = self[symbol.alias_with.name]
                _ = gp.Alias(
                    self,
                    symbol.name,
                    alias_with,
                )
            elif isinstance(symbol, gt.UniverseAlias):
                _ = gp.UniverseAlias(
                    self,
                    symbol.name,
                )
            elif isinstance(symbol, gt.Set):
                _ = gp.Set(
                    self,
                    symbol.name,
                    new_domain,
                    symbol.is_singleton,
                    symbol._records,
                    symbol.domain_forwarding,
                    symbol.description,
                )
            elif isinstance(symbol, gt.Parameter):
                _ = gp.Parameter(
                    self,
                    symbol.name,
                    new_domain,
                    symbol._records,
                    symbol.domain_forwarding,
                    symbol.description,
                )
            elif isinstance(symbol, gt.Variable):
                _ = gp.Variable(
                    self,
                    symbol.name,
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
                    container=self,
                    name=symbol.name,
                    type=symbol_type,
                    domain=new_domain,
                    records=symbol._records,
                    domain_forwarding=symbol.domain_forwarding,
                    description=symbol.description,
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

        return dirty_names, modified_names

    def _clean_dirty_symbols(self, dirty_names: List[str]):
        for name in dirty_names:
            self[name]._is_dirty = False

    def _update_modified_state(self, modified_names: List[str]):
        for name in modified_names:
            self[name].modified = False

    def _run(
        self,
        options: Optional[GamsOptions] = None,
        output: Optional[io.TextIOWrapper] = None,
        backend: Literal["local", "engine", "neos"] = "local",
        engine_config: Optional[EngineConfig] = None,
        neos_client: Optional[neos.NeosClient] = None,
        create_log_file: bool = False,
        is_implicit: bool = False,
    ) -> Union[pd.DataFrame, None]:
        if options is None:
            options = _map_options(
                self.workspace,
                backend=backend,
                options=None,
                global_options=self._options,
                is_seedable=self._is_first_run,
                output=output,
                create_log_file=create_log_file,
            )

        dirty_names, modified_names = self._get_touched_symbol_names()
        gams_string = self._generate_gams_string(backend, dirty_names)

        # Create gdx file to read records from
        self._clean_dirty_symbols(dirty_names)
        self._update_modified_state(modified_names)
        self.isValid(verbose=True, force=True)
        super().write(self._gdx_in, modified_names)

        # If there is no restart checkpoint, set it to None
        checkpoint = self._restart_from if not self._is_first_run else None

        self._job = GamsJob(
            self.workspace,
            job_name=f"_job_{uuid.uuid4()}",
            source=gams_string,
            checkpoint=checkpoint,
        )

        if backend == "local":
            self._run_local(options, output)
        elif backend == "engine":
            assert engine_config
            engine.run(self, options, output, engine_config)
        elif backend == "neos":
            assert neos_client
            neos.run(self, gams_string, options, neos_client)
            if not neos_client.is_blocking:
                return None

        self.loadRecordsFromGdx(
            self._gdx_out, dirty_names + self._import_symbols
        )
        self._restart_from, self._save_to = self._save_to, self._restart_from
        self._is_first_run = False

        return self._prepare_summary(
            is_implicit, options, backend, engine_config
        )

    def _prepare_summary(
        self,
        is_implicit: bool,
        options: GamsOptions,
        backend: str,
        engine_config: EngineConfig | None,
    ) -> Union[pd.DataFrame, None]:
        if is_implicit or options.traceopt != 3:
            return None

        if backend == "engine":
            if engine_config is None:
                return None
            else:
                if engine_config.remove_results:
                    return None

        solve_stat = [
            "",
            "Normal",
            "Iteration",
            "Resource",
            "Solver",
            "EvalError",
            "Capability",
            "License",
            "User",
            "SetupErr",
            "SolverErr",
            "InternalErr",
            "Skipped",
            "SystemErr",
        ]
        HEADER = [
            "Solver Status",
            "Model Status",
            "Objective",
            "Num of Equations",
            "Num of Variables",
            "Model Type",
            "Solver",
            "Solver Time",
        ]
        with open(options.trace) as file:
            line = file.readlines()[-1]
            (
                _,
                model_type,
                solver_name,
                _,
                _,
                _,
                _,
                num_equations,
                num_variables,
                _,
                _,
                _,
                _,
                model_status,
                solver_status,
                objective_value,
                _,
                solver_time,
                _,
                _,
                _,
                _,
            ) = line.split(",")

        dataframe = pd.DataFrame(
            [
                [
                    solve_stat[int(solver_status)],
                    ModelStatus(int(model_status)).name,
                    objective_value,
                    num_equations,
                    num_variables,
                    model_type,
                    solver_name,
                    solver_time,
                ]
            ],
            columns=HEADER,
        )
        return dataframe

    def _run_local(
        self,
        options: GamsOptions,
        output: Union[io.TextIOWrapper, None],
    ):
        try:
            self._job.run(  # type: ignore
                gams_options=options,
                checkpoint=self._save_to,
                create_out_db=False,
                output=output,
            )
        except GamsExceptionExecution as exception:
            exception = customize_exception(
                self.workspace, options, self._job, exception
            )
            raise exception
        finally:
            self._unsaved_statements = []

    def _run_engine(
        self,
        options: GamsOptions,
        output: Union[io.TextIOWrapper, None],
        engine_config: Union[EngineConfig, None],
    ):
        options.previouswork = 1  # In case GAMS version differs on Engine

        assert engine_config

        extra_model_files = engine_config._preprocess_extra_model_files(
            self.workspace, self._gdx_in
        )

        try:
            self._job.run_engine(  # type: ignore
                engine_configuration=engine_config._get_engine_config(),
                extra_model_files=extra_model_files,
                gams_options=options,
                checkpoint=self._save_to,
                output=output,
                create_out_db=False,
                engine_options=engine_config.engine_options,
                remove_results=engine_config.remove_results,
            )
        except GamsException as e:
            raise GamspyException(str(e))
        finally:
            self._unsaved_statements = []
            options.forcework = 0

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

    def gdxInputName(self) -> str:
        """
        Name of the input gdx file

        Returns
        -------
        str
        """
        return os.path.basename(self._gdx_in)

    def gdxOutputName(self) -> str:
        """
        Name of the output gdx file

        Returns
        -------
        str
        """
        return os.path.basename(self._gdx_out)

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
        sense : Optional[Literal[MIN, MAX]], optional
        objective : Optional[Union[Variable, Expression]], optional
        matches : Optional[dict], optional
        limited_variables : Optional[list], optional

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

        self._run(is_implicit=True)

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
        return self._generate_gams_string()

    def _get_unload_symbols_str(
        self, dirty_names: List[str], gdx_out: str
    ) -> str:
        # Write dirty names, import symbols and autogenerated names to gdx
        autogenerated_names = self._get_autogenerated_symbol_names()
        unload_names = dirty_names + autogenerated_names + self._import_symbols

        unload_str = ",".join(unload_names)
        return f"execute_unload '{gdx_out}' {unload_str}\n"

    def _preprocess_gdx_paths(self, backend: str) -> Tuple[str, str]:
        if backend == "engine":
            return (
                os.path.basename(self._gdx_in),
                os.path.basename(self._gdx_out),
            )
        elif backend == "neos":
            return "in.gdx", "output.gdx"

        return self._gdx_in, self._gdx_out

    def _generate_gams_string(
        self,
        backend: str = "local",
        dirty_names: List[str] = [],
    ) -> str:
        LOAD_SYMBOL_TYPES = (gp.Set, gp.Parameter, gp.Variable, gp.Equation)
        gdx_in, gdx_out = self._preprocess_gdx_paths(backend)

        string = f"$onMultiR\n$onUNDF\n$gdxIn {gdx_in}\n"
        for statement in self._unsaved_statements:
            if isinstance(statement, str):
                string += statement + "\n"
            elif isinstance(statement, gp.UniverseAlias):
                continue
            else:
                string += statement.getStatement() + "\n"

                if isinstance(statement, LOAD_SYMBOL_TYPES):
                    string += f"$load {statement.name}\n"

        string += "$offUNDF\n$gdxIn\n"
        string += self._get_unload_symbols_str(dirty_names, gdx_out)

        return string

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

        temp_container = Container(system_directory=self.system_directory)
        temp_container.read(load_from, symbol_names, cast_to_gamspy=False)

        for name in symbol_names:
            if name in self.data.keys():
                updated_records = temp_container[name]._records

                self[name]._records = updated_records
                if updated_records is not None:
                    self[name]._domain_labels = self[name].domain_names
            else:
                self.read(load_from, [name])

    def read(
        self,
        load_from: str,
        symbol_names: Optional[List[str]] = None,
        cast_to_gamspy: bool = True,
    ) -> None:
        """
        Reads specified symbols from the gdx file. If symbol_names are
        not provided, it reads all symbols from the gdx file.

        Parameters
        ----------
        load_from : str
        symbol_names : List[str], optional

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
        super().read(load_from, symbol_names)

        if cast_to_gamspy:
            self._cast_symbols(symbol_names)

    def write(
        self,
        write_to: str,
        symbols: Optional[List[str]] = None,
    ) -> None:
        """
        Writes specified symbols to the gdx file. If symbol_names are
        not provided, it writes all symbols to the gdx file.

        Parameters
        ----------
        write_to : str
        symbols : List[str], optional

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=['i1', 'i2'])
        >>> m.write("test.gdx")

        """
        dirty_names, modified_names = self._get_touched_symbol_names()

        if len(dirty_names) > 0:
            self._run(is_implicit=True)

        super().write(write_to, symbols)

        self._update_modified_state(modified_names)
