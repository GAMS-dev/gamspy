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
import io
import os
import shutil
from typing import Any
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union

import gams.transfer as gt
from gams import DebugLevel
from gams import GamsCheckpoint
from gams import GamsJob
from gams import GamsOptions
from gams import GamsWorkspace
from gams.control.workspace import GamsExceptionExecution
from gams.core import gdx

import gamspy as gp
import gamspy.utils as utils
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


class Container(gt.Container):
    """
    A container is an object that holds all symbols and operates on them.

    Parameters
    ----------
    load_from : str, optional
        Path to the GDX file to be loaded from, by default None
    system_directory : str, optional
        Path to the directory that holds the GAMS installation, by default None
    name : str, optional
        Name of the Container, by default "default"
    working_directory : str, optional
        Path to the working directory to store temporary files such .lst, .gms,
        .gdx, .g00 files.

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
    ):
        system_directory = (
            system_directory
            if system_directory
            else utils._getGAMSPyBaseDirectory()
        )

        self._delayed_execution = delayed_execution
        self._unsaved_statements: dict = {}
        self._use_restart_from = False

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

        # allows interrupt
        self._job: Optional[GamsJob] = None

    def _addGamsCode(self, gams_code: str) -> None:
        """
        Adds an arbitrary GAMS code to the generate .gms file

        Parameters
        ----------
        gams_code : str
        """
        unique_name = utils._getUniqueName()
        self._unsaved_statements[unique_name] = gams_code

    def _addStatement(self, statement) -> None:
        self._unsaved_statements[statement.name] = statement

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
                    symbol.records,
                    symbol.domain_forwarding,
                    symbol.description,
                )
            elif isinstance(symbol, gt.Parameter):
                _ = gp.Parameter(
                    self,
                    symbol.name,
                    new_domain,
                    symbol.records,
                    symbol.domain_forwarding,
                    symbol.description,
                )
            elif isinstance(symbol, gt.Variable):
                _ = gp.Variable(
                    self,
                    symbol.name,
                    symbol.type,
                    new_domain,
                    symbol.records,
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
                    records=symbol.records,
                    domain_forwarding=symbol.domain_forwarding,
                    description=symbol.description,
                )

    def _get_symbol_names_from_gdx(self, load_from: str) -> Tuple[list, list]:
        gdx_handle = utils._openGdxFile(self.system_directory, load_from)
        _, symbol_count, _ = gdx.gdxSystemInfo(gdx_handle)

        existing_names = []
        new_names = []
        for i in range(1, symbol_count + 1):
            _, symbol_name, _, _ = gdx.gdxSymbolInfo(gdx_handle, i)
            if symbol_name in self.data.keys():
                existing_names.append(symbol_name)
            else:
                new_names.append(symbol_name)

        utils._closeGdxHandle(gdx_handle)

        return existing_names, new_names

    def _get_symbol_names_to_load(
        self,
        load_from: str,
        symbol_names: Optional[List[str]] = None,
    ) -> List[str]:
        if not symbol_names:
            existing_names, new_names = self._get_symbol_names_from_gdx(
                load_from
            )

            symbol_types = (gp.Set, gp.Parameter, gp.Variable, gp.Equation)

            symbol_names = new_names
            for name in existing_names:
                if isinstance(self[name], symbol_types):
                    symbol_names.append(name)

        return symbol_names

    def _interrupt(self) -> None:
        """
        Sends interrupt signal to the running job.

        Raises
        ------
        GamspyException
            If the job is not initialized
        """
        if self._job:
            self._job.interrupt()
        else:
            raise GamspyException("There is no job initialized.")

    def _setup_paths(
        self,
    ) -> Tuple[GamsCheckpoint, GamsCheckpoint, str, str]:
        """
        Sets up the paths for .g00, and .gdx files.

        Parameters
        ----------
        working_directory : str, optional

        Returns
        -------
        Tuple[GamsCheckpoint, GamsCheckpoint, str, str]
            save_to, restart_from, gdx_in, gdx_out
        """
        save_to = GamsCheckpoint(self.workspace)
        restart_from = GamsCheckpoint(self.workspace)
        gdx_in = self.working_directory + os.sep + f"{save_to._name}.gdx"
        gdx_out = self.working_directory + os.sep + f"{restart_from._name}.gdx"

        return save_to, restart_from, gdx_in, gdx_out

    def _run(
        self,
        options: Optional["GamsOptions"] = None,
        output: Optional[io.TextIOWrapper] = None,
        backend: Literal["local", "engine"] = "local",
        engine_config: Optional["EngineConfig"] = None,
    ):
        # Set default options if not provided
        if options is None:
            options = GamsOptions(self.workspace)
            options.gdx = self._gdx_out

        # Reset dirty flags for symbols
        for name in self.data.keys():
            if hasattr(self[name], "_is_dirty") and self[name]._is_dirty:
                self[name]._is_dirty = False

        # Create gdx file to read records from
        super().write(self._gdx_in)

        # Generate GAMS code as a string
        if backend == "engine":
            # Engine expects gdx file to be next to the gms file
            gams_string = self.generateGamsString(
                self._gdx_in.split(os.sep)[-1]
            )
        else:
            gams_string = self.generateGamsString()

        # If there is no restart checkpoint, set it to None
        checkpoint = self._restart_from if self._use_restart_from else None

        self._job = GamsJob(
            self.workspace,
            source=gams_string,
            checkpoint=checkpoint,
        )

        # Run the job based on the selected backend
        if backend == "local":
            try:
                self._job.run(
                    gams_options=options,
                    checkpoint=self._save_to,
                    create_out_db=False,
                    output=output,
                )
            except GamsExceptionExecution as e:
                self._unsaved_statements = {}
                message = parse_message(self.workspace, options, self._job)
                e.value = message + e.value if message else e.value
                raise e
        elif backend == "engine":
            options.gdx = self._gdx_out.split(os.sep)[-1]
            if engine_config is None:
                raise GamspyException(
                    "Engine configuration must be defined to run the job with"
                    " GAMS Engine"
                )

            extra_model_files = preprocess_extra_model_files(
                self.workspace, self._gdx_in.split(os.sep)[-1], engine_config
            )

            try:
                self._job.run_engine(
                    engine_configuration=engine_config.get_engine_config(),
                    extra_model_files=extra_model_files,
                    gams_options=options,
                    checkpoint=self._save_to,
                    output=output,
                    create_out_db=False,
                    engine_options=engine_config.engine_options,
                    remove_results=engine_config.remove_results,
                )
            except Exception as e:
                self._unsaved_statements = {}
                raise e
        else:
            self._unsaved_statements = {}
            raise GamspyException(
                f"`{backend}` is not a valid backend. Possible backends:"
                " local, engine"
            )

        self._use_restart_from = True

        self._restart_from, self._save_to = self._save_to, self._restart_from
        self._gdx_in, self._gdx_out = self._gdx_out, self._gdx_in

        self.loadRecordsFromGdx(self._gdx_in)
        self._unsaved_statements = {}

    @property
    def delayed_execution(self):
        return self._delayed_execution

    def gamsJobName(self) -> Union[str, None]:
        """
        Returns the name of the latest GAMS job that was executed

        Returns
        -------
        str | None
        """
        return self._job.name if self._job is not None else None

    def addAlias(
        self, name: str, alias_with: Union["Set", "Alias"]
    ) -> "Alias":
        """
        Creates a new Alias and adds it to the container

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
        if name not in self:
            obj = gp.Alias(self, name, alias_with)

            return obj

        else:
            if not isinstance(alias_with, (gp.Set, gp.Alias)):
                raise TypeError(
                    "Symbol 'alias_with' must be type Set or Alias"
                )

            if isinstance(alias_with, gp.Alias):
                parent = alias_with
                while not isinstance(parent, gp.Set):
                    parent = parent.alias_with
                alias_with = parent

            # allow overwriting
            if isinstance(self.data[name], gp.Alias):
                self.data[name].alias_with = alias_with

                return self.data[name]

            else:
                raise ValueError(
                    f"Attempting to add an Alias symbol named `{name}`,"
                    " however a symbol with this name but different type"
                    " already exists in the Container. Symbol replacement is"
                    " only possible if this symbol is first removed from the"
                    " Container with the removeSymbols() method. "
                )

    def addSet(
        self,
        name: str,
        domain: Optional[List[Union["Set", str]]] = None,
        is_singleton: bool = False,
        records: Optional[Any] = None,
        domain_forwarding: bool = False,
        description: str = "",
        uels_on_axes: bool = False,
    ) -> "Set":
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
        if name not in self:
            obj = gp.Set(
                self,
                name,
                domain,
                is_singleton,
                records,
                domain_forwarding,
                description,
                uels_on_axes,
            )

            return obj

        else:
            # try if argument formats are valid
            m = Container(system_directory=self.system_directory)
            obj = gp.Set(
                m,
                name,
                domain,
                is_singleton,
                records=None,
                domain_forwarding=domain_forwarding,
                description=description,
            )

            # domain handling
            if domain is None:
                domain = ["*"]

            if isinstance(domain, (gp.Set, str)):
                domain = [domain]

            # allow records overwriting
            if (
                isinstance(self[name], gp.Set)
                and utils.checkAllSame(self.data[name].domain, domain)
                and self[name].is_singleton == is_singleton
                and self[name].domain_forwarding == domain_forwarding
            ):
                if records is not None:
                    self[name].setRecords(records)

                # only change the description if a new one is passed
                if description != "":
                    self[name].description = description

                return self[name]

            else:
                raise ValueError(
                    f"Attempting to add a symbol named `{name}` but one"
                    " already exists in the Container. Symbol replacement is"
                    " only possible if the symbol is first removed from the"
                    " Container with the removeSymbols() method. Overwriting"
                    " symbol 'records' and 'description' are possible if all"
                    " other properties have not changed."
                )

    def addParameter(
        self,
        name: str,
        domain: Optional[List[Union[str, "Set"]]] = None,
        records: Optional[Any] = None,
        domain_forwarding: bool = False,
        description: str = "",
        uels_on_axes: bool = False,
    ) -> "Parameter":
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
        if name not in self:
            obj = gp.Parameter(
                self,
                name,
                domain,
                records,
                domain_forwarding,
                description,
                uels_on_axes,
            )
            return obj

        else:
            # try if argument formats are valid
            m = Container(system_directory=self.system_directory)
            obj = gp.Parameter(
                m,
                name,
                domain,
                records=None,
                domain_forwarding=domain_forwarding,
                description=description,
            )

            # domain handling
            if domain is None:
                domain = []

            if isinstance(domain, (gt._abcs.AnyContainerDomainSymbol, str)):
                domain = [domain]

            # allow records overwriting
            if (
                isinstance(self.data[name], gp.Parameter)
                and utils.checkAllSame(self.data[name].domain, domain)
                and self.data[name].domain_forwarding == domain_forwarding
            ):
                if records is not None:
                    self.data[name].setRecords(records)

                # only change the description if a new one is passed
                if description != "":
                    self.data[name].description = description

                return self.data[name]

            else:
                raise ValueError(
                    f"Attempting to add a symbol named `{name}` but one"
                    " already exists in the Container. Symbol replacement is"
                    " only possible if the symbol is first removed from the"
                    " Container with the removeSymbols() method. Overwriting"
                    " symbol 'records' and 'description' are possible if all"
                    " other properties have not changed."
                )

    def addVariable(
        self,
        name: str,
        type: str = "free",
        domain: Optional[List[Union[str, "Set"]]] = None,
        records: Optional[Any] = None,
        domain_forwarding: bool = False,
        description: str = "",
        uels_on_axes: bool = False,
    ) -> "Variable":
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
        if name not in self:
            obj = gp.Variable(
                self,
                name,
                type,
                domain,
                records,
                domain_forwarding,
                description,
                uels_on_axes,
            )
            return obj

        else:
            # try if argument formats are valid
            m = Container(system_directory=self.system_directory)
            obj = gp.Variable(
                m,
                name,
                type,
                domain,
                records=None,
                domain_forwarding=domain_forwarding,
                description=description,
            )

            # domain handling
            if domain is None:
                domain = []

            if isinstance(domain, (gt._abcs.AnyContainerDomainSymbol, str)):
                domain = [domain]

            # allow records overwriting
            if (
                isinstance(self.data[name], gp.Variable)
                and self.data[name].type == type
                and utils.checkAllSame(self.data[name].domain, domain)
                and self.data[name].domain_forwarding == domain_forwarding
            ):
                if records is not None:
                    self.data[name].setRecords(records)

                # only change the description if a new one is passed
                if description != "":
                    self.data[name].description = description

                return self.data[name]

            else:
                raise ValueError(
                    f"Attempting to add a symbol named `{name}` but one"
                    " already exists in the Container. Symbol replacement is"
                    " only possible if the symbol is first removed from the"
                    " Container with the removeSymbols() method. Overwriting"
                    " symbol 'records' and 'description' are possible if all"
                    " other properties have not changed."
                )

    def addEquation(
        self,
        name: str,
        type: Union[str, "EquationType"] = "regular",
        domain: Optional[List[Union["Set", str]]] = None,
        definition: Optional["Expression"] = None,
        records: Optional[Any] = None,
        domain_forwarding: bool = False,
        description: str = "",
        uels_on_axes: bool = False,
        definition_domain: Optional[List[Union["Set", str]]] = None,
    ) -> "Equation":
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
        if name not in self:
            obj = gp.Equation(
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
            return obj

        else:
            # try if argument formats are valid
            m = Container(system_directory=self.system_directory)
            obj = gp.Equation(
                m,
                name,
                type,
                domain,
                records=None,
                domain_forwarding=domain_forwarding,
                description=description,
            )

            # domain handling
            if domain is None:
                domain = []

            if isinstance(domain, (gt._abcs.AnyContainerDomainSymbol, str)):
                domain = [domain]

            # allow records overwriting
            if (
                isinstance(self.data[name], gp.Equation)
                and utils.checkAllSame(self.data[name].domain, domain)
                and self.data[name].domain_forwarding == domain_forwarding
            ):
                if records is not None:
                    self.data[name].setRecords(records)

                # only change the description if a new one is passed
                if description != "":
                    self.data[name].description = description

                return self.data[name]

            else:
                raise ValueError(
                    f"Attempting to add a symbol named `{name}` but one"
                    " already exists in the Container. Symbol replacement is"
                    " only possible if the symbol is first removed from the"
                    " Container with the removeSymbols() method. Overwriting"
                    " symbol 'records' and 'description' are possible if all"
                    " other properties have not changed."
                )

    def addModel(
        self,
        name: str,
        problem: str,
        equations: List["Equation"],
        sense: Optional[Literal["MIN", "MAX"]] = None,
        objective: Optional[Union["Variable", "Expression"]] = None,
        matches: Optional[dict] = None,
        limited_variables: Optional[list] = None,
    ) -> "Model":
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
        model = gp.Model(
            self,
            name,
            problem,
            equations,
            sense,
            objective,
            matches,
            limited_variables,
        )
        return model

    def copy(self, working_directory: Optional[str] = None) -> "Container":
        """
        Creates a copy of the Container

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
        >>> new_cont = m.copy()
        >>> new_cont.data.keys() == m.data.keys()
        True

        """
        self._run()

        m = Container(working_directory=working_directory)
        m.read(self._gdx_in)

        try:
            shutil.copy(
                self._save_to._checkpoint_file_name,
                m._save_to._checkpoint_file_name,
            )
        except FileNotFoundError:
            # save_to might not exist and it's fine
            pass
        except shutil.SameFileError:
            # They can be the same file if their working directories are the same.
            pass
        except Exception as e:
            raise GamspyException(f"Copy failed because {str(e)}")

        try:
            shutil.copy(
                self._restart_from._checkpoint_file_name,
                m._restart_from._checkpoint_file_name,
            )
        except shutil.SameFileError:
            # They can be the same files if their working directories are the same.
            pass
        except Exception as e:
            raise GamspyException(f"Copy failed because {str(e)}")

        try:
            shutil.copy(self._gdx_in, m._gdx_in)
            shutil.copy(self._gdx_out, m._gdx_out)
        except shutil.SameFileError:
            # They can be the same files if their working directories are the same.
            pass
        except Exception as e:
            raise GamspyException(f"Copy failed because {str(e)}")

        # if already defined equations exist, add them to .gms file
        for equation in self.getEquations():
            if equation._definition is not None:
                m._addStatement(equation._definition)

        return m

    def getSets(self) -> List["Set"]:
        """
        Returns all Sets in the container

        Returns
        -------
        List[Set]
        """
        return [self[symbol_name] for symbol_name in self.listSets()]

    def getAliases(self) -> List["Alias"]:
        """
        Returns all Aliases in the container

        Returns
        -------
        List[Alias]
        """
        return [self[symbol_name] for symbol_name in self.listAliases()]

    def getParameters(self) -> List["Parameter"]:
        """
        Returns all parameters in the container

        Returns
        -------
        List[Parameter]
        """
        return [self[symbol_name] for symbol_name in self.listParameters()]

    def getVariables(self) -> List["Variable"]:
        """
        Returns all variables in the container

        Returns
        -------
        List[Variable]
        """
        return [self[symbol_name] for symbol_name in self.listVariables()]

    def getEquations(self) -> List["Equation"]:
        """
        Returns all equations in the container

        Returns
        -------
        List[Equation]
        """
        return [self[symbol_name] for symbol_name in self.listEquations()]

    def generateGamsString(self, gdx_path: Optional[str] = None) -> str:
        """
        Generates the GAMS code

        Returns
        -------
        str
        """
        symbol_types = (gp.Set, gp.Parameter, gp.Variable, gp.Equation)
        possible_undef_types = (gp.Parameter, gp.Variable, gp.Equation)
        gdx_path = gdx_path if gdx_path else self._gdx_in

        string = f"$onMultiR\n$gdxIn {gdx_path}\n"
        for statement in self._unsaved_statements.values():
            if isinstance(statement, str):
                string += statement + "\n"
            else:
                statement_str = statement.getStatement() + "\n"

                if isinstance(statement, symbol_types):
                    statement_str += f"$load {statement.name}\n"

                if isinstance(statement, possible_undef_types):
                    # save old dirty state to avoid self.records recursion
                    old_dirty_state = statement._is_dirty
                    statement._is_dirty = False

                    num_undef = statement.countUndef()

                    # restore old dirty state
                    statement._is_dirty = old_dirty_state

                    if num_undef:
                        statement_str = f"$onUNDF\n{statement_str}$offUNDF"

                string += statement_str + "\n"

        for symbol_name, symbol in self:
            if isinstance(symbol, gp.Alias):
                string += f"$load {symbol.alias_with.name}\n"
            else:
                string += f"$load {symbol_name}\n"

        string += "$gdxIn\n"

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
        """
        symbol_names = self._get_symbol_names_to_load(load_from, symbol_names)

        temp_container = Container(system_directory=self.system_directory)
        temp_container.read(load_from, symbol_names)

        for name in symbol_names:
            if name in self.data.keys():
                statement = self[name]
                updated_records = temp_container[name].records

                statement.records = updated_records

                if updated_records is not None:
                    statement.domain_labels = statement.domain_names
            else:
                self.read(load_from, [name])

    def read(
        self, load_from: str, symbol_names: Optional[List[str]] = None
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
        sequence = symbols if symbols else self.data.keys()
        dirty_symbols = []
        for name in sequence:
            if hasattr(self[name], "_is_dirty") and self[name]._is_dirty:
                dirty_symbols.append(name)

        if len(dirty_symbols) > 0:
            self._run()

        super().write(write_to, symbols)


def parse_message(
    workspace: "GamsWorkspace", options: "GamsOptions", job: "GamsJob"
) -> str:
    error_message = ""
    header = "=" * 80
    footer = "=" * 80
    message_format = "\n\n{header}\nError Summary\n{footer}\n{message}\n"

    lst_filename = options.output if options.output else job._job_name + ".lst"

    lst_path = workspace._working_directory + os.path.sep + lst_filename

    with open(lst_path) as lst_file:
        all_lines = lst_file.readlines()
        num_lines = len(all_lines)

        index = 0
        while index < num_lines:
            line = all_lines[index]

            if line.startswith("****"):
                error_lines = [all_lines[index - 1]]
                temp_index = index

                while (
                    all_lines[temp_index].startswith("****")
                    and temp_index < len(all_lines) - 1
                ):
                    error_lines.append(all_lines[temp_index])
                    temp_index += 1

                error_message = message_format.format(
                    message="".join(error_lines),
                    header=header,
                    footer=footer,
                )
                break

            index += 1

    return error_message


def preprocess_extra_model_files(
    workspace: "GamsWorkspace", gdx_path: "str", engine_config: "EngineConfig"
) -> List[str]:
    """
    Conforms model files to the path requirements of GAMS Engine

    Parameters
    ----------
    engine_config : EngineConfig

    Returns
    -------
    List[str]
    """
    # copy provided extra model files to working directory
    for extra_file in engine_config.extra_model_files:
        shutil.copy(extra_file, workspace.working_directory)

    # trim path and keep only the names of the files
    extra_model_files = [
        os.path.abspath(extra_file).split(os.sep)[-1]
        for extra_file in engine_config.extra_model_files
    ]

    # add name of the gdx file
    extra_model_files.append(os.path.abspath(gdx_path).split(os.sep)[-1])

    return extra_model_files
