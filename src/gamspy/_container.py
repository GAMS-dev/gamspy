import subprocess
import os
import sys
import pandas as pd
import gams.transfer as gt
import gamspy as gp
import gamspy.utils as utils
import gamspy._algebra._expression as expression
from typing import Any, Dict, List, Union, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from gamspy import Alias, Set, Parameter, Variable, Equation, Model
    from gamspy._algebra._expression import Expression


class Container(gt.Container):
    """
    A container is an object that holds all symbols and operates on them.

    Parameters
    ----------
    load_from : Optional[str], optional
        Path to the GDX file to be loaded from, by default None
    system_directory : Optional[str], optional
        Path to the directory that holds the GAMS installation, by default None
    name : str, optional
        Name of the Container, by default "default"
    """

    def __init__(
        self,
        load_from: Optional[str] = None,
        system_directory: Optional[str] = None,
        name: str = "default",
    ):
        self.system_directory = self.get_system_directory(system_directory)
        super().__init__(load_from, self.system_directory)

        self.name = name
        self._statements_dict: dict = {}
        self._unsaved_statements: dict = {}

        # read on demand
        (
            self._gms_path,
            self._lst_path,
            self._save_to,
            self._restart_from,
            self._gdx_path,
        ) = self._setup_paths()
        self._clean_existing_workfiles()

        self._cast_symbols()

    def get_system_directory(self, system_directory: Optional[str]) -> str:
        """
        Finds the system directory. If no existing GAMS installation provided,
        returns minigams directory.

        Parameters
        ----------
        system_directory : Optional[str]

        Returns
        -------
        str
            System directory
        """
        if system_directory:
            return system_directory

        system_directory = os.path.dirname(os.path.realpath(__file__)) + os.sep

        user_os = platform.system().lower()
        system_directory += "minigams" + os.sep + user_os

        if user_os == "darwin":
            system_directory += f"_{platform.machine()}"

        return system_directory

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
                _ = gp.Alias(
                    self,
                    symbol.name,
                    symbol.alias_with,
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
                _ = gp.Equation(
                    self,
                    symbol.name,
                    symbol.type,
                    new_domain,
                    symbol.records,
                    symbol.domain_forwarding,
                    symbol.description,
                )

    def _setup_paths(self) -> Tuple[str, str, str, str, str]:
        """
        Sets up the paths for .gms, .lst, .g00, and .gdx files.

        Returns
        -------
        Tuple[str, str, str, str, str]
            gms_path, save_to, restart_from, gdx_path
        """
        gms_path = os.path.join(os.getcwd(), f"{self.name}.gms")
        if " " in gms_path:
            gms_path = f'"{gms_path}"'

        lst_path = gms_path[:-4] + ".lst"

        save_to = os.path.join(os.getcwd(), f"{self.name}_save.g00")
        if " " in save_to:
            save_to = f'"{save_to}"'

        restart_from = os.path.join(os.getcwd(), f"{self.name}_restart.g00")
        if " " in restart_from:
            restart_from = f'"{restart_from}"'

        gdx_path = os.path.join(os.getcwd(), f"{self.name}.gdx")
        if " " in gdx_path:
            gdx_path = f'"{gdx_path}"'

        return gms_path, lst_path, save_to, restart_from, gdx_path

    def _clean_existing_workfiles(self) -> None:
        """Deletes local workfiles"""
        if os.path.exists(self._restart_from):
            os.remove(self._restart_from)
        if os.path.exists(self._save_to):
            os.remove(self._save_to)

    def _addStatement(self, statement) -> None:
        self._statements_dict[statement.name] = statement
        self._unsaved_statements[statement.name] = statement

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
        domain : List[Union[Set, str]], optional
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
            If there is symbol with same name but different type in the
            Container
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
            try:
                m = Container()
                obj = gp.Set(
                    m,
                    name,
                    domain,
                    is_singleton,
                    records=None,
                    domain_forwarding=domain_forwarding,
                    description=description,
                )
            except (TypeError, ValueError, Exception) as err:
                raise err

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
            try:
                m = Container()
                obj = gp.Parameter(
                    m,
                    name,
                    domain,
                    records=None,
                    domain_forwarding=domain_forwarding,
                    description=description,
                )
            except (TypeError, ValueError, Exception) as err:
                raise err

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
            try:
                m = Container()
                obj = gp.Variable(
                    m,
                    name,
                    type,
                    domain,
                    records=None,
                    domain_forwarding=domain_forwarding,
                    description=description,
                )
            except (TypeError, ValueError, Exception) as err:
                raise err

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
        type: str,
        domain: Optional[List[Union["Set", str]]] = None,
        records: Optional[Any] = None,
        domain_forwarding: bool = False,
        description: str = "",
        uels_on_axes: bool = False,
        definition: Optional["Expression"] = None,
        definition_domain: Optional[List[Union["Set", str]]] = None,
    ) -> "Equation":
        """
        Creates an Equation and adds it to the Container

        Parameters
        ----------
        name : str
        type : str
        domain : List[Set | str], optional
        records : Any, optional
        domain_forwarding : bool, optional
        description : str, optional
        uels_on_axes : bool, optional
        definition : Expression, optional
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
        """
        if name not in self:
            obj = gp.Equation(
                self,
                name,
                type,
                domain,
                records,
                domain_forwarding,
                description,
                uels_on_axes,
                definition,
                definition_domain,
            )
            return obj

        else:
            # try if argument formats are valid
            try:
                m = Container()
                obj = gp.Equation(
                    m,
                    name,
                    type,
                    domain,
                    records=None,
                    domain_forwarding=domain_forwarding,
                    description=description,
                )
            except (TypeError, ValueError, Exception) as err:
                raise err

            # domain handling
            if domain is None:
                domain = []

            if isinstance(domain, (gt._abcs.AnyContainerDomainSymbol, str)):
                domain = [domain]

            # allow records overwriting
            if (
                isinstance(self.data[name], gp.Equation)
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

    def addOptions(self, options: Dict[str, str]) -> None:
        """
        Allows adding options to .gms file

        Parameters
        ----------
        options : Dict[str, str]

        Raises
        ------
        ValueError
            In case the option is not valid
        """
        for key, value in options.items():
            if not key.lower() in utils.GMS_OPTIONS:
                raise ValueError(
                    f"{key} is not a valid option. Valid options:"
                    f" {utils.GMS_OPTIONS}"
                )

            self._addStatement(
                expression.Expression(f"option {key}", "=", value)
            )

    def addGamsCode(self, gams_code: str) -> None:
        """
        Adds an arbitrary GAMS code to the generate .gms file

        Parameters
        ----------
        gams_code : str
        """
        unique_name = utils._getUniqueName()
        self._unsaved_statements[unique_name] = gams_code
        self._statements_dict[unique_name] = gams_code

    def _loadOnDemand(self) -> pd.DataFrame:
        """Loads data of the given symbol from the gdx file."""
        # Save unsaved statements to a file
        self._write_to_gms()

        # Restart from a workfile
        self._restart_from_workfile()

        # Update symbol data
        dirty_symbols = []
        for symbol in self.data.values():
            if hasattr(symbol, "_is_dirty") and symbol._is_dirty:
                dirty_symbols.append(symbol)
                symbol._is_dirty = False

        self.loadRecordsFromGdx(self._gdx_path, dirty_symbols)

        # Empty unsaved statements
        self._unsaved_statements = {}

    def _restart_from_workfile(self) -> None:
        """Restarts from the latest workfile"""
        commands = [
            self._gams_compiler_path,
            self._gms_path,
            f"save={self._save_to}",
            f"gdx={self._gdx_path}",
        ]

        if os.path.exists(self._restart_from):
            commands.append(f"restart={self._restart_from}")

        try:
            _ = subprocess.run(
                " ".join(commands),
                capture_output=True,
                check=True,
                shell=True,
                text=True,
            )
        except Exception as e:
            executed_command = " ".join(commands)
            sys.exit(
                "Could not restart with the following"  # type: ignore
                f" command:\n\n{executed_command}\n\nError log:\n\n{e.output}"
            )

        # https://www.gams.com/latest/docs/UG_SaveRestart.html#UG_SaveRestart_AvoidingCommonMistakes
        self._save_to, self._restart_from = self._restart_from, self._save_to

    def solve(
        self,
        model: "Model",
        problem: str,
        sense: Optional[str] = None,
        objective_variable: Optional["Variable"] = None,
        commandline_options: Optional[dict] = None,
        scenario: Optional["Set"] = None,
        stdout: Optional[str] = None,
    ) -> str:
        """
        Generates the gams string, writes it to a file and runs it

        Parameters
        ----------
        model : Model
        problem : str
        sense : "MIN" or "MAX", optional
        objective_variable : Variable, optional
        commandline_options : dict, optional
        scenario : Set, optional
        stdout : str, optional

        Returns
        -------
        str
            GAMS output

        Raises
        ------
        ValueError
            In case problem is not in possible problem types
        ValueError
            In case sense is different than "MIN" or "MAX"
        TypeError
            In case scenario is not a Set
        TypeError
            In case stdout is not a string
        """
        if problem.upper() not in utils.PROBLEM_TYPES:
            raise ValueError(
                f"Allowed problem types: {utils.PROBLEM_TYPES} but found"
                f" {problem}."
            )

        if sense is not None and sense.upper() not in utils.SENSE_TYPES:
            raise ValueError(
                f"Allowed sense types: {utils.SENSE_TYPES} but found {sense}."
            )

        if scenario is not None and not isinstance(scenario, gp.Set):
            raise TypeError(
                f"scenario must be a Set but found {type(scenario)}"
            )

        if stdout is not None and not isinstance(stdout, str):
            raise TypeError("stdout must be a path for the output file")

        solve_string = f"solve {model.name} using {problem}"

        if sense:
            solve_string += f" {sense}"

        if objective_variable:
            solve_string += f" {objective_variable.gamsRepr()}"

        if scenario:
            solve_string += f" scenario {scenario.gamsRepr()}"

        self._unsaved_statements[utils._getUniqueName()] = solve_string + ";\n"

        self._write_to_gms()
        output = self._run_gms(commandline_options)

        # Write results to the specified output file
        if stdout:
            with open(stdout, "w") as output_file:
                output_file.write(output)

        self._update_status(model)

        self.loadRecordsFromGdx(self._gdx_path)

        return output

    def _update_status(self, model):
        with open(self._lst_path) as listing_file:
            lines = listing_file.read()

            lines = lines.split("\n")

            for line in lines:
                # Set model status
                if line.startswith("**** MODEL STATUS"):
                    status_number = int(line[5:].strip().split()[2])
                    status = gp.ModelStatus(status_number)
                    model.status = status

    def generateGamsString(self, dictionary: Optional[Dict] = None) -> str:
        """
        Generates the GAMS code

        Parameters
        ----------
        dictionary : Dict, optional
            Dictionary that contains the expressions, by default None

        Returns
        -------
        str
        """
        dictionary = (
            self._statements_dict if dictionary is None else dictionary
        )
        return (
            "\n".join(
                [
                    (
                        statement
                        if isinstance(statement, str)
                        else statement.getStatement()
                    )
                    for statement in dictionary.values()
                ]
            )
            + "\n"
        )

    def _write_to_gms(self):
        gams_string = self.generateGamsString(self._unsaved_statements)
        try:
            with open(self._gms_path, "w") as file:
                file.write(gams_string)
        except Exception as e:
            sys.exit(f"Could not write to {self.name}.gms because: {e}")

    def _run_gms(self, commandline_options: Optional[dict] = None):
        commands = [
            self._gams_compiler_path,
            self._gms_path,
            f"save={self._save_to}",
            f"gdx={self._gdx_path}",
        ]

        if os.path.exists(self._restart_from):
            commands.append(f"restart={self._restart_from}")

        if commandline_options:
            for key, value in commandline_options.items():
                commands.append(f"{key}={value}")

        try:
            process = subprocess.run(
                " ".join(commands),
                capture_output=True,
                check=True,
                shell=True,
                text=True,
            )

            # https://www.gams.com/latest/docs/UG_SaveRestart.html#UG_SaveRestart_AvoidingCommonMistakes
            self._save_to, self._restart_from = (
                self._restart_from,
                self._save_to,
            )

            return process.stdout
        except Exception as e:
            executed_command = " ".join(commands)
            sys.exit(
                "Could not run .gms file with the following GAMS"  # type: ignore # noqa: E501
                f" command:\n\n{executed_command}\n\nError log: \n\n{e.output}"
            )

    def loadRecordsFromGdx(
        self,
        load_from: str,
        symbols: Optional[
            List[Union["Set", "Parameter", "Variable", "Equation"]]
        ] = None,
    ) -> None:
        """
        Loads data of the given symbols from a gdx file. If no symbols
        are given, data of all symbols are loaded.

        Parameters
        ----------
        load_from : str
            Path to the gdx file
        symbols : List[Set | Parameter | Variable | Equation], optional
            Symbols whose data will be load from gdx, by default None
        """

        symbol_types = (gp.Set, gp.Parameter, gp.Variable, gp.Equation)

        gdxHandle = utils._openGdxFile(self.system_directory, load_from)

        iterable = symbols if symbols else self._statements_dict.values()

        for statement in iterable:
            if isinstance(statement, symbol_types):
                updated_records = utils._getSymbolData(
                    self._gams2np, gdxHandle, statement.name
                )

                statement.records = updated_records

                if updated_records is not None:
                    statement.records = updated_records.copy()
                    statement.domain_labels = statement.domain_names

        utils._closeGdxHandle(gdxHandle)

        self._unsaved_statements = {}

    def loadSymbolsFromGdx(
        self, load_from: str, symbol_names: List[str]
    ) -> None:
        """
        Loads specified symbols from the gdx file

        Parameters
        ----------
        load_from : str
        symbol_names : List[str]
        """
        self.read(load_from, symbol_names, True)
        self._cast_symbols(symbol_names)
