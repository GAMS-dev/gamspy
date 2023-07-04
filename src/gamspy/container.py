import subprocess
import os
import sys
import pandas as pd
import gams.transfer as gt
import gamspy.utils as utils
import gamspy._algebra._expression as expression
from typing import Dict, List, Union, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from gamspy import Set, Parameter, Variable, Equation, Model


class Container(gt.Container):
    def __init__(
        self,
        load_from: Optional[str] = None,
        system_directory: Optional[str] = None,
        name: str = "default",
    ):
        super().__init__(load_from, system_directory)

        self.name = name
        self._statements_dict = {}
        self._unsaved_statements = {}

        # read on demand
        (
            self._gms_path,
            self._save_to,
            self._restart_from,
            self._gdx_path,
        ) = self._setup_paths()
        self._clean_existing_workfiles()

        self._cast_symbols()

    def _cast_symbols(self):
        import gamspy as gp

        for gt_symbol_name in list(self.data.keys()):
            gt_symbol = self.data[gt_symbol_name]

            del self.data[gt_symbol_name]

            if isinstance(gt_symbol, gt.Alias):
                _ = gp.Alias(
                    self,
                    gt_symbol.name,
                    gt_symbol.alias_with,
                )
            elif isinstance(gt_symbol, gt.Set):
                _ = gp.Set(
                    self,
                    gt_symbol.name,
                    gt_symbol.domain,
                    gt_symbol.is_singleton,
                    gt_symbol.records,
                    gt_symbol.domain_forwarding,
                    gt_symbol.description,
                )
            elif isinstance(gt_symbol, gt.Parameter):
                _ = gp.Parameter(
                    self,
                    gt_symbol.name,
                    gt_symbol.domain,
                    gt_symbol.records,
                    gt_symbol.domain_forwarding,
                    gt_symbol.description,
                )
            elif isinstance(gt_symbol, gt.Variable):
                _ = gp.Variable(
                    self,
                    gt_symbol.name,
                    gt_symbol.type,
                    gt_symbol.domain,
                    gt_symbol.records,
                    gt_symbol.domain_forwarding,
                    gt_symbol.description,
                )
            elif isinstance(gt_symbol, gt.Equation):
                _ = gp.Equation(
                    self,
                    gt_symbol.name,
                    gt_symbol.type,
                    gt_symbol.domain,
                    gt_symbol.records,
                    gt_symbol.domain_forwarding,
                    gt_symbol.description,
                )

    def _setup_paths(self) -> Tuple[str, str, str, str]:
        gms_path = os.path.join(os.getcwd(), f"{self.name}.gms")
        if " " in gms_path:
            gms_path = f'"{gms_path}"'

        save_to = os.path.join(os.getcwd(), f"{self.name}_save.g00")
        if " " in save_to:
            save_to = f'"{save_to}"'

        restart_from = os.path.join(os.getcwd(), f"{self.name}_restart.g00")
        if " " in restart_from:
            restart_from = f'"{restart_from}"'

        gdx_path = os.path.join(os.getcwd(), f"{self.name}.gdx")
        if " " in gdx_path:
            gdx_path = f'"{gdx_path}"'

        return gms_path, save_to, restart_from, gdx_path

    def _clean_existing_workfiles(self) -> None:
        """Deletes local workfiles"""
        if os.path.exists(self._restart_from):
            os.remove(self._restart_from)
        if os.path.exists(self._save_to):
            os.remove(self._save_to)

    def _addStatement(self, statement) -> None:
        self._statements_dict[statement.name] = statement
        self._unsaved_statements[statement.name] = statement

    def addAlias(self, name, alias_with):
        import gamspy as gp

        if name not in self:
            obj = gp.Alias(self, name, alias_with)

            return obj

        else:
            if not isinstance(alias_with, (gt.Set, gt.Alias)):
                raise TypeError("Symbol 'alias_with' must be type Set or Alias")

            if isinstance(alias_with, gt.Alias):
                parent = alias_with
                while not isinstance(parent, gt.Set):
                    parent = parent.alias_with
                alias_with = parent

            # allow overwriting
            if isinstance(self.data[name], gt.Alias):
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
        name,
        domain=None,
        is_singleton=False,
        records=None,
        domain_forwarding=False,
        description="",
        uels_on_axes=False,
    ):
        import gamspy as gp

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
                obj = Set(
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

            if isinstance(domain, (gt.Set, str)):
                domain = [domain]

            # allow records overwriting
            if (
                isinstance(self[name], gt.Set)
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
        name,
        domain=None,
        records=None,
        domain_forwarding=False,
        description="",
        uels_on_axes=False,
    ):
        import gamspy as gp

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
                obj = Parameter(
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
                isinstance(self.data[name], gt.Parameter)
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
        name,
        type="free",
        domain=None,
        records=None,
        domain_forwarding=False,
        description="",
        uels_on_axes=False,
    ):
        import gamspy as gp

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
                obj = Variable(
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
                isinstance(self.data[name], gt.Variable)
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
        name,
        type,
        domain=None,
        records=None,
        domain_forwarding=False,
        description="",
        uels_on_axes=False,
        definition=None,
        definition_domain=None,
    ):
        import gamspy as gp

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
                obj = gt.Equation(
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
                isinstance(self.data[name], gt.Equation)
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
        for key, value in options.items():
            if not key.lower() in utils.GMS_OPTIONS:
                raise ValueError(
                    f"{key} is not a valid option. Valid options:"
                    f" {utils.GMS_OPTIONS}"
                )

            self._addStatement(expression.Expression(f"option {key}", "=", value))

    def addGamsCode(self, gams_code: str) -> None:
        """Adds an arbitrary GAMS code to the generate .gms file

        Parameters
        ----------
        gams_code : str
        """
        self._addStatement(expression.Expression(gams_code, "", ""))

    def _loadOnDemand(self, symbol_name: str) -> pd.DataFrame:
        """Loads data of the given symbol from the gdx file."""
        # Save unsaved statements to a file
        self._write_to_gms()

        # Restart from a workfile
        self._restart_from_workfile()

        # Update symbol data
        gdx_handle = utils._openGdxFile(self.system_directory, self._gdx_path)
        data = utils._getSymbolData(self._gams2np, gdx_handle, symbol_name)
        utils._closeGdxHandle(gdx_handle)

        # Empty unsaved statements
        self._unsaved_statements = {}

        return data

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
                "Could not restart with the following"
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
        stdout=None,
    ):
        """Generates the gams string, writes it to a file and runs it"""
        if not problem.upper() in utils.PROBLEM_TYPES:
            raise ValueError(
                f"Allowed problem types: {utils.PROBLEM_TYPES} but found" f" {problem}."
            )

        if sense is not None and not sense.upper() in utils.SENSE_TYPES:
            raise ValueError(
                f"Allowed sense types: {utils.SENSE_TYPES} but found {sense}."
            )

        if stdout is not None and not isinstance(stdout, str):
            raise ValueError("stdout must be a path for the output file")

        sense = "" if sense is None else sense
        objective = objective_variable.name if objective_variable is not None else ""

        self._unsaved_statements[
            utils._getUniqueName()
        ] = f"solve {model.name} {sense} {objective} using {problem};\n"

        self._write_to_gms()
        output = self._run_gms(commandline_options)

        # Write results to the specified output file
        if stdout:
            with open(stdout, "w") as output_file:
                output_file.write(output)

        self.loadFromGdx(self._gdx_path)

        return output

    def generateGamsString(self, dictionary: Optional[Dict] = None) -> str:
        """Generates the GAMS code

        Parameters
        ----------
        dictionary : Dict, optional
            Dictionary that contains the expressions, by default None

        Returns
        -------
        str
        """
        dictionary = self._statements_dict if dictionary is None else dictionary
        return (
            "\n".join(
                [
                    statement
                    if isinstance(statement, str)
                    else statement.getStatement()
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
                "Could not run .gms file with the following GAMS"
                f" command:\n\n{executed_command}\n\nError log: \n\n{e.output}"
            )

    def loadFromGdx(
        self,
        load_from: str,
        symbols: List[Union["Set", "Parameter", "Variable", "Equation"]] = None,
    ) -> None:
        import gamspy as gp

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
