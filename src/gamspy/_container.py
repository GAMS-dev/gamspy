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

import os
import io
import pandas as pd
from gams import GamsWorkspace, GamsJob, GamsCheckpoint, DebugLevel
import gams.transfer as gt
import gamspy as gp
import gamspy.utils as utils
import gamspy._algebra._expression as expression
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Union,
    Optional,
    Tuple,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from gamspy import Alias, Set, Parameter, Variable, Equation
    from gamspy._algebra._expression import Expression
    from gams import GamsOptions


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
    >>> m = gp.Container()
    >>> m = gp.Container(system_directory=path_to_the_directory)
    >>> m = gp.Container(load_from=path_to_the_gdx)
    """

    def __init__(
        self,
        load_from: Optional[str] = None,
        system_directory: Optional[str] = None,
        name: str = "default",
        working_directory: Optional[str] = None,
    ):
        self.system_directory = utils._getSystemDirectory(system_directory)
        self.workspace = GamsWorkspace(
            working_directory, self.system_directory, DebugLevel.KeepFiles
        )

        self.name = name
        self._statements_dict: dict = {}
        self._unsaved_statements: dict = {}

        super().__init__(load_from, self.system_directory)

        (
            self._gms_path,
            self._lst_path,
            self._save_to,
            self._restart_from,
            self._gdx_path,
        ) = self._setup_paths()

        self._clean_existing_workfiles()
        self._gams_compiler_path = self.system_directory + os.sep + "gams"

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

    def _setup_paths(
        self,
    ) -> Tuple[str, str, GamsCheckpoint, GamsCheckpoint, str]:
        """
        Sets up the paths for .gms, .lst, .g00, and .gdx files.

        Parameters
        ----------
        working_directory : str, optional

        Returns
        -------
        Tuple[str, str, GamsCheckpoint, GamsCheckpoint, str]
            gms_path, lst_path, save_to, restart_from, gdx_path
        """
        directory = self.workspace.working_directory

        if " " in directory:
            raise Exception(
                "Working directory path cannot contain spaces. Working"
                f" directory: {directory}"
            )

        temporary_file_prefix = os.path.join(directory, self.name)
        gms_path = temporary_file_prefix + ".gms"
        lst_path = temporary_file_prefix + ".lst"
        save_to = GamsCheckpoint(
            self.workspace, temporary_file_prefix + "_save.g00"
        )
        restart_from = GamsCheckpoint(
            self.workspace, temporary_file_prefix + "_restart.g00"
        )
        gdx_path = temporary_file_prefix + ".gdx"

        return gms_path, lst_path, save_to, restart_from, gdx_path

    def _clean_existing_workfiles(self) -> None:  # pragma: no cover
        """Deletes local workfiles"""
        if os.path.exists(self._restart_from._checkpoint_file_name):
            os.remove(self._restart_from._checkpoint_file_name)
        if os.path.exists(self._save_to._checkpoint_file_name):
            os.remove(self._save_to._checkpoint_file_name)

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
            m = Container()
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

    def addModel(
        self,
        name: str,
        equations: List["Equation"],
        problem: str,
        sense: Optional[Literal["MIN", "MAX"]] = None,
        objective_variable: Optional["Variable"] = None,
        limited_variables: Optional[list] = None,
    ):
        model = gp.Model(
            self,
            name,
            equations,
            problem,
            sense,
            objective_variable,
            limited_variables,
        )
        return model

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
        dirty_symbols = []
        for symbol in self.data.values():
            if hasattr(symbol, "_is_dirty") and symbol._is_dirty:
                dirty_symbols.append(symbol)

        self._run_job()

        self._unsaved_statements = {}

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

    def _run_job(
        self,
        options: Optional["GamsOptions"] = None,
        output: Optional[io.TextIOWrapper] = None,
    ):
        self.write(self._gdx_path)
        gams_string = self.generateGamsString(self._unsaved_statements)

        checkpoint = (
            self._restart_from
            if os.path.exists(self._restart_from._checkpoint_file_name)
            else None
        )

        job = GamsJob(
            self.workspace,
            source=gams_string,
            checkpoint=checkpoint,
        )

        job.run(
            gams_options=options,
            checkpoint=self._save_to,
            create_out_db=True,
            output=output,
        )

        self._restart_from, self._save_to = self._save_to, self._restart_from

        self._gdx_path = (
            job.out_db.workspace.working_directory
            + os.sep
            + job.out_db.name
            + ".gdx"
        )

        self.loadRecordsFromGdx(self._gdx_path)

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
                    statement.domain_labels = statement.domain_names

        utils._closeGdxHandle(gdxHandle)

        self._unsaved_statements = {}

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
        """
        super().read(load_from, symbol_names)
        self._cast_symbols(symbol_names)

    def write(
        self,
        write_to: str,
        symbol_names: Optional[List[str]] = None,
    ) -> None:
        """
        Writes specified symbols to the gdx file. If symbol_names are
        not provided, it writes all symbols to the gdx file.

        Parameters
        ----------
        write_to : str
        symbol_names : List[str], optional
        """
        sequence = symbol_names if symbol_names else self.data.keys()
        for symbol_name in sequence:
            self[symbol_name]._is_dirty = False

        super().write(write_to, symbol_names)
