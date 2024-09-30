from __future__ import annotations

import io
import os
from typing import TYPE_CHECKING, Iterable

import gams.transfer as gt
from gams import (
    DebugLevel,
    EquType,
    GamsCheckpoint,
    GamsException,
    GamsModelInstanceOpt,
    GamsModifier,
    GamsOptions,
    GamsWorkspace,
    SymbolUpdateType,
    UpdateAction,
    VarType,
)

import gamspy as gp
import gamspy._symbols.implicits as implicits
import gamspy.utils as utils
from gamspy._options import ModelInstanceOptions, Options
from gamspy.exceptions import GamspyException, ValidationError

if TYPE_CHECKING:
    from gamspy import Container, Model, Parameter
    from gamspy._symbols.implicits import ImplicitParameter


VARIABLE_MAP = {
    "binary": VarType.Binary,
    "integer": VarType.Integer,
    "positive": VarType.Positive,
    "negative": VarType.Negative,
    "free": VarType.Free,
    "sos1": VarType.SOS1,
    "sos2": VarType.SOS2,
    "semicont": VarType.SemiCont,
    "semiint": VarType.SemiInt,
}


EQUATION_MAP = {
    "eq": EquType.E,
    "leq": EquType.L,
    "geq": EquType.G,
    "nonbinding": EquType.N,
    "external": EquType.X,
}

UPDATE_ACTION_MAP = {
    "l": UpdateAction.Primal,
    "m": UpdateAction.Dual,
    "up": UpdateAction.Upper,
    "lo": UpdateAction.Lower,
    "fx": UpdateAction.Fixed,
}


class ModelInstance:
    """
    ModelInstance class provides a controlled way of modifying a model instance
    and solving the resulting problem in the most efficient way, by communicating
    only the changes of the model to the solver and doing a hot start (in case of
    a continuous model like LP) without the use of disk IO.

    Parameters
    ----------
    container : Container
    model : Model
    modifiables : list[Parameter  |  ImplicitParameter]
    freeze_options : Options | dict | None, optional
    """

    def __init__(
        self,
        container: Container,
        model: Model,
        modifiables: list[Parameter | ImplicitParameter],
        freeze_options: Options | None = None,
    ):
        self.container = container
        self.job_name = container._job
        self.gms_file = self.job_name + ".gms"
        self.pf_file = self.job_name + ".pf"
        self.save_file = self.job_name + ".g00"
        self._create_restart_file()

        self.model = model
        assert self.model._is_frozen

        self.modifiables = self._init_modifiables(modifiables)
        self.instance_container = gt.Container(
            system_directory=container.system_directory,
        )

        self._debugging_level = self._get_debugging_level(
            container._debugging_level
        )
        self.workspace = GamsWorkspace(
            container.working_directory,
            container.system_directory,
            debug=self._debugging_level,
        )
        self.checkpoint = GamsCheckpoint(
            self.workspace,
            self.save_file,
        )
        self.instance = self.checkpoint.add_modelinstance()
        self.instantiate(model, freeze_options)

    def _get_debugging_level(self, debugging_level: str) -> int:
        DEBUGGING_MAP = {
            "delete": DebugLevel.Off,
            "keep_on_error": DebugLevel.KeepFilesOnError,
            "keep": DebugLevel.KeepFiles,
        }
        return DEBUGGING_MAP[debugging_level]

    def _create_restart_file(self):
        with open(self.gms_file, "w", encoding="utf-8") as gams_file:
            gams_file.write("")

        options = Options()
        scrdir = self.container._process_directory
        extra_options = {
            "input": self.gms_file,
            "sysdir": self.container.system_directory,
            "scrdir": scrdir,
            "scriptnext": os.path.join(scrdir, "gamsnext.sh"),
            "previouswork": 1,
            "license": utils._get_license_path(
                self.container.system_directory
            ),
            "save": self.save_file,
        }
        options._set_extra_options(extra_options)
        options._export(self.pf_file)

        self.container._send_job(self.job_name, self.pf_file)

    def instantiate(self, model: Model, options: Options | None = None):
        modifiers = self._create_modifiers()

        solve_string = f"{model.name} using {model.problem}"

        if model.problem not in [gp.Problem.MCP, gp.Problem.CNS]:
            solve_string += f" {model.sense}"

        if model._objective_variable is not None:
            solve_string += f" {model._objective_variable.gamsRepr()}"

        gams_options = self._prepare_gams_options(options)
        self.instance.instantiate(solve_string, modifiers, gams_options)

    def solve(
        self,
        solver: str | None,
        given_options: ModelInstanceOptions | None = None,
        output: io.TextIOWrapper | None = None,
    ):
        # get options from dict
        options, update_type = self._prepare_options(solver, given_options)

        # update sync_db
        self.container.write(self.instance.sync_db._gmd, eps_to_zero=False)

        for symbol in self.modifiables:
            if (
                isinstance(symbol, implicits.ImplicitParameter)
                and symbol.parent.records is not None
            ):
                parent_name, attr = symbol.name.split(".")
                attr_name = "_".join([parent_name, attr])

                columns = self._get_columns_to_drop(attr)

                self.instance_container[attr_name].setRecords(
                    self.container[parent_name].records.drop(columns, axis=1)
                )

                self.instance_container.write(
                    self.instance.sync_db._gmd, eps_to_zero=False
                )

        try:
            self.instance.solve(update_type, output, mi_opt=options)
        except GamsException as e:
            raise GamspyException(e.value) from e
        self._update_main_container()

        # update model status
        self.model._status = gp.ModelStatus(self.instance.model_status)
        self.model._solve_status = self.instance.solver_status

    def _init_modifiables(
        self, modifiables: list[Parameter | ImplicitParameter]
    ) -> list[Parameter | ImplicitParameter]:
        if not isinstance(modifiables, Iterable):
            raise ValidationError(
                "Modifiables must be iterable (i.e. list, tuple etc.)."
            )

        if any(
            not isinstance(symbol, (gp.Parameter, implicits.ImplicitParameter))
            for symbol in modifiables
        ):
            raise ValidationError(
                "Type of a modifiable must be either Parameter or a Variable attribute (e.g. variable.up)"
            )

        symbols_in_conditions = []
        for equation in self.model.equations:
            symbols_in_conditions += (
                equation._definition._find_symbols_in_conditions()
            )

        will_be_modified: list[Parameter | ImplicitParameter] = []
        for symbol in modifiables:
            if isinstance(symbol, implicits.ImplicitParameter):
                if symbol.parent.name in symbols_in_conditions:
                    raise ValidationError(
                        f"Modifiable symbol `{symbol.parent.name}` cannot be in a condition."
                    )

                attr_name = symbol.name.split(".")[-1]

                # If the symbol attr is fx, then modify level, lower and upper.
                if attr_name == "fx":
                    if not utils.isin(symbol.parent.l, will_be_modified):
                        will_be_modified.append(symbol.parent.l)

                    if not utils.isin(symbol.parent.lo, will_be_modified):
                        will_be_modified.append(symbol.parent.lo)

                    if not utils.isin(symbol.parent.up, will_be_modified):
                        will_be_modified.append(symbol.parent.up)
                else:
                    # if fx already added level, lower or upper, do not add again.
                    if not utils.isin(symbol, will_be_modified):
                        will_be_modified.append(symbol)
            else:
                if symbol.name in symbols_in_conditions:
                    raise ValidationError(
                        f"Modifiable symbol `{symbol.name}` cannot be in a condition."
                    )
                will_be_modified.append(symbol)

        return will_be_modified

    def _prepare_gams_options(
        self, given_options: Options | dict | None
    ) -> GamsOptions:
        options = GamsOptions(self.workspace)
        if given_options is None:
            return options

        options_dict = given_options._get_gams_compatible_options()  # type: ignore

        for key, value in options_dict.items():
            setattr(options, key, value)

        return options

    def _prepare_options(
        self,
        solver: str | None,
        given_options: ModelInstanceOptions | None,
    ) -> tuple[GamsModelInstanceOpt | None, SymbolUpdateType]:
        update_type = SymbolUpdateType.BaseCase
        options = GamsModelInstanceOpt()

        if solver is not None:
            options.solver = solver

        if given_options is not None:
            for key, value in given_options.items():
                setattr(options, key, value)

                if key == "update_type":
                    update_type = value

        return options, update_type

    def _get_columns_to_drop(self, attr):
        attr_map = {
            "l": "level",
            "m": "marginal",
            "lo": "lower",
            "up": "upper",
            "scale": "scale",
        }

        columns = []
        for key, value in attr_map.items():
            if key != attr:
                columns.append(value)

        return columns

    def _create_modifiers(self):
        modifiers = []

        for symbol in self.modifiables:
            if isinstance(symbol, gp.Parameter):
                modifiers.append(
                    GamsModifier(
                        self.instance.sync_db.add_parameter(
                            symbol.name,
                            symbol.dimension,
                            symbol.description,
                        )
                    )
                )

            elif isinstance(symbol, implicits.ImplicitParameter):
                attribute = symbol.name.split(".")[-1]
                update_action = UPDATE_ACTION_MAP[attribute]

                try:
                    sync_db_symbol = self.instance.sync_db[symbol.parent.name]
                except GamsException:
                    if isinstance(symbol.parent, gp.Variable):
                        sync_db_symbol = self.instance.sync_db.add_variable(
                            symbol.parent.name,
                            symbol.parent.dimension,
                            VARIABLE_MAP[symbol.parent.type],
                        )

                    elif isinstance(symbol.parent, gp.Equation):
                        sync_db_symbol = self.instance.sync_db.add_equation(
                            symbol.parent.name,
                            symbol.parent.dimension,
                            EQUATION_MAP[symbol.parent.type],
                        )

                attr_name = "_".join(symbol.name.split("."))

                domain = (
                    ["*"] * symbol.parent.dimension
                    if symbol.parent.dimension
                    else None
                )
                _ = gt.Parameter(
                    self.instance_container,
                    attr_name,
                    domain=domain,
                )

                data_symbol = self.instance.sync_db.add_parameter(
                    attr_name,
                    symbol.parent.dimension,
                )

                modifiers.append(
                    GamsModifier(sync_db_symbol, update_action, data_symbol)
                )
            else:
                raise ValidationError(
                    f"Symbol type {type(symbol)} cannot be modified in a"
                    " frozen solve"
                )

        return modifiers

    def _update_main_container(self):
        temp = gt.Container(
            system_directory=self.container.system_directory,
        )
        temp.read(self.instance.sync_db._gmd)

        prev_state = self.container._options.miro_protect
        for name in temp.data:
            if name in self.container.data:
                self.container._options.miro_protect = False
                self.container[name].records = temp[name].records
                self.container[name].domain_labels = self.container[
                    name
                ].domain_names

            if name in [symbol.name for symbol in self.modifiables]:
                _ = gp.Variable(
                    self.container,
                    name + "_var",
                    domain=self.container[name].domain,
                    records=temp[name + "_var"].records,
                )
        self.container._options.miro_protect = prev_state

        if self.model._objective_variable is not None:
            self.model._objective_value = temp[
                self.model._objective_variable.name
            ].toValue()
