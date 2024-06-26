from __future__ import annotations

import io
import os
from typing import TYPE_CHECKING

import gams.transfer as gt
from gams import (
    EquType,
    GamsCheckpoint,
    GamsException,
    GamsModelInstanceOpt,
    GamsModifier,
    GamsOptions,
    SymbolUpdateType,
    UpdateAction,
    VarType,
)

import gamspy as gp
import gamspy._symbols.implicits as implicits
import gamspy.utils as utils
from gamspy._options import ModelInstanceOptions, Options
from gamspy.exceptions import ValidationError

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
    "cone": EquType.C,
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
        freeze_options: Options | dict | None = None,
    ):
        self.container = container
        self.job_name = container._job
        self.gms_file = self.job_name + ".gms"
        self.pf_file = self.job_name + ".pf"
        self.save_file = self.job_name + ".g00"
        self._create_restart_file()

        self.modifiables = self._init_modifiables(modifiables)
        self.instance_container = gt.Container(
            system_directory=container.system_directory,
        )
        self.model = model

        self.checkpoint = GamsCheckpoint(container.workspace, self.save_file)
        self.instance = self.checkpoint.add_modelinstance()
        self.instantiate(model, freeze_options)

    def _create_restart_file(self):
        with open(self.gms_file, "w") as gams_file:
            gams_file.write("")

        options = Options()
        scrdir = os.path.join(self.container.working_directory, "225a")
        extra_options = {
            "input": self.gms_file,
            "sysdir": self.container.system_directory,
            "scrdir": scrdir,
            "scriptnext": os.path.join(scrdir, "gamsnext.sh"),
            "writeoutput": 0,
            "logoption": 0,
            "previouswork": 1,
            "license": utils._get_license_path(
                self.container.system_directory
            ),
            "save": self.save_file,
        }
        options._set_extra_options(extra_options)
        options.export(self.pf_file)

        self.container._send_job(self.job_name, self.pf_file)

    def instantiate(
        self, model: Model, freeze_options: Options | dict | None = None
    ):
        modifiers = self._create_modifiers()

        options = self._prepare_freeze_options(freeze_options)

        solve_string = f"{model.name} using {model.problem}"

        if model.sense:
            solve_string += f" {model.sense}"

        if model._objective_variable is not None:
            solve_string += f" {model._objective_variable.gamsRepr()}"

        self.instance.instantiate(solve_string, modifiers, options)

    def solve(
        self,
        given_options: ModelInstanceOptions | dict | None = None,
        output: io.TextIOWrapper | None = None,
    ):
        # get options from dict
        options, update_type = self._prepare_options(given_options)

        # update sync_db
        self.container.write(self.instance.sync_db._gmd)

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

                self.instance_container.write(self.instance.sync_db._gmd)

        self.instance.solve(update_type, output, mi_opt=options)
        self._update_main_container()

        # update model status
        self.model.status = gp.ModelStatus(self.instance.model_status)
        self.model.solver_status = self.instance.solver_status

    def _init_modifiables(
        self, modifiables: list[Parameter | ImplicitParameter]
    ) -> list[Parameter | ImplicitParameter]:
        will_be_modified: list[Parameter | ImplicitParameter] = []
        for symbol in modifiables:
            if isinstance(symbol, implicits.ImplicitParameter):
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
                will_be_modified.append(symbol)

        return will_be_modified

    def _prepare_freeze_options(
        self, given_options: Options | dict | None
    ) -> GamsOptions:
        if isinstance(given_options, Options):
            given_options = given_options._get_gams_compatible_options()

        options = GamsOptions(self.model.container.workspace)

        if given_options is None:
            return options

        for key, value in given_options.items():
            setattr(options, key, value)

        return options

    def _prepare_options(
        self, given_options: ModelInstanceOptions | dict | None
    ) -> tuple[GamsModelInstanceOpt | None, SymbolUpdateType]:
        update_type = SymbolUpdateType.BaseCase

        if given_options is None:
            return None, update_type

        options = GamsModelInstanceOpt()

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

        for name in temp.data:
            if name in self.container.data:
                self.container[name].records = temp[name].records

        if self.model._objective_variable is not None:
            self.model.objective_value = temp[
                self.model._objective_variable.name
            ].toValue()
