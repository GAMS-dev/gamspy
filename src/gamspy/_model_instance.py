from __future__ import annotations

import io
from typing import TYPE_CHECKING

from gams import (
    EquType,
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


variable_map = {
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


equation_map = {
    "eq": EquType.E,
    "leq": EquType.L,
    "geq": EquType.G,
    "nonbinding": EquType.N,
    "external": EquType.X,
    "cone": EquType.C,
}

update_action_map = {
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
        self.modifiables = self._init_modifiables(modifiables)
        self.main_container = container
        self.instance_container = gp.Container(
            system_directory=container.system_directory,
            working_directory=container.workspace.working_directory,
        )
        self.model = model

        self.checkpoint = self.main_container._restart_from
        self.instance = self.checkpoint.add_modelinstance()
        self.instantiate(model, freeze_options)

    def instantiate(
        self, model: Model, freeze_options: Options | dict | None = None
    ):
        options = self._prepare_freeze_options(freeze_options)
        options.license = self.model.container._license_path

        solve_string = f"{model.name} using {model.problem}"

        if model.sense:
            solve_string += f" {model.sense}"

        if model._objective_variable is not None:
            solve_string += f" {model._objective_variable.gamsRepr()}"

        modifiers = self._create_modifiers()

        self.instance.instantiate(solve_string, modifiers, options)

    def solve(
        self,
        given_options: ModelInstanceOptions | dict | None = None,
        output: io.TextIOWrapper | None = None,
    ):
        # get options from dict
        options, update_type = self._prepare_options(given_options)

        # update sync_db
        self.main_container.write(self.instance.sync_db._gmd)

        for modifiable in self.modifiables:
            if (
                isinstance(modifiable, implicits.ImplicitParameter)
                and modifiable.parent.records is not None
            ):
                parent_name, attr = modifiable.name.split(".")
                attr_name = "_".join([parent_name, attr])

                columns = self._get_columns_to_drop(attr)

                self.instance_container[attr_name].setRecords(
                    self.main_container[parent_name].records.drop(
                        columns, axis=1
                    )
                )

                self.instance_container.write(self.instance.sync_db._gmd)

        # solve
        self.instance.solve(
            update_type=update_type, output=output, mi_opt=options
        )

        # update main container
        self._update_main_container()

        # update model status
        self.model.status = gp.ModelStatus(self.instance.model_status)
        self.model.solver_status = self.instance.solver_status

    def _init_modifiables(
        self, modifiables: list[Parameter | ImplicitParameter]
    ) -> list[Parameter | ImplicitParameter]:
        will_be_modified: list[Parameter | ImplicitParameter] = []
        for modifiable in modifiables:
            if isinstance(modifiable, implicits.ImplicitParameter):
                attr_name = modifiable.name.split(".")[-1]

                # If the modifiable attr is fx, then modify level, lower and upper.
                if attr_name == "fx":
                    if not utils.isin(modifiable.parent.l, will_be_modified):
                        will_be_modified.append(modifiable.parent.l)

                    if not utils.isin(modifiable.parent.lo, will_be_modified):
                        will_be_modified.append(modifiable.parent.lo)

                    if not utils.isin(modifiable.parent.up, will_be_modified):
                        will_be_modified.append(modifiable.parent.up)
                else:
                    # if fx already added level, lower or upper, do not add again.
                    if not utils.isin(modifiable, will_be_modified):
                        will_be_modified.append(modifiable)
            else:
                will_be_modified.append(modifiable)

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

        for modifiable in self.modifiables:
            if isinstance(modifiable, gp.Parameter):
                modifiers.append(
                    GamsModifier(
                        self.instance.sync_db.add_parameter(
                            modifiable.name,
                            modifiable.dimension,
                            modifiable.description,
                        )
                    )
                )

                modifiable._is_frozen = True
            elif isinstance(modifiable, implicits.ImplicitParameter):
                attribute = modifiable.name.split(".")[-1]
                update_action = update_action_map[attribute]

                try:
                    sync_db_symbol = self.instance.sync_db[
                        modifiable.parent.name
                    ]
                except GamsException:
                    if isinstance(modifiable.parent, gp.Variable):
                        sync_db_symbol = self.instance.sync_db.add_variable(
                            modifiable.parent.name,
                            modifiable.parent.dimension,
                            variable_map[modifiable.parent.type],
                        )

                    elif isinstance(modifiable.parent, gp.Equation):
                        sync_db_symbol = self.instance.sync_db.add_equation(
                            modifiable.parent.name,
                            modifiable.parent.dimension,
                            equation_map[modifiable.parent.type],
                        )

                attr_name = "_".join(modifiable.name.split("."))

                domain = (
                    ["*"] * modifiable.parent.dimension
                    if modifiable.parent.dimension
                    else None
                )
                _ = gp.Parameter(
                    self.instance_container,
                    attr_name,
                    domain=domain,
                )

                data_symbol = self.instance.sync_db.add_parameter(
                    attr_name,
                    modifiable.parent.dimension,
                )

                modifiers.append(
                    GamsModifier(sync_db_symbol, update_action, data_symbol)
                )

                modifiable.parent._is_frozen = True
            else:
                raise ValidationError(
                    f"Symbol type {type(modifiable)} cannot be modified in a"
                    " frozen solve"
                )

        return modifiers

    @property
    def solver_status(self):
        return self.instance.solver_status

    def _update_main_container(self):
        temp = gp.Container(
            system_directory=self.main_container.system_directory,
            working_directory=self.main_container.workspace.working_directory,
        )
        temp.read(self.instance.sync_db._gmd)

        for name in temp.data:
            if name in self.main_container.data:
                self.main_container[name].records = temp[name].records
