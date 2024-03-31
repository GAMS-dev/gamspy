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
import uuid
from typing import TYPE_CHECKING

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
        self.container = container
        save_file = self._create_restart_file()
        restart_from = GamsCheckpoint(container.workspace, save_file)

        self.modifiables = self._init_modifiables(modifiables)
        self.instance_container = gp.Container(
            system_directory=container.system_directory,
            working_directory=container.workspace.working_directory,
        )
        self.model = model

        self.checkpoint = restart_from
        self.instance = self.checkpoint.add_modelinstance()
        self.instantiate(model, freeze_options)

    def _create_restart_file(self):
        job_id = f"_instance_restart_{uuid.uuid4()}"
        job_name = os.path.join(self.container.working_directory, job_id)
        with open(job_name + ".gms", "w") as gams_file:
            gams_file.write("")

        options = GamsOptions(self.container.workspace)
        options._input = job_name + ".gms"
        options._sysdir = utils._get_gamspy_base_directory()
        scrdir = os.path.join(self.container.working_directory, "225a")
        options._scrdir = scrdir
        options._scriptnext = os.path.join(scrdir, "gamsnext.sh")
        options._writeoutput = 0
        options.previouswork = 1
        options._logoption = 0
        options._save = job_name + ".g00"

        pf_file = job_name + ".pf"
        options.export(pf_file)

        self.container._send_job(job_name, pf_file)

        return job_name + ".g00"

    def instantiate(
        self, model: Model, freeze_options: Options | dict | None = None
    ):
        options = self._prepare_freeze_options(freeze_options)

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
        self.container.write(self.instance.sync_db._gmd)

        for modifiable in self.modifiables:
            if (
                isinstance(modifiable, implicits.ImplicitParameter)
                and modifiable.parent.records is not None
            ):
                parent_name, attr = modifiable.name.split(".")
                attr_name = "_".join([parent_name, attr])

                columns = self._get_columns_to_drop(attr)

                self.instance_container[attr_name].setRecords(
                    self.container[parent_name].records.drop(columns, axis=1)
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
            system_directory=self.container.system_directory,
            working_directory=self.container.workspace.working_directory,
        )
        temp.read(self.instance.sync_db._gmd)

        for name in temp.data:
            if name in self.container.data:
                self.container[name].records = temp[name].records
