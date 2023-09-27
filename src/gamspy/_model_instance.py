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
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union

from gams import EquType
from gams import GamsModelInstanceOpt
from gams import GamsModifier
from gams import GamsOptions
from gams import SymbolUpdateType
from gams import UpdateAction
from gams import VarType

import gamspy as gp
import gamspy._symbols.implicits as implicits
from gamspy.exceptions import GamspyException

if TYPE_CHECKING:
    from gamspy import Parameter, Model, Container
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

update_type_map = {
    "0": SymbolUpdateType.Zero,
    "base_case": SymbolUpdateType.BaseCase,
    "accumulate": SymbolUpdateType.Accumulate,
    "inherit": SymbolUpdateType._Inherit,
}


class ModelInstance:
    def __init__(
        self,
        container: "Container",
        model: "Model",
        modifiables: List[Union["Parameter", "ImplicitParameter"]],
        freeze_options: Optional[dict] = None,
    ) -> None:
        self.modifiables = modifiables
        self.main_container = container
        self.instance_container = gp.Container(
            name="instance_container",
            system_directory=container.system_directory,
            working_directory=container.workspace.working_directory,
        )
        self.model = model

        self.checkpoint = self.main_container._restart_from
        self.instance = self.checkpoint.add_modelinstance()
        self.instantiate(model, freeze_options)

    def instantiate(
        self, model: "Model", freeze_options: Optional[dict] = None
    ):
        options = self._prepare_freeze_options(freeze_options)

        solve_string = (
            f"{model.name} use"  # type: ignore
            f" {model.problem} {model.sense} {model._objective_variable.name}"
        )

        modifiers = self._create_modifiers()

        self.instance.instantiate(solve_string, modifiers, options)

    def solve(
        self,
        options_dict: Optional[dict] = None,
        output: Optional[io.TextIOWrapper] = None,
    ):
        # get options from dict
        options, update_type = self._prepare_options(options_dict)

        # update sync_db
        self.main_container.write(self.instance.sync_db._gmd)

        for modifiable in self.modifiables:
            if isinstance(modifiable, implicits.ImplicitParameter):
                if modifiable.parent.records is not None:
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

    def _prepare_freeze_options(
        self, options_dict: Optional[dict]
    ) -> GamsOptions:
        if options_dict is None:
            return

        options = GamsOptions(self.model.container.workspace)

        for key, value in options_dict.items():
            setattr(options, key, value)

    def _prepare_options(
        self, options_dict: Optional[dict]
    ) -> Tuple[Optional[GamsModelInstanceOpt], SymbolUpdateType]:
        update_type = SymbolUpdateType.BaseCase

        if options_dict is None:
            return None, update_type

        possible_options = [
            "solver",
            "opt_file",
            "no_match_limit",
            "debug",
            "update_type",
        ]
        options = GamsModelInstanceOpt()

        for key, value in options_dict.items():
            if key in possible_options:
                setattr(options, key, value)
            else:
                raise GamspyException(
                    f"{key} is not a model instance option. All options:"
                    f" {possible_options}"
                )

            if key == "update_type":
                update_type = update_type_map[key]

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
                raise GamspyException(
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

        for name in temp.data.keys():
            if name in self.main_container.data.keys():
                self.main_container[name].records = temp[name].records
