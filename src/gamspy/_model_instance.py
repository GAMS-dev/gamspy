import math
import gamspy as gp
import gamspy._symbols._implicits as implicits
from gamspy.exceptions import GamspyException
from gams import (
    GamsModifier,
    UpdateAction,
    VarType,
    EquType,
)


from typing import (
    List,
    Union,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from gamspy import Parameter, Model, Container
    from gamspy._symbols._implicits import ImplicitParameter


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
    def __init__(
        self,
        container: "Container",
        model: "Model",
        modifiables: List[Union["Parameter", "ImplicitParameter"]],
    ) -> None:
        self.modifiables = modifiables
        self.main_container = container

        self.checkpoint = self.main_container._restart_from
        self.instance = self.checkpoint.add_modelinstance()
        self.instantiate(model)

    def update_sync_db(self):
        self.main_container.write(self.instance.sync_db._gmd)

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

                attr_param = gp.Parameter(
                    self.main_container,
                    attr_name,
                    domain=modifiable.parent.domain,
                )

                def value_func(seed=None, size=None):
                    return math.inf

                attr_param.generateRecords(density=1.0, func=value_func)
                print(attr_param.records)

                data_symbol = self.instance.sync_db.add_parameter(
                    attr_name,
                    modifiable.parent.dimension,
                )

                modifiers.append(
                    GamsModifier(sync_db_symbol, update_action, data_symbol)
                )
            else:
                raise GamspyException(
                    f"Symbol type {type(modifiable)} cannot be modified in a"
                    " frozen solve"
                )

        return modifiers

    def instantiate(self, model: "Model"):
        solve_string = (
            f"{model.name} use"  # type: ignore
            f" {model.problem} {model.sense} {model._objective_variable.name}"
        )

        modifiers = self._create_modifiers()

        self.instance.instantiate(solve_string, modifiers)

    def solve(self):
        self.instance.solve()

    @property
    def model_status(self):
        return self.instance.model_status

    @property
    def solver_status(self):
        return self.instance.solver_status

    def update_main_container(self):
        instance_container = gp.Container(name="instance_container")
        instance_container.read(self.instance.sync_db._gmd)

        for name in instance_container.data.keys():
            if name in self.main_container.data.keys():
                self.main_container[name].setRecords(
                    instance_container[name].records
                )
