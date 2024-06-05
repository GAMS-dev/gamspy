from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

import pandas as pd

from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    import io

    from gamspy import Container, Model, Options
    from gamspy._backend.engine import EngineClient, GAMSEngine
    from gamspy._backend.local import Local
    from gamspy._backend.neos import NeosClient, NEOSServer

SOLVE_STATUS = [
    "",
    "Normal",
    "Iteration",
    "Resource",
    "Solver",
    "EvalError",
    "Capability",
    "License",
    "User",
    "SetupErr",
    "SolverErr",
    "InternalErr",
    "Skipped",
    "SystemErr",
]
HEADER = [
    "Solver Status",
    "Model Status",
    "Objective",
    "Num of Equations",
    "Num of Variables",
    "Model Type",
    "Solver",
    "Solver Time",
]


def backend_factory(
    container: Container,
    options: Options | None = None,
    output: io.TextIOWrapper | None = None,
    backend: Literal["local", "engine", "neos"] = "local",
    client: EngineClient | NeosClient | None = None,
    model: Model | None = None,
) -> Local | GAMSEngine | NEOSServer:
    if backend == "neos":
        from gamspy._backend.neos import NEOSServer

        return NEOSServer(container, options, client, model)  # type: ignore
    elif backend == "engine":
        from gamspy._backend.engine import GAMSEngine

        return GAMSEngine(container, client, options, output, model)  # type: ignore
    elif backend == "local":
        from gamspy._backend.local import Local

        return Local(container, options, output, model)

    raise ValidationError(
        f"`{backend}` is not a valid backend. Possible backends:"
        " local, engine, and neos"
    )


class Backend(ABC):
    def __init__(self, container: Container, gdx_in: str, gdx_out: str):
        self.container = container
        self.gdx_in = gdx_in
        self.gdx_out = gdx_out

    @abstractmethod
    def is_async(self): ...

    @abstractmethod
    def solve(self, keep_flags: bool = False): ...

    def preprocess(self, keep_flags: bool = False):
        (
            dirty_names,
            modified_names,
        ) = self.container._get_touched_symbol_names()
        self.clean_dirty_symbols(dirty_names)

        if len(modified_names) != 0:
            self.container.write(self.container._gdx_in, modified_names)

        gams_string = self.container._generate_gams_string(
            self.gdx_in, self.gdx_out, dirty_names, modified_names
        )

        if not keep_flags:
            self.update_modified_state(modified_names)

        return gams_string, dirty_names

    def prepare_summary(self, working_directory: str, trace_file: str):
        from gamspy._model import ModelStatus

        with open(
            os.path.join(working_directory, trace_file), encoding="utf-8"
        ) as file:
            line = file.readlines()[-1]
            (
                _,
                model_type,
                solver_name,
                _,
                _,
                _,
                _,
                num_equations,
                num_variables,
                _,
                _,
                _,
                _,
                model_status,
                solver_status,
                objective_value,
                _,
                solver_time,
                _,
                _,
                _,
                _,
            ) = line.split(",")

        dataframe = pd.DataFrame(
            [
                [
                    SOLVE_STATUS[int(solver_status)],
                    ModelStatus(int(model_status)).name,
                    objective_value,
                    num_equations,
                    num_variables,
                    model_type,
                    solver_name,
                    solver_time,
                ]
            ],
            columns=HEADER,
        )
        return dataframe

    def update_modified_state(self, modified_names: list[str]):
        for name in modified_names:
            if not name.startswith("autogenerated_"):
                self.container[name].modified = False

    def clean_dirty_symbols(self, dirty_names: list[str]):
        for name in dirty_names:
            if self.container[name].synchronize:
                self.container[name]._is_dirty = False
