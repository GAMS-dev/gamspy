from __future__ import annotations

import os
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

import pandas as pd
from gams import DebugLevel

import gamspy.utils as utils
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    import io

    from gamspy import Container, Model, Options
    from gamspy._backend.engine import EngineClient, GAMSEngine
    from gamspy._backend.local import Local
    from gamspy._backend.neos import NeosClient, NEOSServer
    from gamspy._symbols.symbol import Symbol

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
    load_symbols: list[Symbol] | None = None,
) -> Local | GAMSEngine | NEOSServer:
    if backend == "neos":
        from gamspy._backend.neos import NEOSServer

        return NEOSServer(
            container,
            options,  # type: ignore
            client,  # type: ignore
            output,
            model,
            load_symbols,
        )
    elif backend == "engine":
        from gamspy._backend.engine import GAMSEngine

        return GAMSEngine(
            container,
            client,  # type: ignore
            options,  # type: ignore
            output,
            model,
            load_symbols,
        )
    elif backend == "local":
        from gamspy._backend.local import Local

        return Local(container, options, output, model, load_symbols)

    raise ValidationError(
        f"`{backend}` is not a valid backend. Possible backends:"
        " local, engine, and neos"
    )


class Backend(ABC):
    def __init__(
        self,
        container: Container,
        model: Model,
        options: Options,
        output: io.TextIOWrapper | None,
        load_symbols: list[Symbol] | None,
    ):
        self.container = container
        self.model = model
        self.options = options
        self.output = output
        self.load_symbols = load_symbols
        if load_symbols is not None:
            self.load_symbols = [symbol.name for symbol in load_symbols]  # type: ignore

    @abstractmethod
    def is_async(self): ...

    @abstractmethod
    def run(self, keep_flags: bool = False): ...

    def get_job_name(self):
        job_name = self.container._job

        if self.container._debugging_level == DebugLevel.KeepFiles:
            job_name = os.path.join(
                self.container.working_directory, "_" + str(uuid.uuid4())
            )

        return job_name

    def preprocess(self, gdx_in: str, keep_flags: bool = False):
        modified_names = self.container._get_touched_symbol_names()

        if len(modified_names) != 0:
            self.container.write(
                self.container._gdx_in, modified_names, eps_to_zero=False
            )

        gams_string = self.container._generate_gams_string(
            gdx_in, modified_names
        )

        if not keep_flags:
            self.update_modified_state(modified_names)

        return gams_string

    def postprocess(self):
        if self.load_symbols is not None:
            symbols = self.load_symbols
        else:
            symbols = utils._get_symbol_names_from_gdx(
                self.container.system_directory, self.container._gdx_out
            )

        if len(symbols) != 0:
            self.container._load_records_from_gdx(
                self.container._gdx_out, symbols
            )

    def prepare_summary(self, trace_file: str) -> pd.DataFrame:
        from gamspy._model import ModelStatus

        with open(trace_file, encoding="utf-8") as file:
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
