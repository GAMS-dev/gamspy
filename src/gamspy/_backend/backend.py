from __future__ import annotations

import os
from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Literal
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union

import pandas as pd

from gamspy.exceptions import GamspyException

if TYPE_CHECKING:
    import io
    from gamspy._backend.engine import EngineConfig, GAMSEngine
    from gamspy._backend.neos import NeosClient, NEOSServer
    from gamspy._backend.local import Local
    from gams import GamsOptions
    from gamspy import Container

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
    container: "Container",
    options: Optional["GamsOptions"] = None,
    output: Optional["io.TextIOWrapper"] = None,
    backend: Literal["local", "engine", "neos"] = "local",
    engine_config: Optional["EngineConfig"] = None,
    neos_client: Optional["NeosClient"] = None,
) -> Union["Local", "GAMSEngine", "NEOSServer"]:
    if backend == "neos":
        from gamspy._backend.neos import NEOSServer

        return NEOSServer(container, options, neos_client)
    elif backend == "engine":
        from gamspy._backend.engine import GAMSEngine

        return GAMSEngine(container, engine_config, options, output)
    elif backend == "local":
        from gamspy._backend.local import Local

        return Local(container, options, output)
    else:
        raise GamspyException(
            f"`{backend}` is not a valid backend. Possible backends:"
            " local, engine, and neos"
        )


class Backend(ABC):
    def __init__(self, container: "Container", gdx_in: str, gdx_out: str):
        self.container = container
        self.gdx_in = gdx_in
        self.gdx_out = gdx_out

    @abstractmethod
    def is_async(self):
        ...

    def preprocess(self):
        dirty_names, modified_names = (
            self.container._get_touched_symbol_names()
        )
        self.clean_dirty_symbols(dirty_names)
        self.container.isValid(verbose=True, force=True)
        self.container.write(self.container._gdx_in, modified_names)

        gams_string = self.container._generate_gams_string(
            self.gdx_in, self.gdx_out, dirty_names, modified_names
        )

        return gams_string, dirty_names, modified_names

    @abstractmethod
    def run(self, gams_string: str):
        ...

    @abstractmethod
    def postprocess(
        self,
        dirty_names: List[str],
        modified_names: List[str],
        is_implicit: bool = False,
        keep_flags: bool = False,
    ):
        ...

    def prepare_summary(self, working_directory: str, trace_file: str):
        from gamspy._model import ModelStatus

        with open(os.path.join(working_directory, trace_file)) as file:
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

    def update_modified_state(self, modified_names: List[str]):
        for name in modified_names:
            self.container[name].modified = False

    def clean_dirty_symbols(self, dirty_names: List[str]):
        for name in dirty_names:
            self.container[name]._is_dirty = False
