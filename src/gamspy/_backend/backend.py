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
    @abstractmethod
    def is_async(self):
        ...

    @abstractmethod
    def preprocess(self, dirty_names: List[str], modified_names: List[str]):
        ...

    @abstractmethod
    def run(self):
        ...

    @abstractmethod
    def postprocess(self, is_implicit: bool = False):
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
