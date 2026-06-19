from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, TextIO, no_type_check

import pandas as pd

import gamspy.utils as utils
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from pathlib import Path

    from gamspy import Container, Model, Options
    from gamspy._backend.engine import EngineClient, GAMSEngine
    from gamspy._backend.local import Local
    from gamspy._backend.neos import NeosClient, NEOSServer

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


@no_type_check
def backend_factory(
    container: Container,
    options: Options | None = None,
    solver: str | None = None,
    solver_options: dict | Path | None = None,
    output: TextIO | None = None,
    backend: Literal["local", "engine", "neos"] = "local",
    client: EngineClient | NeosClient | None = None,
    model: Model | None = None,
) -> Local | GAMSEngine | NEOSServer:  # pragma: no cover
    if backend == "neos":
        from gamspy._backend.neos import NeosClient, NEOSServer

        if client is None or not isinstance(client, NeosClient):
            raise ValidationError(
                "`NeosClient` must be provided to solve on NEOS Server."
            )

        return NEOSServer(
            container,
            options,
            solver,
            solver_options,
            output,
            model,
            client,
        )
    elif backend == "engine":
        from gamspy._backend.engine import EngineClient, GAMSEngine

        if client is None or not isinstance(client, EngineClient):
            raise ValidationError(
                "`engine_client` must be provided to solve on GAMS Engine"
            )

        return GAMSEngine(
            container,
            options,
            solver,
            solver_options,
            output,
            model,
            client,
        )
    elif backend == "local":
        from gamspy._backend.local import Local

        return Local(
            container,
            options,
            solver,
            solver_options,
            output,
            model,
        )

    raise ValidationError(
        f"`{backend}` is not a valid backend. Possible backends:"
        " local, engine, and neos"
    )


class Backend(ABC):
    def __init__(
        self,
        backend_type: str,
        container: Container,
        options: Options,
        solver: str | None,
        solver_options: dict | Path | None,
        output: TextIO | None,
    ):
        self.backend_type = backend_type
        self.container = container
        self.options = options
        self.solver = solver
        self.solver_options = solver_options
        self.output = output

        self.job_name = self.get_job_name()
        self.gms_file = self.job_name + ".gms"
        self.lst_file = self.job_name + ".lst"
        self.pf_file = self.job_name + ".pf"
        self.restart_file = self.job_name + ".g00"
        self.trace_file = self.job_name + ".txt"

    @abstractmethod
    def is_async(self): ...

    @abstractmethod
    def run(self): ...

    def get_job_name(self):
        job_name = self.container._job

        if self.container._debugging_level == "keep":
            job_name = os.path.join(
                self.container.working_directory,
                "_" + utils._get_unique_name(),
            )
            self.container._job = job_name
            self.container._gdx_in = f"{job_name}in.gdx"
            self.container._gdx_out = f"{job_name}out.gdx"

        return job_name

    def preprocess(self):
        symbol_names = self.container._symbols_to_unload()

        if len(symbol_names) != 0:
            self.container._write(
                self.container._gdx_in, symbol_names, eps_to_zero=False
            )

        gdx_in = self.container._gdx_in
        if self.backend_type == "engine":
            gdx_in = os.path.basename(gdx_in)
        elif self.backend_type == "neos":
            gdx_in = "in.gdx"
        gams_string = self.container._generate_gams_string(gdx_in, symbol_names)

        return gams_string

    def parse_listings(self, model: Model) -> None:
        listing_file = (
            self.options.listing_file
            if self.options.listing_file
            else self.job_name + ".lst"
        )
        if self.options.equation_listing_limit:
            utils._parse_generated_equations(model, listing_file)

        if self.options.variable_listing_limit:
            utils._parse_generated_variables(model, listing_file)

    def prepare_summary(self, model: Model) -> pd.DataFrame:
        df = pd.DataFrame(
            [
                [
                    model.solve_status.name,
                    model.status.name,
                    model.objective_value,
                    model.num_equations,
                    model.num_variables,
                    model.used_model_type,
                    self.solver.upper(),  # type: ignore
                    model.total_solver_time,
                ]
            ],
            columns=HEADER,
        )
        return df
