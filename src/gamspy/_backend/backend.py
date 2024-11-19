from __future__ import annotations

import os
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

import pandas as pd

import gamspy._symbols as syms
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
    solver: str | None = None,
    solver_options: dict | None = None,
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
            solver,
            solver_options,
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
            solver,
            solver_options,
            output,
            model,
            load_symbols,
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
            load_symbols,
        )

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
        solver: str | None,
        solver_options: dict | None,
        output: io.TextIOWrapper | None,
        load_symbols: list[Symbol] | None,
    ):
        self.container = container
        self.model = model
        self.options = options
        self.solver = solver
        self.solver_options = solver_options
        self.output = output
        self.load_symbols = load_symbols
        if load_symbols is not None:
            self.load_symbols = [symbol.name for symbol in load_symbols]  # type: ignore

    @abstractmethod
    def is_async(self): ...

    @abstractmethod
    def run(
        self,
        relaxed_domain_mapping: bool = False,
        gams_to_gamspy: bool = False,
    ): ...

    def get_job_name(self):
        job_name = self.container._job

        if self.container._debugging_level == "keep":
            job_name = os.path.join(
                self.container.working_directory, "_" + str(uuid.uuid4())
            )

        return job_name

    def preprocess(self, gdx_in: str):
        modified_names = self.container._get_modified_symbols()

        if len(modified_names) != 0:
            self.container.write(
                self.container._gdx_in, modified_names, eps_to_zero=False
            )

        gams_string = self.container._generate_gams_string(
            gdx_in, modified_names
        )

        self.update_modified_state(modified_names)

        return gams_string

    def load_records(self, relaxed_domain_mapping: bool = False):
        if self.load_symbols is not None:
            symbols = self.load_symbols
        else:
            symbols = utils._get_symbol_names_from_gdx(
                self.container.system_directory, self.container._gdx_out
            )
            filtered_names = []
            for name in symbols:
                # addGamsCode symbols
                if name not in self.container:
                    filtered_names.append(name)
                    continue

                symbol = self.container[name]
                if isinstance(symbol, syms.Alias):
                    filtered_names.append(name)
                    continue

                if symbol.synchronize:
                    filtered_names.append(name)

            symbols = filtered_names

        if len(symbols) != 0:
            self.container._load_records_from_gdx(
                self.container._gdx_out, symbols
            )

        if relaxed_domain_mapping:
            # Best attempt approach to map relaxed domain to actual symbols
            for name in symbols:
                symbol = self.container[name]

                new_domain = []
                for elem in symbol.domain:
                    if (
                        isinstance(elem, str)
                        and elem != "*"
                        and elem in self.container
                    ):
                        new_domain.append(self.container[elem])
                    else:
                        new_domain.append(elem)

                symbol.domain = new_domain
                symbol.dimension = len(new_domain)
                if isinstance(symbol, (syms.Variable, syms.Equation)):
                    symbol._update_attr_domains()

    def parse_listings(self):
        listing_file = (
            self.options.listing_file
            if self.options.listing_file
            else self.job_name + ".lst"
        )
        if self.options.equation_listing_limit:
            utils._parse_generated_equations(self.model, listing_file)

        if self.options.variable_listing_limit:
            utils._parse_generated_variables(self.model, listing_file)

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
