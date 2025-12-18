from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections import deque
from typing import TYPE_CHECKING, Literal, no_type_check

import pandas as pd

import gamspy._symbols as syms
import gamspy.utils as utils
from gamspy.exceptions import GamspyException, ValidationError

if TYPE_CHECKING:
    import io
    from pathlib import Path

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


@no_type_check
def backend_factory(
    container: Container,
    options: Options | None = None,
    solver: str | None = None,
    solver_options: dict | Path | None = None,
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
            options,
            solver,
            solver_options,
            client,
            output,
            model,
            load_symbols,
        )
    elif backend == "engine":
        from gamspy._backend.engine import GAMSEngine

        return GAMSEngine(
            container,
            client,
            options,
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


def _cast_value(value: str, cast_to: type):
    return float("nan") if value == "NA" else cast_to(value)


class Backend(ABC):
    def __init__(
        self,
        backend_type: str,
        container: Container,
        model: Model | None,
        options: Options,
        solver: str | None,
        solver_options: dict | Path | None,
        output: io.TextIOWrapper | None,
        load_symbols: list[Symbol] | None,
    ):
        self.backend_type = backend_type
        self.container = container
        self.model = model
        self.options = options
        self.solver = solver
        self.solver_options = solver_options
        self.output = output
        if load_symbols is not None:
            self.load_symbols: list[str] = [
                symbol.name  # type: ignore
                for symbol in load_symbols
            ]

        self.job_name = self.get_job_name()
        self.gms_file = self.job_name + ".gms"
        self.lst_file = self.job_name + ".lst"
        self.pf_file = self.job_name + ".pf"
        self.restart_file = self.job_name + ".g00"
        self.trace_file = self.job_name + ".txt"
        self.write_trace_template()

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
                self.container.working_directory,
                "_" + utils._get_unique_name(),
            )
            self.container._job = job_name
            self.container._gdx_in = f"{job_name}in.gdx"
            self.container._gdx_out = f"{job_name}out.gdx"

        return job_name

    def preprocess(self):
        modified_names = self.container._get_modified_symbols()

        if len(modified_names) != 0:
            try:
                self.container.write(
                    self.container._gdx_in, modified_names, eps_to_zero=False
                )
            except Exception as e:
                # Unfortunately, GTP raises a blind exception here. Turn it into a GamspyException.
                raise GamspyException(str(e)) from e

        gdx_in = self.container._gdx_in
        if self.backend_type == "engine":
            gdx_in = os.path.basename(gdx_in)
        elif self.backend_type == "neos":
            gdx_in = "in.gdx"
        gams_string = self.container._generate_gams_string(gdx_in, modified_names)
        self.make_unmodified(modified_names)

        return gams_string

    def load_records(self, relaxed_domain_mapping: bool = False):
        if hasattr(self, "load_symbols"):
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
                if type(symbol) is syms.Alias:
                    filtered_names.append(name)
                    continue

                if symbol.synchronize:
                    filtered_names.append(name)

            symbols = filtered_names

        if len(symbols) != 0:
            self.container._load_records_from_gdx(self.container._gdx_out, symbols)
            self.make_unmodified(symbols)

        if relaxed_domain_mapping:
            # Best attempt approach to map relaxed domain to actual symbols
            for name in symbols:
                symbol = self.container[name]

                new_domain = []
                for elem in symbol.domain:
                    if type(elem) is str and elem != "*" and elem in self.container:
                        new_domain.append(self.container[elem])
                    else:
                        new_domain.append(elem)

                symbol.domain = new_domain
                symbol.dimension = len(new_domain)
                if type(symbol) in (syms.Variable, syms.Equation):
                    symbol._update_attr_domains()

    def write_trace_template(self) -> None:
        # Custom trace file template.
        if os.path.isfile(self.trace_file):
            return

        lines = [
            "* Trace Record Definition",
            "* GamsSolve",
            "* SolverName NumberOfDomainViolations ETAlg ETSolve ETSolver NumberOfIterations Marginals ModelStatus NumberOfNodes SolveNumber NumberOfDiscreteVariables NumberOfEquations NumberOfNonlinearNonZeros NumberOfNonZeros NumberOfVariables ",
            "* ObjectiveValueEstimate ObjectiveValue ModelType ModelGenerationTime SolverTime SolverStatus SolverVersion",
        ]
        with open(self.trace_file, "w", encoding="utf-8") as file:
            file.write("\n".join(lines) + "\n")

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
        from gamspy._model import ModelStatus, SolveStatus

        with open(trace_file, encoding="utf-8") as file:
            # We only need the last line. deque provides functionality similar to the tail filter in Unix.
            line = deque(file, maxlen=1)[0]
            (
                solver_name,
                num_domain_violations,
                algorithm_time,
                total_solve_time,
                total_solver_time,
                num_iterations,
                marginals,
                model_status,
                num_nodes,
                solve_number,
                num_discrete_variables,
                num_equations,
                num_nonlinear_zeros,
                num_nonzeros,
                num_variables,
                objective_estimation,
                objective_value,
                used_model_type,
                model_generation_time,
                solve_model_time,
                solve_status,
                solver_version,
            ) = line.split(" ")

        assert self.model is not None
        self.model._num_domain_violations = _cast_value(num_domain_violations, int)
        self.model._algorithm_time = _cast_value(algorithm_time, float)
        self.model._total_solve_time = _cast_value(total_solve_time, float)
        self.model._total_solver_time = _cast_value(total_solver_time, float)
        self.model._num_iterations = _cast_value(num_iterations, int)
        self.model._marginals = _cast_value(marginals, int)
        self.model._status = ModelStatus(int(model_status))
        self.model._num_nodes_used = _cast_value(num_nodes, int)
        self.model._solve_number = _cast_value(solve_number, int)
        self.model._num_discrete_variables = _cast_value(num_discrete_variables, int)
        self.model._num_equations = _cast_value(num_equations, int)
        self.model._num_nonlinear_zeros = _cast_value(num_nonlinear_zeros, int)
        self.model._num_nonzeros = _cast_value(num_nonzeros, int)
        self.model._num_variables = _cast_value(num_variables, int)
        self.model._objective_estimation = _cast_value(objective_estimation, float)
        self.model._objective_value = _cast_value(objective_value, float)
        self.model._used_model_type = used_model_type
        self.model._model_generation_time = _cast_value(model_generation_time, float)
        self.model._solve_model_time = _cast_value(solve_model_time, float)
        self.model.solve_status = SolveStatus(int(solve_status))
        self.model._solver_version = _cast_value(solver_version.strip(), int)

        dataframe = pd.DataFrame(
            [
                [
                    SOLVE_STATUS[int(solve_status)],
                    self.model._status.name,
                    self.model._objective_value,
                    self.model._num_equations,
                    self.model._num_variables,
                    used_model_type,
                    solver_name,
                    self.model._total_solver_time,
                ]
            ],
            columns=HEADER,
        )
        return dataframe

    def make_unmodified(self, modified_names: list[str]):
        for name in modified_names:
            if not name.startswith("autogenerated_"):
                self.container[name].modified = False
