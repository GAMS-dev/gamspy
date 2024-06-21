from __future__ import annotations

import io
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Literal, Optional

from gams import SymbolUpdateType, GamsOptions
from pydantic import BaseModel

from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from gams import GamsWorkspace
    from gamspy._model import Problem

multi_solve_map = {"replace": 0, "merge": 1, "clear": 2}

# GAMSPy to GAMS Control mapping
option_map = {
    "cns": "cns",
    "dnlp": "dnlp",
    "emp": "emp",
    "lp": "lp",
    "mcp": "mcp",
    "minlp": "minlp",
    "mip": "mip",
    "miqcp": "miqcp",
    "mpec": "mpec",
    "nlp": "nlp",
    "qcp": "qcp",
    "rminlp": "rminlp",
    "rmip": "rmip",
    "rmiqcp": "rmiqcp",
    "rmpec": "rmpec",
    "allow_suffix_in_equation": "suffixalgebravars",
    "allow_suffix_in_limited_variables": "suffixdlvars",
    "basis_detection_threshold": "bratio",
    "compile_error_limit": "cerr",
    "domain_violation_limit": "domlim",
    "job_time_limit": "etlim",
    "job_heap_limit": "heaplimit",
    "hold_fixed_variables": "holdfixed",
    "integer_variable_upper_bound": "intvarup",
    "iteration_limit": "iterlim",
    "keep_temporary_files": "keep",
    "listing_file": "output",
    "log_file": "logfile",
    "variable_listing_limit": "limcol",
    "equation_listing_limit": "limrow",
    "node_limit": "nodlim",
    "absolute_optimality_gap": "optca",
    "relative_optimality_gap": "optcr",
    "profile": "profile",
    "profile_tolerance": "profiletol",
    "time_limit": "reslim",
    "savepoint": "savepoint",
    "seed": "seed",
    "report_solution": "solprint",
    "show_os_memory": "showosmemory",
    "solver_link_type": "solvelink",
    "merge_strategy": "solveopt",
    "step_summary": "stepsum",
    "suppress_compiler_listing": "suppress",
    "report_solver_status": "sysout",
    "threads": "threads",
    "write_listing_file": "writeoutput",
    "zero_rounding_threshold": "zerores",
    "report_underflow": "zeroresrep",
}


class Options(BaseModel):
    cns: Optional[str] = None
    dnlp: Optional[str] = None
    emp: Optional[str] = None
    lp: Optional[str] = None
    mcp: Optional[str] = None
    minlp: Optional[str] = None
    mip: Optional[str] = None
    miqcp: Optional[str] = None
    mpec: Optional[str] = None
    nlp: Optional[str] = None
    qcp: Optional[str] = None
    rminlp: Optional[str] = None
    rmip: Optional[str] = None
    rmiqcp: Optional[str] = None
    rmpec: Optional[str] = None
    allow_suffix_in_equation: Optional[bool] = None
    allow_suffix_in_limited_variables: Optional[bool] = None
    basis_detection_threshold: Optional[float] = None
    compile_error_limit: Optional[int] = None
    domain_violation_limit: Optional[int] = None
    job_time_limit: Optional[float] = None
    job_heap_limit: Optional[float] = None
    hold_fixed_variables: Optional[bool] = None
    integer_variable_upper_bound: Optional[int] = None
    iteration_limit: Optional[int] = None
    keep_temporary_files: Optional[int] = None
    listing_file: Optional[str] = None
    log_file: Optional[str] = None
    variable_listing_limit: Optional[int] = None
    equation_listing_limit: Optional[int] = None
    node_limit: Optional[int] = None
    absolute_optimality_gap: Optional[float] = None
    relative_optimality_gap: Optional[float] = None
    profile: Optional[int] = None
    profile_tolerance: Optional[float] = None
    time_limit: Optional[float] = None
    savepoint: Optional[Literal[0, 1, 2, 3, 4]] = None
    seed: Optional[int] = None
    report_solution: Optional[Literal[0, 1, 2]] = None
    show_os_memory: Optional[Literal[0, 1, 2]] = None
    solver_link_type: Optional[Literal[0, 1, 2, 3, 4, 5, 6, 7]] = None
    merge_strategy: Optional[Literal["replace", "merge", "clear"]] = None
    step_summary: Optional[bool] = None
    suppress_compiler_listing: Optional[bool] = None
    report_solver_status: Optional[bool] = None
    threads: Optional[int] = None
    write_listing_file: Optional[bool] = None
    zero_rounding_threshold: Optional[float] = None
    report_underflow: Optional[bool] = None

    def _get_gams_compatible_options(self, output: io.TextIOWrapper | None) -> dict:
        gamspy_options = self.model_dump(exclude_none=True)
        if "allow_suffix_in_equation" in gamspy_options:
            allows_suffix = gamspy_options["allow_suffix_in_equation"]
            gamspy_options["allow_suffix_in_equation"] = (
                "on" if allows_suffix else "off"
            )

        if "allow_suffix_in_limited_variables" in gamspy_options:
            allows_suffix = gamspy_options["allow_suffix_in_limited_variables"]
            gamspy_options["allow_suffix_in_limited_variables"] = (
                "on" if allows_suffix else "off"
            )

        if "merge_strategy" in gamspy_options:
            strategy = gamspy_options["merge_strategy"]
            gamspy_options["merge_strategy"] = multi_solve_map[strategy]

        if "listing_file" in gamspy_options:
            os.makedirs(
                Path(gamspy_options["listing_file"]).parent.absolute(),
                exist_ok=True,
            )
            if not os.path.isabs(gamspy_options["listing_file"]):
                gamspy_options["listing_file"] = os.path.abspath(
                    gamspy_options["listing_file"]
                )

        if "log_file" in gamspy_options:
            os.makedirs(
                Path(gamspy_options["log_file"]).parent.absolute(),
                exist_ok=True,
            )
            if not os.path.isabs(gamspy_options["log_file"]):
                gamspy_options["log_file"] = os.path.abspath(
                    gamspy_options["log_file"]
                )

        gams_options = dict()
        for key, value in gamspy_options.items():
            value = int(value) if isinstance(value, bool) else value
            gams_options[option_map[key]] = value

        gams_options["previouswork"] = (
            1  # # In case GAMS version differs on backend
        )
        gams_options["traceopt"] = 3

        if self.log_file:
            if output is not None:
                gams_options["logoption"] = 4
            else:
                gams_options["logoption"] = 2
        else:
            if output is not None:
                gams_options["logoption"] = 3
            else:
                gams_options["logoption"] = 0

        return gams_options

    def _set_solver_options(
        self,
        working_directory: str,
        solver: str | None,
        problem: Problem,
        solver_options: dict | None,
    ):
        """Set the solver and the solver options"""
        if solver:
            self._solver = (str(problem), solver)

        if solver_options:
            if solver is None:
                raise ValidationError(
                    "You need to provide a 'solver' to apply solver options."
                )

            solver_file_name = os.path.join(
                working_directory, f"{solver.lower()}.123"
            )

            with open(solver_file_name, "w", encoding="utf-8") as solver_file:
                for key, value in solver_options.items():
                    solver_file.write(f"{key} {value}\n")

            self._solver_options_file = "123"

    def _set_extra_options(self, options: dict) -> None:
        """Set extra options of the backend"""
        self._extra_options = options

    def export(self, pf_file: str, output: io.TextIOWrapper | None = None) -> None:
        all_options = dict()
        # Solver options
        if hasattr(self, "_solver"):
            problem_type, solver = self._solver
            all_options[problem_type] = solver

        if hasattr(self, "_solver_options_file"):
            all_options["optfile"] = self._solver_options_file

        # Extra options
        if hasattr(self, "_extra_options") and self._extra_options:
            all_options.update(**self._extra_options)

        # User options
        user_options = self._get_gams_compatible_options(output)
        all_options.update(**user_options)

        # Generate pf file
        with open(pf_file, "w") as file:
            file.write("\n".join([f"{key} = {value}" for key, value in all_options.items()]))

    def _get_gams_options(self, workspace: GamsWorkspace, output: io.TextIOWrapper | None = None) -> GamsOptions:
        temp_path = os.path.join(workspace.working_directory, "dummy.pf")
        self.export(temp_path, output)
        gams_options = GamsOptions(workspace, opt_file=temp_path)

        return gams_options


update_type_map = {
    "0": SymbolUpdateType.Zero,
    "base_case": SymbolUpdateType.BaseCase,
    "accumulate": SymbolUpdateType.Accumulate,
    "inherit": SymbolUpdateType._Inherit,
}


class ModelInstanceOptions(BaseModel):
    solver: Optional[str] = None
    opt_file: int = -1
    no_match_limit: int = 0
    debug: bool = False
    update_type: Literal["0", "base_case", "accumulate", "inherit"] = (
        "base_case"
    )

    def items(self):
        dictionary = self.model_dump()
        dictionary["update_type"] = update_type_map[dictionary["update_type"]]

        return dictionary
