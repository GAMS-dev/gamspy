from __future__ import annotations

import io
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

from gams import SymbolUpdateType
from pydantic import BaseModel, ConfigDict

from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
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
    "monitor_process_tree_memory": "procTreeMemMonitor",
    "memory_tick_interval": "procTreeMemTicks",
    "profile": "profile",
    "profile_file": "profileFile",
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
    """
    Options class to set GAMS options for the model.

    Attributes
    ----------
    
    cns: str | None
        Default **cns** solver
    dnlp: str | None
        Default **dnlp** solver
    emp: str | None
        Default **emp** solver
    lp: str | None
        Default **lp** solver
    mcp: str | None
        Default **mcp** solver
    minlp: str | None
        Default **minlp** solver
    mip: str | None
        Default **mip** solver
    miqcp: str | None
        Default **miqcp** solver
    mpec: str | None
        Default **mpec** solver
    nlp: str | None
        Default **nlp** solver
    qcp: str | None
        Default **qcp** solver
    rminlp: str | None
        Default **rminlp** solver
    rmip: str | None
        Default **rmip** solver
    rmiqcp: str | None
        Default **rmiqcp** solver
    rmpec: str | None
        Default **rmpec** solver
    allow_suffix_in_equation: bool | None
        Flag to allow variables with suffixes in model algebra
    allow_suffix_in_limited_variables: bool | None
        Flag to allow **domain limited variables** with suffixes in model
    basis_detection_threshold: float | None
        Basis detection threshold
    compile_error_limit: int = 1
        Compile time error limit
    domain_violation_limit: int | None
        Domain violation limit
    job_time_limit: float | None
        Elapsed time limit (seconds)
    job_heap_limit: float | None
        Maximum Heap size allowed (MB)
    hold_fixed_variables: bool | None
        Treat fixed variables as constants
    integer_variable_upper_bound: int | None
        Set mode for default upper bounds on integer variables
    iteration_limit: int | None
        Iteration limit of solver
    keep_temporary_files: bool = False
        Controls keeping or deletion of process directory and scratch files
    listing_file: str | None
        Listing file name
    log_file: str | None
        Log file name
    variable_listing_limit: int | None
        Maximum number of columns listed in one variable block
    equation_listing_limit: int | None
        Maximum number of rows listed in one equation block
    node_limit: int | None
        Node limit in branch and bound tree
    absolute_optimality_gap: float | None
        Absolute Optimality criterion solver default
    relative_optimality_gap: float | None
        Relative Optimality criterion solver default
    memory_tick_interval: float | None
        Wait interval between memory monitor checks: ticks = milliseconds
    monitor_process_tree_memory: bool | None
        Monitor the memory used by the GAMS process tree
    profile: int | None
        Execution profiling
    profile_file: str
        Write profile information to this file
    profile_tolerance: float | None
        Minimum time a statement must use to appear in profile generated output
    redirect_log_to_stdout: Optional[bool] = False
        description
    time_limit: float | None
        Wall-clock time limit for solver
    savepoint: Optional[Literal[0, 1, 2, 3, 4]] = None
        Save solver point in GDX file
        
        * **Option 0**: No point GDX file is to be saved
        
        * **Option 1**: A point GDX file from the last solve is to be saved
        
        * **Option 2**: A point GDX file from every solve is to be saved
        
        * **Option 3**: A point GDX file from the last solve is to be saved in the scratch directory
        
        * **Option 4**: A point GDX file from every solve is to be saved in the scratch directory
    seed: int | None
        Random number seed
    report_solution: Literal[0, 1, 2] = 2
        Solution report print option

        * **Option 0**: Remove solution listings following solves

        * **Option 1**: Include solution listings following solves

        * **Option 2**: Suppress all solution information
    show_os_memory: Literal[0, 1, 2] = 0
        Show the memory usage reported by the Operating System instead of the internal counting
        
        * **Option 0**: Show memory reported by internal accounting
        
        * **Option 1**: Show resident set size reported by operating system
        
        * **Option 2**: Show virtual set size reported by operating system
    solver_link_type: Optional[Literal[0, 1, 2, 3, 4, 5, 6, 7]] = None
        Solver link option
        
        * **Option 0**: Model instance and entire GAMS state saved to scratch directory, GAMS exits (and vacates memory), and the solver script is called. After the solver terminates, GAMS restarts from the saved state and continues to executing
        
        * **Option 1**: Model instance saved to scratch directory, the solver is called from a shell while GAMS remains open
        
        * **Option 2**: Model instance saved to scratch directory, the solver is called with a spawn (if possible) or a shell (if spawn is not possible) while GAMS remains open - If this is not supported by the selected solver, it gets reset to **1** automatically
        
        * **Option 3**: Model instance saved to scratch directory, the solver starts the solution and GAMS continues
        
        * **Option 4**: Model instance saved to scratch directory, the solver starts the solution and GAMS waits for the solver to come back but uses same submission process as **3** (test mode)
        
        * **Option 5**: The model instance is passed to the solver in-memory - If this is not supported by the selected solver, it gets reset to **2** automatically
        
        * **Option 6**: The model instance is passed to the solver in-memory, the solver starts the solution and GAMS continues
        
        * **Option 7**: The model instance is passed to the solver in-memory, the solver starts the solution and GAMS waits for the solver to come back but uses same submission process as **6** (test mode)
    merge_strategy: Optional[Literal["replace", "merge", "clear"]] = None
        * **Replace**: The solution information for all equations and variables is merged into the existing solution information
        
        * **Merge**: The solution information for all equations appearing in the model is completely replaced by the new model results; variables are only replaced if they appear in the final model
        
        * **Clear**: The solution information for all equations appearing in the model is completely replaced; in addition, variables appearing in the symbolic equations but removed by conditionals will be removed
    step_summary: bool | None
        Summary of computing resources used by job steps
    suppress_compiler_listing: bool = False
        Compiler listing option
    report_solver_status: bool | None
        Solver Status file reporting option
    threads: int | None
        Number of processors to be used by a solver
    write_listing_file: bool = True
        Switch to write a Listing file
    zero_rounding_threshold: float | None
        The results of certain operations will be set to zero if abs(result) LE ZeroRes
    report_underflow: bool | None
        Report underflow as a warning when abs(results) LE ZeroRes and result set to zero
    """
    model_config = ConfigDict(extra="forbid")
    
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
    monitor_process_tree_memory: Optional[bool] = None
    memory_tick_interval: Optional[float] = None
    profile: Optional[int] = None
    profile_file: Optional[str] = None
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

    def _get_gams_compatible_options(self, output: io.TextIOWrapper | None = None) -> dict:
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

        if "log_file" in gamspy_options:
            os.makedirs(
                Path(gamspy_options["log_file"]).parent.absolute(),
                exist_ok=True,
            )

        gams_options = dict()
        for key, value in gamspy_options.items():
            value = int(value) if isinstance(value, bool) else value
            gams_options[option_map[key]] = value

        gams_options["previouswork"] = (
            1  # # In case GAMS version differs on backend
        )
        gams_options["traceopt"] = 3
        gams_options["gdxSymbols"] = "newOrChanged"

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

            solver_file_name = os.path.join(working_directory, f"{solver.lower()}.123")

            with open(solver_file_name, "w", encoding="utf-8") as solver_file:
                for key, value in solver_options.items():
                    solver_file.write(f"{key} {value}\n")

            self._solver_options_file = "123"

    def _set_extra_options(self, options: dict) -> None:
        """Set extra options of the backend"""
        self._extra_options = options

    @staticmethod
    def from_file(path: str) -> Options:
        """
        Generates an Options object with the key-value pairs in a file.
        The file in given path must consist of one key-value pair in each line.

        Parameters
        ----------
        path : str
            Path to the option file.

        Returns
        -------
        Options

        Raises
        ------
        ValidationError
            In case the given path is not a file.
        """
        if not os.path.isfile(path):
            raise ValidationError(f"No such file in the given path: {path}")
        
        attributes = dict()
        with open(path, encoding="utf-8") as file:
            lines = file.readlines()

            for line in lines:
                if line == "\n" or line == "":
                    continue

                key, value = line.split("=")
                attributes[key.strip()] = value.strip()
        
        return Options(**attributes)

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
        with open(pf_file, "w", encoding="utf-8") as file:
            file.write("\n".join([f"{key} = {value}" for key, value in all_options.items()]))


update_type_map = {
    "0": SymbolUpdateType.Zero,
    "base_case": SymbolUpdateType.BaseCase,
    "accumulate": SymbolUpdateType.Accumulate,
    "inherit": SymbolUpdateType._Inherit,
}


class ModelInstanceOptions(BaseModel):
    no_match_limit: int = 0
    debug: bool = False
    update_type: Literal["0", "base_case", "accumulate", "inherit"] = (
        "base_case"
    )

    def items(self):
        dictionary = self.model_dump()
        dictionary["update_type"] = update_type_map[dictionary["update_type"]]

        return dictionary.items()
