from __future__ import annotations

import io
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Union
import warnings

from pydantic import BaseModel, ConfigDict

from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from gamspy import Container
    from gamspy._model import Problem
    from types import FrameType

SOLVE_LINK_MAP = {"disk": 2, "memory": 5}
SOLVE_LINK_MAP_REVERSE = dict(zip(SOLVE_LINK_MAP.values(), SOLVE_LINK_MAP.keys()))

# GAMSPy to GAMS option mapping
OPTION_MAP = {
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
    "license": "license",
    "allow_suffix_in_equation": "suffixalgebravars",
    "allow_suffix_in_limited_variables": "suffixdlvars",
    "append_to_log_file": "appendLog",
    "basis_detection_threshold": "bratio",
    "compile_error_limit": "cerr",
    "domain_violation_limit": "domlim",
    "hold_fixed_variables": "holdfixed",
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
    "reference_file": "reference",
    "time_limit": "reslim",
    "savepoint": "savepoint",
    "seed": "seed",
    "report_solution": "solprint",
    "show_os_memory": "showosmemory",
    "solve_link_type": "solvelink",
    "merge_strategy": "solveopt",
    "step_summary": "stepsum",
    "suppress_compiler_listing": "suppress",
    "report_solver_status": "sysout",
    "threads": "threads",
    "write_listing_file": "writeoutput",
    "zero_rounding_threshold": "zerores",
    "report_underflow": "zeroresrep",
}
OPTION_MAP_REVERSE = dict(zip(OPTION_MAP.values(), OPTION_MAP.keys()))

MODEL_ATTR_OPTION_MAP = {
    "generate_name_dict": "dictfile",
    "enable_scaling": "scaleopt",
    "min_improvement_threshold": "cheat",
    "cutoff": "cutOff",
    "default_point": "defPoint",
    "enable_prior": "priorOpt",
    "infeasibility_tolerance": "tolInfRep",
    "try_partial_integer_solution": "tryInt",
    "examine_linearity": "tryLinear",
    "bypass_solver": "justscrdir",
}
EXECUTION_OPTIONS = {"loadpoint": "execute_loadpoint"}


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
    append_to_log_file: bool | None
        Setting this option to True means that the log file will be appended to and not overwritten (replaced).
    basis_detection_threshold: float | None
        Basis detection threshold
    compile_error_limit: int = 1
        Compile time error limit
    domain_violation_limit: int | None
        Domain violation limit
    cutoff: float | None
        Within a branch-and-bound based solver, the parts of the tree with an objective value worse than
        the cutoff value are ignored. Note that this may speed up the initial phase of the branch and
        bound algorithm (before the first integer solution is found). However, the true optimum may be
        beyond the cutoff value. In this case the true optimum will be missed and moreover, no solution
        will be found. Observe that this option is specified in absolute terms.
    default_point: int | None
        This option determines the point that is passed to the solver as a basis. By default, the levels
        and marginals from the current basis are passed to the solver. In some circumstances (mostly during debugging),
        it can be useful to pass a standard default input point, i.e. with all levels set to 0 or lower bound.

        * **Option 0**	Pass user defined levels and marginals to solver

        * **Option 1**	Pass default levels and marginals to solver

        * **Option 2**	Pass default marginals to solver
    generate_name_dict: bool | None
        If this option is set, it will instruct GAMS to make the GAMS names
        of variables and equations that have been generated by the solve statement
        available to the solver. In many solver links, these names are registered
        with the solver and hence messages from the solver that involve variables
        and equations (e.g. an infeasible row or duplicate columns) can be easily
        interpreted by the user. However, the dictionary comes at a price.
        Generating the names and calculating and storing the map takes time and space.
        In addition, GAMS names take up space in the solver. Thus, if the user needs
        very fast generation and does not need names, setting dictFile to zero is a
        good option.
    enable_scaling: bool | None
        This option determines whether GAMS will employ user-specified variable and equation scaling factors.
        It must be set to True if scaling factors are to be used.
    enable_prior: bool | None
        Instructs the solver to use the priority branching information passed by GAMS through variable suffix
        values variable.prior. If and how priorities are used is solver-dependent.
    infeasibility_tolerance: float | None
        This option sets the tolerance for marking an equation infeasible in the equation listing. By default,
        1.0e-13.
    try_partial_integer_solution: bool | None
        Signals the solver to make use of a partial or near-integer-feasible solution stored in current variable
        values to get a quick integer-feasible point. The exact form of implementation depends on the solver and
        may be partly controlled by solver settings or options. See the solver manuals for details.
    examine_linearity: bool | None
        Examine empirical NLP model to see if there are any NLP terms active. If there are none the default LP
        solver will be used. If this option is set to True, empirical NLP models will be examined to determine
        if there are any active NLP terms. If there are none, the default LP solver will be used. The procedure
        also checks to see if QCP and DNLP models can be reduced to an LP; MIQCP and MINLP can be solved as a MIP;
        RMIQCP and RMINLP can be solved as an RMIP.
    bypass_solver: bool | None
        If True, GAMSPy does not pass the generated model to the solver. Useful for model generation time analysis.
    hold_fixed_variables: bool | None
        Treat fixed variables as constants
    iteration_limit: int | None
        Iteration limit of solver
    keep_temporary_files: bool = False
        Controls keeping or deletion of process directory and scratch files
    license: str | None
        Absolute path of the license.
    listing_file: str | None
        Listing file name
    loadpoint: os.PathLike | str | None
        Path to the loadpoint GDX file that contains starting point records.
    log_file: str | None
        Log file name
    variable_listing_limit: int
        Maximum number of columns listed in one variable block
    equation_listing_limit: int
        Maximum number of rows listed in one equation block
    min_improvement_threshold: float | None
        For a branch-and-bound based solver, each new feasible solution must be at least the value of min_improvement_threshold
        better than the current best feasible solution. Note that this may speed up the search, but may cause some solutions,
        including optimal ones, to be missed. If a model has been solved with a nonzero min_improvement_threshold, then the
        optimal solution will be within the min_improvement_threshold or less of the found solution. Observe that the option
        min_improvement_threshold is specified in absolute terms, therefore non-negative values are appropriate for both
        minimization and maximization models.
    cutoff: float | None
        Within a branch-and-bound based solver, the parts of the tree with an objective value worse than the cutoff value are
        ignored. Note that this may speed up the initial phase of the branch and bound algorithm (before the first integer
        solution is found). However, the true optimum may be beyond the cutoff value. In this case the true optimum will be
        missed and moreover, no solution will be found.
    miro_protect:
        Protects MIRO input symbol records from being re-assigned, by default True.
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

        * **Option 0**: Do not profile.

        * **Option 1**: Minimum profiling.

        * **Option n**: Profiling depth for nested control structures.
    profile_file: str
        Write profile information to this file
    profile_tolerance: float | None
        Minimum time a statement must use to appear in profile generated output
    reference_file: str | None
        Symbol reference file
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
    solve_link_type: Optional[Literal["disk", "memory"]] = None
        Solver link option

        * **disk**: Model instance saved to scratch directory, the solver is called with a spawn (if possible) or a shell (if spawn is not possible) while GAMS remains open.

        * **memory**: The model instance is passed to the solver in-memory - If this is not supported by the selected solver, it gets reset to **disk** automatically.
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
    append_to_log_file: Optional[bool] = None
    basis_detection_threshold: Optional[float] = None
    compile_error_limit: Optional[int] = None
    domain_violation_limit: Optional[int] = None
    generate_name_dict: Optional[bool] = None
    enable_scaling: Optional[bool] = None
    enable_prior: Optional[bool] = None
    infeasibility_tolerance: Optional[float] = None
    try_partial_integer_solution: Optional[bool] = None
    examine_linearity: Optional[bool] = None
    bypass_solver: Optional[bool] = None
    min_improvement_threshold: Optional[float] = None
    cutoff: Optional[float] = None
    default_point: Optional[int] = None
    miro_protect: bool = True
    hold_fixed_variables: Optional[bool] = None
    iteration_limit: Optional[int] = None
    keep_temporary_files: Optional[int] = None
    license: Optional[str] = None
    listing_file: Optional[str] = None
    loadpoint: Optional[Union[str, os.PathLike]] = None
    log_file: Optional[str] = None
    variable_listing_limit: int = 0
    equation_listing_limit: int = 0
    node_limit: Optional[int] = None
    absolute_optimality_gap: Optional[float] = None
    relative_optimality_gap: Optional[float] = None
    monitor_process_tree_memory: Optional[bool] = None
    memory_tick_interval: Optional[float] = None
    profile: Optional[int] = None
    profile_file: Optional[str] = None
    profile_tolerance: Optional[float] = None
    reference_file: Optional[str] = None
    time_limit: Optional[float] = None
    savepoint: Optional[Literal[0, 1, 2, 3, 4]] = None
    seed: Optional[int] = None
    report_solution: Literal[0, 1, 2] = 0
    show_os_memory: Optional[Literal[0, 1, 2]] = None
    solve_link_type: Optional[Literal["disk", "memory"]] = None
    merge_strategy: Optional[Literal["replace", "merge", "clear"]] = None
    step_summary: Optional[bool] = None
    suppress_compiler_listing: Optional[bool] = None
    report_solver_status: Optional[bool] = None
    threads: Optional[int] = None
    write_listing_file: Optional[bool] = None
    zero_rounding_threshold: Optional[float] = None
    report_underflow: Optional[bool] = None

    def model_post_init(self, context: Any) -> None:
        self._extra_options: dict[str, Any] = dict()
        self._debug_options: dict[str, Any] = dict()
        self._solver: str | None = None
        self._problem: str | None = None
        self._solver_options_file: str = "0"
        self._frame: FrameType | None = None

    @staticmethod
    def fromGams(options: dict) -> Options:
        """
        Generates a gp.Options object from a dictionary of GAMS options 
        where keys are the GAMS option names.

        Parameters
        ----------
        options : dict
            GAMS options.

        Returns
        -------
        Options
            Generated GAMSPy options

        Raises
        ------
        ValidationError
            In case the option is not supported in GAMSPy.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> options = gp.Options.fromGams({"reslim": 5})

        """
        gamspy_options = {}
        for key, value in options.items():
            if key.lower() in OPTION_MAP_REVERSE:
                if key.lower() == "solvelink":
                    try:
                        value = SOLVE_LINK_MAP_REVERSE[value]
                    except KeyError:
                        raise ValidationError(f"`{value}` is not a valid value for `{key}`. Possible values are 2 and 5.")

                gamspy_options[OPTION_MAP_REVERSE[key.lower()]] = value
            else:
                raise ValidationError(f"`{key}` is not a supported option in GAMSPy.")

        return Options(**gamspy_options) 

    def _get_gams_compatible_options(
        self, output: io.TextIOWrapper | None = None
    ) -> dict:
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

        if "solve_link_type" in gamspy_options:
            link_type = gamspy_options["solve_link_type"]
            gamspy_options["solve_link_type"] = SOLVE_LINK_MAP[link_type]

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
            if key not in OPTION_MAP:
                continue

            value = int(value) if isinstance(value, bool) else value
            gams_options[OPTION_MAP[key]] = value

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
        container: Container,
        solver: str,
        problem: Problem,
        solver_options: dict | None,
    ):
        """Set the solver and the solver options"""
        self._solver = solver
        self._problem = str(problem)

        if solver_options:
            container.writeSolverOptions(solver, solver_options)
            self._solver_options_file = "1"
        else:
            self._solver_options_file = "0"

    def _set_extra_options(self, options: dict) -> None:
        """Set extra options of the backend"""
        self._extra_options = options

    def _set_debug_options(self, options: dict) -> None:
        """Set debugging options"""
        self._debug_options = options

    @staticmethod
    def fromFile(path: str) -> Options:
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
                    continue  # pragma: no cover

                key, value = line.split("=")
                attributes[key.strip()] = value.strip()

        return Options(**attributes)

    def export(self, pf_file: str) -> None:
        """
        Exports options to the pf_file. Each line contains a key-value pair.

        Parameters
        ----------
        pf_file : str
        """
        self._export(pf_file)

    def _export(
        self, pf_file: str, output: io.TextIOWrapper | None = None
    ) -> None:
        """
        Exports options to the pf_file. Each line contains a key-value pair.

        Parameters
        ----------
        pf_file : str
        output : io.TextIOWrapper | None, optional
        """
        all_options = dict()
        # Solver options
        if self._solver is not None:
            all_options[self._problem] = self._solver

        all_options["optfile"] = self._solver_options_file

        # Extra options
        all_options.update(**self._extra_options)
        all_options.update(**self._debug_options)

        if self._frame is not None:
            filename = self._frame.f_code.co_filename
            line_number = self._frame.f_lineno
            all_options["GP_SolveLine"] = f"{filename} line {line_number}"

        # User options
        user_options = self._get_gams_compatible_options(output)
        all_options.update(**user_options)

        # Generate pf file
        with open(pf_file, "w", encoding="utf-8") as file:
            file.write(
                "\n".join(
                    [
                        f'{key} = "{value}"'
                        for key, value in all_options.items()
                    ]
                )
            )

class FreezeOptions(BaseModel):
    no_match_limit: int = 0
    debug: bool = False
    update_type: Literal["0", "base_case", "accumulate", "inherit"] = (
        "base_case"
    )

class ModelInstanceOptions(BaseModel):
    no_match_limit: int = 0
    debug: bool = False
    update_type: Literal["0", "base_case", "accumulate", "inherit"] = (
        "base_case"
    )

    def __init__(self, **kwargs):
        warnings.warn(
            "ModelInstanceOptions will be renamed to FreezeOptions in GAMSPy 1.9.0. Please use FreezeOptions instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**kwargs)

class ConvertOptions(BaseModel):
    """
    Options for the conversion of GAMSPy models into different formats.

    Attributes
    ----------
    AmplNLBin : Optional[bool]
        Enables binary .nl file. False by default.
    AmplNlInitDual : Optional[int]
        Specifies which initial equation marginal values to write to the .nl file.
        
        * **0**: Write no values 
        
        * **1**: Write only nondefault values
        
        * **2**: Write all values.

        1 by default.
    AmplNlInitPrimal : Optional[int]
        Specifies which initial primal values to write to the .nl file.
        
        * **0**: Write no values
        
        * **1**: Write only nondefault values
        
        * **2**: Write all values.

        2 by default.
    GDXHessian : Optional[bool]
        Controls whether Hessian information is included in GDX Jacobian file.
        False by default.
    GDXNames : Optional[bool]
        Controls whether variable and equation names are included in GDX Jacobian file.
        True by default.
    GDXQuadratic : Optional[bool]
        Specifies whether quadratic terms are included in GDX Jacobian file.
        False by default.
    GDXUELs : Optional[bool]
        Controls whether Universal Element List (UEL) information is included in GDX Jacobian file.
        True by default.
    GAMSInsert : Optional[str]
        Allows the insertion of custom GAMS code into the model.
    HeaderTimeStamp : Optional[bool]
        Specifies a timestamp to include in the header of the output file.
    GDXIntervalEval : Optional[bool]
        Controls the inclusion of interval evaluation (symbols `A_int` and `e_int`) into the GDX Jacobian format.
        False by default.
    GAMSObjVar : Optional[str]
        Specifies the name of the objective variable in the GAMS scalar model.
    PermuteEqus : Optional[bool]
        Enables or disables the permutation of equations.
        False by default.
    PermuteVars : Optional[bool]
        Enables or disables the permutation of variables.
        False by default.
    QExtractAlg : Optional[int]
        Specifies the algorithm used for quadratic extraction.
        
        * **0**: Automatic

        * **1**: ThreePass: Uses a three-pass forward / backward / forward AD technique to compute function / gradient / Hessian values and a hybrid scheme for storage.

        * **2**: DoubleForward: Uses forward-mode AD to compute and store function, gradient, and Hessian values at each node or stack level as required. The gradients and Hessians are stored in linked lists.

        * **3**: Concurrent: Uses ThreePass and DoubleForward in parallel. As soon as one finishes, the other one stops.

        0 by default.
    Reform : Optional[int]
        Controls the reformulation of certain structures in the model.
        0: No reformulation, 1: Apply reformulation.
    SkipNRows : Optional[int]
        Skip constraints of type `NONBINDING`.
    Width : Optional[int]
        Sets the width for certain output formats, such as tables or reports.
    """
    
    model_config = ConfigDict(extra="forbid")

    AmplNLBin: Optional[bool] = None
    AmplNlInitDual: Optional[int] = None
    AmplNlInitPrimal: Optional[int] = None
    GDXHessian: Optional[bool] = None
    GDXNames: Optional[bool] = None
    GDXQuadratic: Optional[bool] = None
    GDXUELs: Optional[bool] = None
    GAMSInsert: Optional[str] = None
    HeaderTimeStamp: Optional[bool] = None
    GDXIntervalEval: Optional[bool] = None
    GAMSObjVar: Optional[str] = None
    PermuteEqus: Optional[bool] = None
    PermuteVars: Optional[bool] = None
    QExtractAlg: Optional[int] = None
    Reform: Optional[int] = None
    SkipNRows: Optional[int] = None
    Width: Optional[int] = None
