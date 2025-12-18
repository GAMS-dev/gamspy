from __future__ import annotations

import inspect
import logging
import os
import threading
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from gams.core.gdx import GMS_UEL_IDENT_SIZE

import gamspy as gp
import gamspy._algebra.expression as expression
import gamspy._algebra.operation as operation
import gamspy._miro as miro
import gamspy._symbols.implicits as implicits
import gamspy._validation as validation
import gamspy.utils as utils
from gamspy._backend.backend import backend_factory
from gamspy._convert import (
    GamsConverter,
    LatexConverter,
    get_convert_solver_options,
)
from gamspy._model_instance import ModelInstance
from gamspy._options import (
    EXECUTION_OPTIONS,
    MODEL_ATTR_OPTION_MAP,
    ConvertOptions,
    FreezeOptions,
    Options,
)
from gamspy.exceptions import GamspyException, ValidationError

if TYPE_CHECKING:
    import io
    from typing import Literal

    import pandas as pd

    from gamspy import Container, Equation, Parameter, Variable
    from gamspy._algebra.expression import Expression
    from gamspy._algebra.operation import Operation
    from gamspy._backend.engine import EngineClient
    from gamspy._backend.neos import NeosClient
    from gamspy._symbols.implicits import ImplicitParameter, ImplicitVariable
    from gamspy._symbols.symbol import Symbol
    from gamspy.math import MathOp

GMS_MAX_LINE_LENGTH = 80000
MAX_MODEL_DECLARATION_LENGTH = 75
MAX_SYMBOL_NAME_LENGTH = GMS_UEL_IDENT_SIZE - 1
MAX_MATCHING_LENGTH = (
    MAX_SYMBOL_NAME_LENGTH + MAX_SYMBOL_NAME_LENGTH + MAX_MODEL_DECLARATION_LENGTH + 1
)
MAX_NUM_MODEL_ELEMS = int(GMS_MAX_LINE_LENGTH / MAX_MATCHING_LENGTH)
IS_MIRO_INIT = os.getenv("MIRO", False)

logger = logging.getLogger("MODEL")
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(name)s - %(levelname)s] %(message)s")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class Problem(Enum):
    """An enumeration for problem all problem types"""

    LP = "LP"
    "Linear Programming"

    NLP = "NLP"
    """Non-Linear Programming"""

    QCP = "QCP"
    """Quadratically Constrained Programs"""

    DNLP = "DNLP"
    """Nonlinear Programming with Discontinuous Derivatives"""

    MIP = "MIP"
    """Mixed Integer Programming"""

    RMIP = "RMIP"
    """Relaxed Mixed Integer Program"""

    MINLP = "MINLP"
    """Mixed Integer Nonlinear Program"""

    RMINLP = "RMINLP"
    """Relaxed Mixed Integer Nonlinear Program"""

    MIQCP = "MIQCP"
    """Mixed Integer Quadratically Constrained Program"""

    RMIQCP = "RMIQCP"
    """Relaxed Mixed Integer Quadratically Constrained Program"""

    MCP = "MCP"
    """Mixed Complementarity Problem"""

    CNS = "CNS"
    """Constrained Nonlinear System"""

    MPEC = "MPEC"
    """Mathematical Programs with Equilibrium Constraints"""

    RMPEC = "RMPEC"
    """Relaxed Mathematical Program with Equilibrium Constraints"""

    EMP = "EMP"
    """Extended Mathematical Program"""

    MPSGE = "MPSGE"
    """General Equilibrium"""

    @classmethod
    def values(cls):
        """Convenience function to return all values of enum"""
        return list(cls._value2member_map_.keys())

    def __str__(self) -> str:
        return self.value


class Sense(Enum):
    """An enumeration for sense types"""

    MIN = "MIN"
    """Minimize the objective."""

    MAX = "MAX"
    """Maximize the objective."""

    FEASIBILITY = "FEASIBILITY"
    """Assess feasibility."""

    @classmethod
    def values(cls):
        """Convenience function to return all values of enum"""
        return list(cls._value2member_map_.keys())

    def __str__(self) -> str:
        return self.value


class ModelStatus(Enum):
    """An enumeration for model status types"""

    OptimalGlobal = 1
    """The solution is optimal, that is, it is feasible (within tolerances) and it has been proven that no other feasible solution with better objective value exists."""

    OptimalLocal = 2
    """A local optimum for an NLP has been found. That is, a solution that is feasible (within tolerances) and it has been proven that there exists a neighborhood of this solution in which no other feasible solution with better objective value exists."""

    Unbounded = 3
    """The solution is unbounded. This message is reliable if the problem is linear, but occasionally it appears for difficult nonlinear problems that are not truly unbounded, but that lack some strategically placed bounds to limit the variables to sensible values."""

    InfeasibleGlobal = 4
    """The problem has been proven to be infeasible. If this was not intended, something is probably misspecified in the logic or the data."""

    InfeasibleLocal = 5
    """No feasible point could be found for the NLP problem from the given starting point. It does not necessarily mean that no feasible point exists."""

    InfeasibleIntermed = 6
    """The current solution is not feasible, but the solver stopped, either because of a limit (for example, iteration or resource) or because of some sort of difficulty. The solver status will give more information."""

    Feasible = 7
    """A feasible solution to a problem without discrete variables has been found."""

    Integer = 8
    """A feasible solution to a problem with discrete variables has been found."""

    NonIntegerIntermed = 9
    """An incomplete solution to a problem with discrete variables. A feasible solution has not yet been found."""

    IntegerInfeasible = 10
    """It has been proven that there is no feasible solution to a problem with discrete variables."""

    LicenseError = 11
    """The solver cannot find the appropriate license key needed to use a specific subsolver."""

    ErrorUnknown = 12
    """After a solver error the model status is unknown."""

    ErrorNoSolution = 13
    """An error occurred and no solution has been returned. No solution will be returned to GAMS because of errors in the solution process."""

    NoSolutionReturned = 14
    """A solution is not expected for this solve. For example, the CONVERT solver only reformats the model but does not give a solution."""

    SolvedUnique = 15
    """Indicates the solution returned is unique, i.e. no other solution exists. Used for CNS models. Examples where this status could be returned include non-singular linear models, triangular models with constant non-zero elements on the diagonal, and triangular models where the functions are monotone in the variable on the diagonal."""

    Solved = 16
    """Indicates the model has been solved: used for CNS models. The solution might or might not be unique. If the solver uses status SOLVED SINGULAR wherever possible then this status implies that the Jacobian is non-singular, i.e. that the solution is at least locally unique."""

    SolvedSingular = 17
    """Indicates the CNS model has been solved, but the Jacobian is singular at the solution. This can indicate that other solutions exist, either along a line (for linear models) or a curve (for nonlinear models) including the solution returned."""

    UnboundedNoSolution = 18
    """The model is unbounded and no solution can be provided."""

    InfeasibleNoSolution = 19
    """The model is infeasible and no solution can be provided."""


class SolveStatus(Enum):
    """An enumeration for solve status types"""

    NormalCompletion = 1
    """The solver terminated in a normal way."""

    IterationInterrupt = 2
    """The solver was interrupted because it used too many iterations. The option `iteration_limit` may be used to increase the iteration limit if everything seems normal."""

    ResourceInterrupt = 3
    """The solver was interrupted because it used too much time. The option `time_limit` may be used to increase the time limit if everything seems normal."""

    TerminatedBySolver = 4
    """The solver encountered some difficulty and was unable to continue."""

    EvaluationInterrupt = 5
    """Too many evaluations of nonlinear terms at undefined values. We recommend to use variable bounds to prevent forbidden operations, such as division by zero. The rows in which the errors occur are listed just before the solution."""

    CapabilityError = 6
    """The solver does not have the capability required by the model. For example, some solvers do not support certain types of discrete variables or support a more limited set of functions than other solvers."""

    LicenseError = 7
    """The solver cannot find the appropriate license key needed to use a specific subsolver."""

    UserInterrupt = 8
    """The user has sent a signal to interrupt the solver."""

    SetupError = 9
    """The solver encountered a fatal failure during problem set-up time."""

    SolverError = 10
    """The solver encountered a fatal error."""

    InternalError = 11
    """The solver encountered an internal fatal error."""

    Skipped = 12
    """The entire solve step has been skipped."""

    SystemError = 13
    """This indicates a completely unknown or unexpected error condition."""


class FileFormat(Enum):
    """An enumeration for file format types"""

    AMPL = "ampl.mod"
    """AMPL input format."""

    AMPLNL = "ampl.nl"
    """AMPL nl format."""

    CPLEXLP = "cplex.lp"
    """CPLEX LP format."""

    CPLEXMPS = "cplex.mps"
    """CPLEX MPS format."""

    GAMSDict = "dict.txt"
    """GAMS dictionary format."""

    GAMSDictMap = "dictmap.gdx"
    """GAMS dictionary map format."""

    GAMSJacobian = "jacobian.gms"
    """Jacobian in GAMS."""

    GAMSPyJacobian = "jacobian.py"
    """Jacobian in GAMSPy."""

    GDXJacobian = "jacobian.gdx"
    """GDX file with model data incl. Jacobian and Hessian evaluated at current point."""

    FileList = "files.txt"
    """List of file formats generated."""

    FixedMPS = "fixed.mps"
    """Fixed format mps file."""

    GAMS = "gams.gms"
    """GAMS scalar model."""

    JuMP = "jump.jl"
    """JuMP scalar model."""

    LINGO = "lingo.lng"
    """Lingo format."""

    OSiL = "osil.xml"
    """Optimization Services instance Language (OSiL) format."""

    Pyomo = "pyomo.py"
    """Pyomo concrete scalar model."""


INTERRUPT_STATUS = [
    SolveStatus.IterationInterrupt,
    SolveStatus.ResourceInterrupt,
    SolveStatus.EvaluationInterrupt,
    SolveStatus.UserInterrupt,
    SolveStatus.TerminatedBySolver,
]

ERROR_STATUS = {
    SolveStatus.CapabilityError: "The solver does not have the capability required by the model.",
    SolveStatus.LicenseError: "The solver cannot find the appropriate license key needed to use a specific subsolver.",
    SolveStatus.SetupError: "The solver encountered a fatal failure during problem set-up time.",
    SolveStatus.SolverError: "The solver encountered a fatal error.",
    SolveStatus.InternalError: "The solver encountered an internal fatal error.",
    SolveStatus.SystemError: "This indicates a completely unknown or unexpected error condition.",
}


# GAMS name -> GAMSPy name
ATTRIBUTE_MAP = {
    "domUsd": "_num_domain_violations",
    "etAlg": "_algorithm_time",
    "etSolve": "_total_solve_time",
    "etSolver": "_total_solver_time",
    "iterUsd": "_num_iterations",
    "marginals": "_marginals",
    "maxInfes": "_max_infeasibility",
    "meanInfes": "_mean_infeasibility",
    "modelStat": "_status",
    "nodUsd": "_num_nodes_used",
    "number": "_solve_number",
    "numDepnd": "_num_dependencies",
    "numDVar": "_num_discrete_variables",
    "numEqu": "_num_equations",
    "numInfes": "_num_infeasibilities",
    "numNLIns": "_num_nonlinear_insts",
    "numNLNZ": "_num_nonlinear_zeros",
    "numNOpt": "_num_nonoptimalities",
    "numNZ": "_num_nonzeros",
    "numRedef": "_num_mcp_redefinitions",
    "numVar": "_num_variables",
    "numVarProj": "_num_bound_projections",
    "objEst": "_objective_estimation",
    "objVal": "_objective_value",
    "procUsed": "_used_model_type",
    "resGen": "_model_generation_time",
    "resUsd": "_solve_model_time",
    "solveStat": "_solve_status",
    "sumInfes": "_sum_infeasibilities",
    "sysVer": "_solver_version",
}


class Model:
    """
    Represents a list of equations to be solved.

    Parameters
    ----------
    container : Container
        Container of the model.
    name : str, optional
        Name of the model. Name is autogenerated by default.
    description : str, optional
        Description of the model.
    equations : Sequence[Equation]
        Sequence of Equation objects.
    problem : Problem | str, optional
        'LP', 'NLP', 'QCP', 'DNLP', 'MIP', 'RMIP', 'MINLP', 'RMINLP', 'MIQCP', 'RMIQCP', 'MCP', 'CNS', 'MPEC', 'RMPEC', 'EMP', or 'MPSGE',
        by default Problem.LP.
    sense : Sense | str, optional
        "MIN", "MAX", or "FEASIBILITY". By default, Sense.FEASIBILITY
    objective : Variable | Expression, optional
        Objective variable to minimize or maximize or objective itself.
    matches : dict[Equation | Sequence[Equation], Variable | Sequence[Variable]], optional
        Equation - Variable matches for MCP models.
    limited_variables : Sequence[ImplicitVariable], optional
        Allows limiting the domain of variables used in a model.
    external_module: str, optional
        The name of the external module in which the external equations are implemented

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> v = gp.Variable(m, "v")
    >>> e = gp.Equation(m, "e", definition= v == 5)
    >>> my_model = gp.Model(m, "my_model", problem="LP", equations=[e])

    """

    # Prefix for auto-generated symbols
    _generate_prefix = "autogenerated_"
    _autogen_symbols: ClassVar[set[Variable | Equation]] = set()

    def __init__(
        self,
        container: Container | None = None,
        name: str | None = None,
        description: str = "",
        problem: Problem | str = Problem.MIP,
        equations: Sequence[Equation] | None = None,
        sense: Sense | str = Sense.FEASIBILITY,
        objective: Variable | Expression | Operation | None = None,
        matches: dict[
            Equation | Sequence[Equation],
            Variable | Sequence[Variable],
        ]
        | None = None,
        limited_variables: Sequence[ImplicitVariable] | None = None,
        external_module: str | None = None,
    ):
        self._auto_id = "m" + utils._get_unique_name()
        if equations is None:
            equations = []

        self.description = description

        if container is not None:
            self.container = container
        else:
            self.container = gp._ctx_managers[(os.getpid(), threading.get_native_id())]

        assert self.container is not None

        if name is not None:
            name = validation.validate_name(name)
            self.name = validation.validate_model_name(name)
        else:
            self.name = self.container._get_symbol_name(prefix="m")

        self._matches = matches
        self.problem, self.sense = validation.validate_model(
            equations, matches, problem, sense
        )
        self.equations = list(equations)
        self._objective = objective
        self._objective_variable = self._set_objective_variable(objective)
        if (
            self._objective_variable is not None
            and self._objective_variable.type.lower() != "free"
        ):
            raise ValidationError(
                f"Objective variable `{self._objective_variable}` must be a free variable"
            )

        if limited_variables is not None and not isinstance(
            limited_variables, Sequence
        ):
            raise ValidationError(
                f"`limited_variables must be an Iterable of ImplicitVariable objects but found {limited_variables}`"
            )

        self._limited_variables: Sequence[ImplicitVariable] | None = limited_variables

        if not self.equations and not self._matches:
            raise ValidationError("Model requires at least one equation.")

        self._external_module_file: str | None = None
        self._external_module: str | None = None

        if external_module is not None:
            self.external_module = external_module
        else:
            # To avoid adding it twice
            self.container._add_statement(self)

        # allow freezing
        self._is_frozen: bool = False

        # Attributes
        self._num_domain_violations: int | None = None
        self._algorithm_time: float | None = None
        self._total_solve_time: float | None = None
        self._total_solver_time: float | None = None
        self._num_iterations: int | None = None
        self._marginals: int | None = None
        self._max_infeasibility: float | None = None
        self._mean_infeasibility: float | None = None
        self._status: ModelStatus | None = None
        self._num_nodes_used: int | None = None
        self._solve_number: int | None = None
        self._num_dependencies: int | None = None
        self._num_discrete_variables: int | None = None
        self._num_equations: int | None = None
        self._num_infeasibilities: int | None = None
        self._num_nonlinear_insts: int | None = None
        self._num_nonlinear_zeros: int | None = None
        self._num_nonoptimalities: int | None = None
        self._num_nonzeros: int | None = None
        self._num_mcp_redefinitions: int | None = None
        self._num_variables: int | None = None
        self._num_bound_projections: int | None = None
        self._objective_estimation: float | None = None
        self._objective_value: float | None = None
        self._used_model_type: str | None = None
        self._model_generation_time: float | None = None
        self._solve_model_time: float | None = None
        self._sum_infeasibilities: float | None = None
        self._solve_status: SolveStatus | None = None
        self._solver_version: int | None = None

        self._updated_attrs: dict[str, bool] = {}

        self.container.models.update({self.name: self})
        self.container._synch_with_gams()

    def _serialize(self) -> dict:
        info: dict[str, Any] = {
            "name": self.name,
            "problem": str(self.problem),
            "sense": str(self.sense),
        }

        # equations
        equations = [equation.name for equation in self.equations]
        info["equations"] = equations

        # matches
        if self._matches is not None:
            matches: dict = {}
            for key, value in self._matches.items():
                if isinstance(key, gp.Equation):
                    if isinstance(value, gp.Variable):
                        matches[key.name] = value.name
                    else:
                        matches[key.name] = (variable.name for variable in value)
                else:
                    assert isinstance(value, gp.Variable)
                    for equation in key:
                        matches[equation.name] = value.name

            info["_matches"] = matches

        # objective variable
        if self._objective_variable is not None:
            info["_objective_variable"] = self._objective_variable.name

        # attributes
        for attr_name in ATTRIBUTE_MAP.values():
            if attr_name in ("_status", "_solve_status"):
                attr_value = getattr(self, attr_name, None)
                if attr_value is not None:
                    attr_value = attr_value.value

                info[attr_name] = attr_value
            else:
                info[attr_name] = getattr(self, attr_name, None)

        return info

    def __repr__(self) -> str:
        return f"Model(name='{self.name}', problem='{self.problem!s}', equations={self.equations}, sense='{self.sense!s}', objective={self._objective_variable}, matches={self._matches}, limited_variables={self._limited_variables}"

    def __str__(self) -> str:
        return (
            f"Model {self.name}:\n  Problem Type: {self.problem}\n  Sense:"
            f" {self.sense}\n  Equations: {self.equations}"
        )

    @property
    def num_domain_violations(self) -> int | None:
        """
        Number of domain violations.

        Returns
        -------
        int | None
        """
        return self._num_domain_violations

    @property
    def algorithm_time(self) -> float | None:
        """
        Solver dependent timing information. This attribute was intended to allow
        solvers to return the elapsed time used by the solve algorithm without
        including any model generation, communication, or setup time. However,
        solvers are free to adapt this convention and return time-related information
        (but not necessarily elapsed time) for executing the solve algorithm. Please
        inspect your solver manual for the actual meaning of the value returned in
        this attribute.

        Returns
        -------
        float | None
        """
        return self._algorithm_time

    @property
    def total_solve_time(self) -> float | None:
        """
        Elapsed time it took to execute a solve statement in total. This model
        attribute returns the elapsed time it took to execute a solve statement in total.
        This time includes the model generation time, the time to read and write files,
        the time to create the solution report and the time taken by the actual solve.
        The time is expressed in seconds of wall-clock time.

        Returns
        -------
        float | None
        """
        return self._total_solve_time

    @property
    def total_solver_time(self) -> float | None:
        """
        Elapsed time taken by the solver only. This model attribute returns the elapsed
        time taken by the solver only. This does not include the GAMS model generation time
        and the time taken to report and load the solution back into the GAMS database.
        The time is expressed in seconds of wall-clock time.

        Returns
        -------
        float | None
        """
        return self._total_solver_time

    @property
    def num_iterations(self) -> int | None:
        """
        Number of iterations used.

        Returns
        -------
        int | None
        """
        return self._num_iterations

    @property
    def marginals(self) -> int | None:
        """
        Indicates whether there are marginals.

        Returns
        -------
        int | None
        """
        return self._marginals

    @property
    def status(self) -> ModelStatus | None:
        """
        Model status after solve.

        Returns
        -------
        ModelStatus | None
        """
        return self._status

    @property
    def num_nodes_used(self) -> int | None:
        """
        Number of nodes used by the MIP solver.

        Returns
        -------
        int | None
        """
        return self._num_nodes_used

    @property
    def solve_number(self) -> int | None:
        """
        Number of the last solve.

        Returns
        -------
        int | None
        """
        return self._solve_number

    @property
    def num_discrete_variables(self) -> int | None:
        """
        Number of discrete variables.

        Returns
        -------
        int | None
        """
        return self._num_discrete_variables

    @property
    def num_equations(self) -> float | None:
        """
        Number of equations.

        Returns
        -------
        float | None
        """
        return self._num_equations

    @property
    def num_nonlinear_zeros(self) -> int | None:
        """
        Number of nonlinear nonzeros.

        Returns
        -------
        int | None
        """
        return self._num_nonlinear_zeros

    @property
    def num_nonzeros(self) -> int | None:
        """
        Number of nonzero entries in the model coefficient matrix.

        Returns
        -------
        int | None
        """
        return self._num_nonzeros

    @property
    def num_variables(self) -> int | None:
        """
        Number of variables.

        Returns
        -------
        int | None
        """
        return self._num_variables

    @property
    def objective_estimation(self) -> float | None:
        """
        Estimate of the best possible solution for a mixed-integer model

        Returns
        -------
        float | None
        """
        return self._objective_estimation

    @property
    def objective_value(self) -> float | None:
        """
        Objective function value

        Returns
        -------
        float | None
        """
        return self._objective_value

    @property
    def used_model_type(self) -> str | None:
        """
        Model type.

        Returns
        -------
        str | None
        """
        return self._used_model_type

    @property
    def model_generation_time(self) -> float | None:
        """
        Time GAMS took to generate the model in wall-clock seconds.

        Returns
        -------
        float | None
        """
        return self._model_generation_time

    @property
    def solve_model_time(self) -> float | None:
        """
        Time the solver used to solve the model in seconds

        Returns
        -------
        float | None
        """
        return self._solve_model_time

    @property
    def solve_status(self) -> SolveStatus | None:
        """
        Indicates the solver termination condition.

        Returns
        -------
        SolveStatus | None
        """
        return self._solve_status

    @solve_status.setter
    def solve_status(self, value: SolveStatus) -> None:
        if value in INTERRUPT_STATUS:
            logger.warning(
                f"The solve was interrupted! Solve status: {value.name}. "
                "For further information, see https://gamspy.readthedocs.io/en/latest/reference/gamspy._model.html#gamspy.SolveStatus."
            )
        elif value in ERROR_STATUS:
            raise GamspyException(
                f"Solve status: {value.name}. {ERROR_STATUS[value]}",
                value.value,
            )

        self._solve_status = value

    @property
    def solver_version(self) -> int | None:
        """
        Solver version.

        Returns
        -------
        int | None
        """
        return self._solver_version

    ### Model attributes that require a call to GAMS to be updated. Other attributes above are updated via trace file generated by the last solve.
    @property
    def num_dependencies(self) -> int | None:
        """
        Number of dependencies in a CNS model.

        Returns
        -------
        int | None
        """
        self._update_model_attribute("_num_dependencies", "numDepnd")
        return self._num_dependencies

    @property
    def num_infeasibilities(self) -> int | None:
        """
        Number of infeasibilities.

        Returns
        -------
        int | None
        """
        self._update_model_attribute("_num_infeasibilities", "numInfes")
        return self._num_infeasibilities

    @property
    def num_nonlinear_insts(self) -> int | None:
        """
        Number of nonlinear instructions.

        Returns
        -------
        int | None
        """
        self._update_model_attribute("_num_nonlinear_insts", "numNLIns")
        return self._num_nonlinear_insts

    @property
    def num_nonoptimalities(self) -> int | None:
        """
        Number of nonoptimalities.

        Returns
        -------
        int | None
        """
        self._update_model_attribute("_num_nonoptimalities", "numNOpt")
        return self._num_nonoptimalities

    @property
    def num_mcp_redefinitions(self) -> int | None:
        """
        Number of MCP redefinitions.

        Returns
        -------
        int | None
        """
        self._update_model_attribute("_num_mcp_redefinitions", "numRedef")
        return self._num_mcp_redefinitions

    @property
    def num_bound_projections(self) -> int | None:
        """
        Number of bound projections during model generation.

        Returns
        -------
        int | None
        """
        self._update_model_attribute("_num_bound_projections", "numVarProj")
        return self._num_bound_projections

    @property
    def sum_infeasibilities(self) -> float | None:
        """
        Sum of infeasibilities.

        Returns
        -------
        float | None
        """
        self._update_model_attribute("_sum_infeasibilities", "sumInfes")
        return self._sum_infeasibilities

    @property
    def max_infeasibility(self) -> float | None:
        """
        Maximum of infeasibilities

        Returns
        -------
        float | None
        """
        self._update_model_attribute("_max_infeasibility", "maxInfes")
        return self._max_infeasibility

    @property
    def mean_infeasibility(self) -> float | None:
        """
        Mean of infeasibilities

        Returns
        -------
        float | None
        """
        self._update_model_attribute("_mean_infeasibility", "meanInfes")
        return self._mean_infeasibility

    @property
    def external_module(self) -> str | None:
        """
        Name of the external module in which the external equations are implemented.
        By default, this parameter is set to None. When provided, it triggers the
        opening of the specified file using a File statement and incorporates the
        file into the model by adding it as an external module.

        This feature requires a solid understanding of programming, compilation,
        and linking processes. For more information, please refer to the
        https://gamspy.readthedocs.io/en/latest/user/advanced/external_equations.html .

        Returns
        -------
        str | None
        """
        return self._external_module

    @external_module.setter
    def external_module(self, value: str | None):
        if self._external_module_file is not None:
            self.container._add_statement(f"putclose {self._external_module_file};")

        write_model_statement = value != self._external_module
        self._external_module = None
        self._external_module_file = None

        if value is not None:
            filename = "f" + utils._get_unique_name()
            self._external_module_file = filename
            self._external_module = value
            self.container._add_statement(f"File {filename} / '{value}' /;")

        if write_model_statement:
            self.container._add_statement(self)

    def _generate_obj_var_and_equation(self) -> tuple[Variable, Equation]:
        variable = gp.Variable._constructor_bypass(
            self.container,
            f"{self.name}_objective_variable",
            domain=[],
        )
        equation = gp.Equation._constructor_bypass(
            self.container,
            f"{self.name}_objective",
            domain=[],
        )
        self._autogen_symbols.add(variable.name)
        self._autogen_symbols.add(equation.name)

        return variable, equation

    def _set_objective_variable(
        self,
        assignment: None | Variable | Operation | Expression | MathOp = None,
    ) -> Variable | None:
        """
        Returns objective variable. If the assignment is an Expression
        or an Operation (Sum, Product etc.), it automatically generates
        an objective variable and a equation.

        Returns
        -------
        Variable | None

        Raises
        ------
        TypeError
            In case assignment is not a Variable, Expression or an Operation.
        """

        if assignment is not None and not isinstance(
            assignment,
            (
                gp.Variable,
                expression.Expression,
                operation.Operation,
                gp.math.MathOp,
            ),
        ):
            raise TypeError(
                "Objective must be a Variable or an Expression but"
                f" {type(assignment)} given"
            )

        if self.sense == gp.Sense.FEASIBILITY:
            if assignment is not None:
                raise ValidationError(
                    "Cannot set an objective when the sense is FEASIBILITY!"
                )

            if self.problem in (Problem.CNS, Problem.MCP, Problem.EMP):
                return None

            variable, equation = self._generate_obj_var_and_equation()
            statement = expression.Expression(
                implicits.ImplicitEquation(
                    equation,
                    name=equation.name,
                    type=equation.type,
                    domain=[],
                ),
                "..",
                variable == 0,
            )
            self.container._add_statement(statement)
            equation._definition = statement
            equation.modified = False
            variable.modified = False

            if equation.name not in (symbol.name for symbol in self.equations):
                self.equations.append(equation)

            return variable

        if isinstance(
            assignment,
            (expression.Expression, operation.Operation, gp.math.MathOp),
        ):
            variable, equation = self._generate_obj_var_and_equation()

            # Sum((i,j),c[i,j]*x[i,j])->Sum((i,j),c[i,j]*x[i,j]) =e= var
            assignment = assignment == variable

            # equation .. Sum((i,j),c[i,j]*x[i,j]) =e= var
            statement = expression.Expression(
                implicits.ImplicitEquation(
                    equation,
                    name=equation.name,
                    type=equation.type,
                    domain=[],
                ),
                "..",
                assignment,
            )
            self.container._add_statement(statement)
            equation._definition = statement
            equation.modified = False
            variable.modified = False
            if equation.name not in (symbol.name for symbol in self.equations):
                self.equations.append(equation)

            return variable

        return assignment

    def _generate_solve_string(self) -> str:
        solve_statement = [f"solve {self.name} using {self.problem}"]

        if self.sense == gp.Sense.FEASIBILITY:
            # Set sense as min or max for feasibility
            self.sense = gp.Sense("MIN")

        if self.problem not in (Problem.MCP, Problem.CNS, Problem.EMP):
            solve_statement.append(str(self.sense))

        if self._objective_variable is not None:
            solve_statement.append(self._objective_variable.gamsRepr())

        return " ".join(solve_statement)

    def _add_runtime_options(self, options: Options, backend: str = "local") -> None:
        for key, value in options.model_dump(exclude_none=True).items():
            if key in MODEL_ATTR_OPTION_MAP:
                if isinstance(value, bool):
                    value = int(value)
                elif isinstance(value, str):
                    value = f"'{value}'"

                self.container._add_statement(
                    f"{self.name}.{MODEL_ATTR_OPTION_MAP[key]} = {value};\n"
                )
            elif key in EXECUTION_OPTIONS:
                if key == "loadpoint":
                    if isinstance(value, os.PathLike):
                        value = os.path.abspath(value)

                    if backend == "engine":
                        value = os.path.relpath(value, self.container.working_directory)

                self.container._add_statement(f"{EXECUTION_OPTIONS[key]} '{value}';\n")

    def _update_model_attribute(self, python_attr: str, gams_attr: str) -> None:
        if self._updated_attrs.get(python_attr, False):
            # If it's already updated, return
            return

        symbol_name = f"{self._generate_prefix}{gams_attr}_{self._auto_id}"
        self.container._add_statement(f"Parameter {symbol_name};")
        self.container._add_statement(f"{symbol_name} = {self.name}.{gams_attr};")
        self.container._synch_with_gams(gams_to_gamspy=True)

        gdx_handle = utils._open_gdx_file(
            self.container.system_directory, self.container._gdx_out
        )
        data = utils._get_scalar_data(self.container._gams2np, gdx_handle, symbol_name)
        setattr(self, python_attr, data)
        utils._close_gdx_handle(gdx_handle)

        self._updated_attrs[python_attr] = True

    def computeInfeasibilities(self) -> dict[str, pd.DataFrame]:
        """
        Computes infeasabilities for all equations of the model

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary of infeasibilities where equation names are keys and
            infeasibilities are values

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["i1", "i2"])
        >>> j = gp.Set(m, name="j", records=["j1", "j2", "j3"])
        >>> a = gp.Parameter(m, name="a", domain=i, records=[("i1", 350), ("i2", 600)])
        >>> b = gp.Parameter(m, name="b", domain=j, records=[("j1", 400), ("j2", 450), ("j3", 420)])
        >>> x = gp.Variable(m, name="x", domain=[i,j], type="Positive")
        >>> s = gp.Equation(m, name="s", domain=i)
        >>> d = gp.Equation(m, name="d", domain=j)
        >>> s[i] = gp.Sum(j, x[i, j]) <= a[i]
        >>> d[j] = gp.Sum(i, x[i, j]) >= b[j]
        >>> my_model = gp.Model(m, name="my_model", equations=m.getEquations(), problem="LP", sense="min", objective=gp.Sum((i, j), x[i, j]))
        >>> summary = my_model.solve()
        >>> infeasibilities = my_model.computeInfeasibilities()
        >>> infeasibilities["s"].infeasibility.item()
        320.0

        """
        infeas_dict = {}

        for equation in self.equations:
            if equation.records is None:
                continue

            infeas_rows = utils._calculate_infeasibilities(equation)
            infeas_dict[equation.name] = infeas_rows

            if equation._definition is not None:
                names = equation._definition._find_all_symbols()
                names = [
                    name
                    for name in names
                    if name in self.container.data
                    and isinstance(self.container[name], gp.Variable)
                ]

                for name in names:
                    variable = self.container[name]
                    infeas_rows = utils._calculate_infeasibilities(variable)
                    infeas_dict[variable.name] = infeas_rows

        return infeas_dict

    def convert(
        self,
        path: str | Path,
        file_format: FileFormat | Sequence[FileFormat],
        options: ConvertOptions | None = None,
    ) -> None:
        """
        Converts the model to one or more specified file formats.

        Parameters
        ----------
        path : str | Path
            Path to the directory where the converted model files will be saved.
        file_format : FileFormat | Sequence[FileFormat]
            File format(s) to convert the model to. Can be a single FileFormat or a list of FileFormats.
        options : ConvertOptions, optional
            Additional options to customize the conversion process.

        Raises
        ------
        ValueError
            If the specified file format is not supported.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> v = gp.Variable(m, "v")
        >>> e = gp.Equation(m, "e", definition= v == 5)
        >>> my_model = gp.Model(m, "my_model", problem="LP", equations=[e])
        >>> my_model.convert("output_directory", gp.FileFormat.GAMS)
        >>> my_model.convert("output_directory", [gp.FileFormat.GAMS, gp.FileFormat.AMPL])

        """
        path = Path(path)
        os.makedirs(path, exist_ok=True)
        solver_options = get_convert_solver_options(path, file_format, options)
        self.solve(solver="convert", solver_options=solver_options)

    def getEquationListing(
        self,
        infeasibility_threshold: float | None = None,
    ) -> str:
        """
        Returns the generated equations.

        Parameters
        ----------
        infeasibility_threshold: float, optional
            Filters out equations with infeasibilities that are above this value.

        Returns
        -------
        str
        """
        listings = []
        for equation in self.equations:
            listing = equation.getEquationListing(
                infeasibility_threshold=infeasibility_threshold
            )

            if listing:
                listings.append(listing)

        return "\n".join(listings)

    def getVariableListing(self) -> str:
        """
        Returns the variable listing.

        Returns
        -------
        str
        """
        if not hasattr(self, "_variables"):
            raise ValidationError(
                "The model must be solved with `variable_listing_limit` option for this functionality to work."
            )

        listings = []
        for variable in self._variables:
            listings.append(variable.getVariableListing())

        return "\n".join(listings)

    def interrupt(self) -> None:
        """Sends interrupt signal to the running job."""
        self.container._interrupt()

    def freeze(
        self,
        modifiables: list[Parameter | ImplicitParameter],
        options: Options | None = None,
    ) -> None:
        """
        Instantiates a model instance. After calling freeze, only modifiables can be modified.

        Parameters
        ----------
        modifiables : list[Parameter  |  ImplicitParameter]
            Modifiable symbols.
        options : Options | None, optional
            GAMSPy options, by default None
        """
        self._is_frozen = True
        if options is None:
            options = self.container._options

        self.instance = ModelInstance(
            self.container, self, modifiables, options, self.container.output
        )

    def unfreeze(self) -> None:
        """Unfreezes the model"""
        self._is_frozen = False
        self.instance.close_license_session()

    def solve(
        self,
        solver: str | None = None,
        options: Options | None = None,
        solver_options: dict | str | Path | None = None,
        freeze_options: FreezeOptions | None = None,
        output: io.TextIOWrapper | None = None,
        backend: Literal["local", "engine", "neos"] = "local",
        client: EngineClient | NeosClient | None = None,
        load_symbols: list[Symbol] | None = None,
    ) -> pd.DataFrame | None:
        """
        Solves the model with given options.

        Parameters
        ----------
        solver : str, optional
            Solver name
        options : Options, optional
            GAMSPy options.
        solver_options : dict | str | Path, optional
            Dictionary of solver options or path to an existing option file.
        freeze_options : FreezeOptions, optional
            Options to solve a frozen model.
        output : TextIOWrapper, optional
            Output redirection target
        backend : str, optional
            Backend to run on
        client : EngineClient, NeosClient, optional
            EngineClient to communicate with GAMS Engine or NEOS Client to communicate with NEOS Server
        load_symbols : list[Symbol], optional
            Specifies the symbols that need to be loaded. If not given, all symbols are loaded after solve.

        Returns
        -------
        DataFrame, optional
            Summary of the solve

        Raises
        ------
        ValidationError
            In case engine_config is not provided for `engine` backend or
            neos_client is not provided for `neos` backend.
        ValueError
            In case problem is not in possible problem types
        ValueError
            In case sense is different than "MIN" or "MAX"

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> v = gp.Variable(m, "v")
        >>> e = gp.Equation(m, "e", definition= v == 5)
        >>> my_model = gp.Model(m, "my_model", problem="LP", equations=[e], sense="max", objective=v)
        >>> solved = my_model.solve()

        """
        self._updated_attrs = {}
        if output is None:
            output = self.container.output

        if solver is None:
            solver = utils.getDefaultSolvers(self.container.system_directory)[
                str(self.problem).upper()
            ].lower()
        else:
            if not isinstance(solver, str):
                raise TypeError(
                    f"`solver` argument must be of type `str` but given `{type(solver)}`"
                )
            solver = solver.lower()

        if solver == "conopt":
            solver = "conopt4"

        validation.validate_solver_args(
            self.container.system_directory,
            backend,
            solver,
            self.problem,
            options,
            output,
            load_symbols,
        )
        validation.validate_equations(self)

        if options is None:
            options = self.container._options

        if isinstance(solver_options, (str, Path)):
            solver_options = Path(solver_options)

        # Only for local until GAMS Engine and NEOS Server backends adopt the new GP_SolveLine option.
        if backend == "local":
            frame = inspect.currentframe().f_back
            assert isinstance(options, gp.Options)
            options._frame = frame

        if solver_options is not None:
            self.container.writeSolverOptions(solver, solver_options)

        if self._is_frozen:
            instance_options = FreezeOptions()

            if freeze_options is not None:
                instance_options = freeze_options

            summary = self.instance.solve(
                solver, instance_options, solver_options, output
            )
            return summary

        self.container._add_statement(self.getDeclaration())
        self._add_runtime_options(options, backend)
        self.container._add_statement(self._generate_solve_string() + "\n")
        options._set_model_info(solver, self.problem, solver_options)

        runner = backend_factory(
            self.container,
            options,
            solver,
            solver_options,
            output,
            backend,
            client,
            self,
            load_symbols,
        )

        summary = runner.run(relaxed_domain_mapping=False, gams_to_gamspy=True)

        if IS_MIRO_INIT:
            miro._write_default_gdx_miro(self.container)

        return summary

    def getDeclaration(self) -> str:
        """
        Declaration of the Model in GAMS

        Returns
        -------
        str

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> v = gp.Variable(m, "v")
        >>> e = gp.Equation(m, "e", definition= v == 5)
        >>> my_model = gp.Model(m, "my_model", problem="LP", equations=[e])
        >>> my_model.getDeclaration()
        'Model my_model / e,my_model_objective /;'

        """

        def build_str_with_new_lines(elements: list[str]) -> str:
            num_elements = len(elements)
            if num_elements < MAX_NUM_MODEL_ELEMS:
                return ",".join(elements)

            count = 0
            return_str = ""
            while count < num_elements:
                batch = elements[count : count + MAX_NUM_MODEL_ELEMS]
                return_str += ",".join(batch) + "\n"
                count += MAX_NUM_MODEL_ELEMS

            return return_str

        equations_in_matches = []
        if self._matches is not None:
            for key in self._matches:
                if isinstance(key, gp.Equation):
                    equations_in_matches.append(key)
                else:
                    equations_in_matches += list(key)

        equations = []
        for equation in self.equations:
            if self._matches is not None:
                if equation not in equations_in_matches:
                    equations.append(equation.name)
            else:
                equations.append(equation.name)

        equations_str = build_str_with_new_lines(equations)

        if self._matches is not None:
            matches = []
            for key, value in self._matches.items():
                if isinstance(key, gp.Equation):
                    if isinstance(value, gp.Variable):
                        matches.append(f"{key.name}.{value.name}")
                    else:
                        matches.append(
                            f"{key.name}:({'|'.join([variable.name for variable in value])})"
                        )
                else:
                    assert isinstance(value, gp.Variable)
                    matches.append(
                        f"({'|'.join([equation.name for equation in key])}):{value.name}"
                    )

            matches_str = build_str_with_new_lines(matches)

            equations_str = (
                ",".join([equations_str, matches_str]) if equations else matches_str
            )

        if self._external_module_file is not None:
            equations_str = ",".join([equations_str, self._external_module_file])

        if self._limited_variables is not None:
            limited_variables_str = ",".join(
                [variable.gamsRepr() for variable in self._limited_variables]
            )
            equations_str += "," + limited_variables_str

        model_str = f"Model {self.name}"
        if self.description:
            model_str += ' "' + self.description + '"'

        if equations_str != "":
            model_str += f" / {equations_str} /"
        model_str += ";"

        return model_str

    def toGams(
        self,
        path: str | Path,
        options: Options | None = None,
        *,
        dump_gams_state: bool = False,
    ) -> None:
        """
        Generates GAMS model under path/<model_name>.gms.

        Parameters
        ----------
        path : str | Path
            Path to the directory which will contain the GAMS model.
        options : Options | None, optional
            GAMSPy options, by default None
        dump_gams_state : bool, optional
            Whether to dump the state as a GAMS save file, by default False

        Raises
        ------
        ValidationError
            In case the given options is not of type gp.Options.
        """
        if options is not None and not isinstance(options, gp.Options):
            raise ValidationError(
                f"`options` must be of type gp.Options of found {type(options)}"
            )

        path = Path(path)
        converter = GamsConverter(self, path, options, dump_gams_state)
        converter.convert()

    def toLatex(self, path: str | Path, generate_pdf: bool = False) -> None:
        """
        Generates a latex file that contains the model definition under path/<model_name>.tex

        Parameters
        ----------
        path : str | Path
            Path to the directory which will contain the .tex file.
        """
        path = Path(path)
        converter = LatexConverter(self, path)
        converter.convert()

        if generate_pdf:
            converter.to_pdf()
