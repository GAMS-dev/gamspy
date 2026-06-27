from __future__ import annotations

import inspect
import logging
import os
import threading
import warnings
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

from gams.core.gdx import GMS_UEL_IDENT_SIZE

import gamspy as gp
import gamspy._algebra.expression as expression
import gamspy._algebra.operation as operation
import gamspy._gdx as gdxio
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
from gamspy._internals import ATTR_PREFIX, MODEL_ATTRIBUTE_MAP
from gamspy._model_instance import ModelInstance
from gamspy._options import (
    EXECUTION_OPTIONS,
    MODEL_ATTR_OPTION_MAP,
    ConvertOptions,
    FreezeOptions,
    Options,
    write_solver_options,
)
from gamspy.exceptions import GamspyException, ValidationError

if TYPE_CHECKING:
    from typing import Literal, TextIO

    import pandas as pd

    from gamspy import Container, Equation, Parameter, Variable
    from gamspy._algebra.expression import Expression
    from gamspy._algebra.operation import Operation
    from gamspy._backend.engine import EngineClient
    from gamspy._backend.neos import NeosClient
    from gamspy._guss import GUSSScenarioDict
    from gamspy._symbols.implicits import ImplicitParameter, ImplicitVariable
    from gamspy._types import SymbolType
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

    @classmethod
    def values(cls):
        """Convenience function to return all values of enum"""
        return list(cls._value2member_map_.keys())

    def __str__(self) -> str:
        return self.value


INT_TO_PROBLEM: dict[int, str] = {
    1: str(Problem.LP),
    2: str(Problem.MIP),
    3: str(Problem.RMIP),
    4: str(Problem.NLP),
    5: str(Problem.MCP),
    6: str(Problem.MPEC),
    7: str(Problem.RMPEC),
    8: str(Problem.CNS),
    9: str(Problem.DNLP),
    10: str(Problem.RMINLP),
    11: str(Problem.MINLP),
    12: str(Problem.QCP),
    13: str(Problem.MIQCP),
    14: str(Problem.RMIQCP),
    15: str(Problem.EMP),
}


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

    NotSolved = 0
    """In case the model is not solved yet."""

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


class Model:
    """
    Represents a collection of equations and variables to be solved as a mathematical optimization problem.
    https://gamspy.readthedocs.io/en/latest/user/basics/model.html

    Parameters
    ----------
    container : Container
        The Container object that this model belongs to.
    name : str, optional
        Name of the model. If omitted, a unique name is generated automatically.
    description : str, optional
        A human-readable description of the model.
    equations : Sequence[Equation], optional
        A list or sequence of Equation objects that define the constraints of the model.
        No equations are used by default.
    problem : Problem | str, optional
        The type of mathematical problem to solve (e.g., 'LP', 'MIP', 'NLP').
        Default is Problem.MIP.
    sense : Sense | str, optional
        The optimization sense: "MIN" (minimize), "MAX" (maximize), or "FEASIBILITY".
        Default is Sense.FEASIBILITY.
    objective : Variable | Expression | Operation, optional
        The objective to optimize. Can be a scalar variable, an expression, or an operation (like Sum).
    matches : dict[Equation | Sequence[Equation], Variable | Sequence[Variable]], optional
        Used for defining complementarity problems (MCP). Maps equations to their complementary variables.
    limited_variables : Sequence[ImplicitVariable], optional
        Allows limiting the domain of variables included in the model to a specific subset.
    external_module : str, optional
        The name of an external module file (e.g., 'my_external.dll' or 'my_external.so')
        where external equations are implemented.

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, description="items")
    >>> p = gp.Parameter(m, description="profits", domain=i)
    >>> w = gp.Parameter(m, description="weights", domain=i)
    >>> c = gp.Parameter(m, description="capacity")
    >>> x = gp.Variable(m, domain=i, type=gp.VariableType.BINARY)
    >>> capacity_restriction = gp.Equation(m, definition=gp.Sum(i, w[i] * x[i]) <= c)

    >>> # Instantiate the Model
    >>> knapsack = gp.Model(
    ...     m,
    ...     equations=m.getEquations(), # Automatically grabs all equations in the container
    ...     problem=gp.Problem.MIP,     # Mixed Integer Program
    ...     sense=gp.Sense.MAX,         # Maximize profit
    ...     objective=gp.Sum(i, p[i] * x[i]),
    ... )

    """

    # Prefix for auto-generated symbols
    _autogen_symbols: ClassVar[set[str]] = set()

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
        if container is not None and not isinstance(container, gp.Container):
            raise TypeError(
                f"Container must of type `Container` but found {type(container)}"
            )

        self._is_mpsge = False
        self._auto_id = "m" + utils._get_unique_name()
        if equations is None:
            equations = []

        self.description = description

        if container is not None:
            self.container = container
        else:
            try:
                self.container = gp._ctx_managers[
                    (os.getpid(), threading.get_native_id())
                ]
            except KeyError as e:
                raise ValidationError(
                    "Model is missing required argument `container`."
                ) from e

        if name is not None:
            name = validation.validate_name(name)
            self.name = validation.validate_model_name(name)
        else:
            self.name = self.container._get_symbol_name(prefix="m")

        self._matches = matches
        self.problem, self.sense = validation.validate_model(
            equations, matches, problem, sense
        )
        self.equations: list[Equation] = list(equations)
        self._variable_names: list[str] | None = None
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
        self._num_domain_violations: float | None = None
        self._algorithm_time: float | None = None
        self._total_solve_time: float | None = None
        self._total_solver_time: float | None = None
        self._num_iterations: float | None = None
        self._marginals: float | None = None
        self._max_infeasibility: float | None = None
        self._mean_infeasibility: float | None = None
        self._status: ModelStatus = ModelStatus.NotSolved
        self._num_nodes_used: float | None = None
        self._solve_number: float | None = None
        self._num_dependencies: float | None = None
        self._num_discrete_variables: float | None = None
        self._num_equations: float | None = None
        self._num_infeasibilities: float | None = None
        self._num_nonlinear_insts: float | None = None
        self._num_nonlinear_zeros: float | None = None
        self._num_nonoptimalities: float | None = None
        self._num_nonzeros: float | None = None
        self._num_mcp_redefinitions: float | None = None
        self._num_variables: float | None = None
        self._num_bound_projections: float | None = None
        self._objective_estimation: float | None = None
        self._objective_value: float | None = None
        self._used_model_type: str | None = None
        self._model_generation_time: float | None = None
        self._solve_model_time: float | None = None
        self._sum_infeasibilities: float | None = None
        self._solve_status: SolveStatus | None = None
        self._solver_version: float | None = None

        self._default_solver = utils.getDefaultSolvers(self.container.system_directory)[
            str(self.problem).upper()
        ].lower()

        self.container.models.update({self.name: self})
        self._attr_symbol_names = self._create_model_attributes()
        self.container._synch_with_gams()
        self._attr_gdx_file = os.path.join(
            self.container.working_directory, "_" + utils._get_unique_name() + ".gdx"
        )

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
        for attr_name in MODEL_ATTRIBUTE_MAP.values():
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
    def num_domain_violations(self) -> float:
        """
        Number of domain violations in the solution.

        Returns
        -------
        float
        """
        if self._num_domain_violations is None:
            raise ValidationError(
                "The model is not solved yet. Please solve the model first to get the number of domain violations in the solution."
            )

        return self._num_domain_violations

    @property
    def algorithm_time(self) -> float:
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
        float
        """
        if self._algorithm_time is None:
            raise ValidationError(
                "The model is not solved yet. Please solve the model first to get the algorithm time."
            )

        return self._algorithm_time

    @property
    def total_solve_time(self) -> float:
        """
        Elapsed time it took to execute a solve statement in total. This model
        attribute returns the elapsed time it took to execute a solve statement in total.
        This time includes the model generation time, the time to read and write files,
        the time to create the solution report and the time taken by the actual solve.
        The time is expressed in seconds of wall-clock time.

        Returns
        -------
        float
        """
        if self._total_solve_time is None:
            raise ValidationError(
                "The model is not solved yet. Please solve the model first to get the total solve time."
            )

        return self._total_solve_time

    @property
    def total_solver_time(self) -> float:
        """
        Elapsed time taken by the solver only. This model attribute returns the elapsed
        time taken by the solver only. This does not include the GAMS model generation time
        and the time taken to report and load the solution back into the GAMS database.
        The time is expressed in seconds of wall-clock time.

        Returns
        -------
        float
        """
        if self._total_solver_time is None:
            raise ValidationError(
                "The model is not solved yet. Please solve the model first to get the total solver time."
            )
        return self._total_solver_time

    @property
    def num_iterations(self) -> float:
        """
        Number of iterations used.

        Returns
        -------
        float
        """
        if self._num_iterations is None:
            raise ValidationError(
                "The model is not solved yet. Please solve the model first to get the number of iterations."
            )

        return self._num_iterations

    @property
    def marginals(self) -> float:
        """
        Indicates whether there are marginals.

        Returns
        -------
        float
        """
        if self._marginals is None:
            raise ValidationError(
                "The model is not solved yet. Please solve the model first to get the marginals."
            )

        return self._marginals

    @property
    def max_infeasibility(self) -> float:
        """
        Maximum of infeasibilities

        Returns
        -------
        float
        """
        if self._max_infeasibility is None:
            raise ValidationError(
                "The model is not solved yet. Please solve the model first to get the maximum of infeasibilities."
            )

        return self._max_infeasibility

    @property
    def mean_infeasibility(self) -> float:
        """
        Mean of infeasibilities

        Returns
        -------
        float
        """
        if self._mean_infeasibility is None:
            raise ValidationError(
                "The model is not solved yet. Please solve the model first to get the mean of infeasibilities."
            )

        return self._mean_infeasibility

    @property
    def status(self) -> ModelStatus:
        """
        Model status after solve.

        Returns
        -------
        ModelStatus
        """
        return self._status

    @property
    def num_nodes_used(self) -> float | None:
        """
        Number of nodes used by the MIP solver.

        Returns
        -------
        float | None
        """
        return self._num_nodes_used

    @property
    def solve_number(self) -> float | None:
        """
        Number of the last solve.

        Returns
        -------
        float | None
        """
        return self._solve_number

    @property
    def num_dependencies(self) -> float | None:
        """
        Number of dependencies in a CNS model.

        Returns
        -------
        float | None
        """
        return self._num_dependencies

    @property
    def num_discrete_variables(self) -> float | None:
        """
        Number of discrete variables.

        Returns
        -------
        float | None
        """
        return self._num_discrete_variables

    @property
    def num_equations(self) -> float:
        """
        Number of equations.

        Returns
        -------
        float
        """
        if self._num_equations is None:
            raise ValidationError(
                "The model is not solved yet. Please solve the model first to get the number of equations in the model instance."
            )

        return self._num_equations

    @property
    def num_infeasibilities(self) -> float | None:
        """
        Number of infeasibilities.

        Returns
        -------
        float | None
        """
        return self._num_infeasibilities

    @property
    def num_nonlinear_insts(self) -> float | None:
        """
        Number of nonlinear instructions.

        Returns
        -------
        float | None
        """
        return self._num_nonlinear_insts

    @property
    def num_nonlinear_zeros(self) -> float | None:
        """
        Number of nonlinear nonzeros.

        Returns
        -------
        float | None
        """
        return self._num_nonlinear_zeros

    @property
    def num_nonoptimalities(self) -> float | None:
        """
        Number of nonoptimalities.

        Returns
        -------
        float | None
        """
        return self._num_nonoptimalities

    @property
    def num_nonzeros(self) -> float:
        """
        Number of nonzero entries in the model coefficient matrix.

        Returns
        -------
        float
        """
        if self._num_nonzeros is None:
            raise ValidationError(
                "The model is not solved yet. Please solve the model first to get the number of nonzeros."
            )

        return self._num_nonzeros

    @property
    def num_mcp_redefinitions(self) -> float | None:
        """
        Number of MCP redefinitions.

        Returns
        -------
        float | None
        """
        return self._num_mcp_redefinitions

    @property
    def num_variables(self) -> float:
        """
        Number of variables in the model instance.

        Returns
        -------
        float
        """
        if self._num_variables is None:
            raise ValidationError(
                "The model is not solved yet. Please solve the model first to get the number of variables in the model instance."
            )

        return self._num_variables

    @property
    def num_bound_projections(self) -> float | None:
        """
        Number of bound projections during model generation.

        Returns
        -------
        float | None
        """
        return self._num_bound_projections

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
    def objective_value(self) -> float:
        """
        Objective function value

        Returns
        -------
        float
        """
        if self._objective_value is None:
            raise ValidationError(
                "The model is not solved yet. Please solve the model first before getting the objective value."
            )

        return self._objective_value

    @property
    def used_model_type(self) -> str | None:
        """
        Integer number that indicates the used model type.

        Returns
        -------
        str | None
        """
        return self._used_model_type

    @property
    def model_generation_time(self) -> float:
        """
        Time GAMS took to generate the model in wall-clock seconds.

        Returns
        -------
        float
        """
        if self._model_generation_time is None:
            raise ValidationError(
                "The model is not solved yet. Please solve the model first to get the model generation time."
            )

        return self._model_generation_time

    @property
    def solve_model_time(self) -> float:
        """
        Time the solver used to solve the model in seconds

        Returns
        -------
        float
        """
        if self._solve_model_time is None:
            raise ValidationError(
                "The model is not solved yet. Please solve the model first to get the solve time."
            )

        return self._solve_model_time

    @property
    def solve_status(self) -> SolveStatus:
        """
        Indicates the solver termination condition.

        Returns
        -------
        SolveStatus
        """
        if self._solve_status is None:
            raise ValidationError(
                "The model is not solved yet. Please solve the model first to get the solve status."
            )

        return self._solve_status

    @property
    def sum_infeasibilities(self) -> float | None:
        """
        Sum of infeasibilities.

        Returns
        -------
        float | None
        """
        return self._sum_infeasibilities

    @property
    def solver_version(self) -> float:
        """
        Solver version.

        Returns
        -------
        float
        """
        if self._solver_version is None:
            raise ValidationError(
                "The model is not solved yet. Please solve the model first to get the solver version."
            )
        return self._solver_version

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
            equation._should_unload_to_gams = False
            variable._should_unload_to_gams = False

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
            equation._should_unload_to_gams = False
            variable._should_unload_to_gams = False
            if equation.name not in (symbol.name for symbol in self.equations):
                self.equations.append(equation)

            return variable

        return assignment

    def _validate_scenario(
        self,
        scenario: GUSSScenarioDict | None,
        freeze_options: FreezeOptions | None = None,
    ) -> None:
        if scenario is None:
            return

        from gamspy._guss import GUSSScenarioDict

        if not isinstance(scenario, GUSSScenarioDict):
            raise ValidationError("`scenario` must be a GUSSScenarioDict.")

        if scenario.container != self.container:
            raise ValidationError(
                "`scenario` must belong to the same container as the model."
            )

        if freeze_options is not None or self._is_frozen:
            raise ValidationError(
                "`scenario` cannot be used with frozen solves or freeze_options."
            )

    def _generate_solve_string(self, scenario: GUSSScenarioDict | None = None) -> str:
        self._validate_scenario(scenario)
        solve_statement = [f"solve {self.name} using {self.problem}"]

        if self.sense == gp.Sense.FEASIBILITY:
            # Set sense as min or max for feasibility
            self.sense = gp.Sense("MIN")

        if self.problem not in (Problem.MCP, Problem.CNS, Problem.EMP):
            solve_statement.append(str(self.sense))

        if self._objective_variable is not None:
            solve_statement.append(self._objective_variable.gamsRepr())

        if scenario is not None:
            solve_statement.append(f"scenario {scenario.name}")

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

    def _create_model_attributes(self) -> list[str]:
        self.container._add_statement("$offListing")
        symbol_names = []
        for attr_name in MODEL_ATTRIBUTE_MAP:
            symbol_name = f"{ATTR_PREFIX}{attr_name}_{self._auto_id}"
            symbol_names.append(symbol_name)
            self.container._add_statement(f"Parameter {symbol_name};")

        self.container._add_statement("$onListing")

        return symbol_names

    def _assign_model_attributes(
        self, backend: Literal["engine", "neos", "local"] = "local"
    ) -> None:
        self.container._add_statement("$offListing")
        for symbol_name, attr_name in zip(
            self._attr_symbol_names, MODEL_ATTRIBUTE_MAP, strict=True
        ):
            self.container._add_statement(f"{symbol_name} = {self.name}.{attr_name};")

        attr_gdx_file = self._attr_gdx_file
        if backend == "engine":
            attr_gdx_file = os.path.basename(self._attr_gdx_file)
        elif backend == "neos":
            attr_gdx_file = "gams1.gdx"

        self.container._add_statement(
            f"execute_unload '{attr_gdx_file}', {','.join(self._attr_symbol_names)};"
        )
        self.container._add_statement("$onListing")

    def _update_model_attributes(self) -> None:
        records = gdxio._get_model_attr_records(
            self.container, self._attr_gdx_file, self._attr_symbol_names
        )
        for python_attr, symbol_name in zip(
            MODEL_ATTRIBUTE_MAP.values(), self._attr_symbol_names, strict=True
        ):
            data = records[symbol_name]

            if python_attr == "_status":
                setattr(self, python_attr, ModelStatus(data))
            elif python_attr == "_solve_status":
                status = SolveStatus(data)
                setattr(self, python_attr, status)

                if status in INTERRUPT_STATUS:
                    logger.warning(
                        f"The solve was interrupted! Solve status: {status.name}. "
                        "For further information, see https://gamspy.readthedocs.io/en/latest/reference/gamspy._model.html#gamspy.SolveStatus."
                    )
                elif status in ERROR_STATUS:
                    raise GamspyException(
                        f"Solve status: {status.name}. {ERROR_STATUS[status]}",
                        status.value,
                    )
            elif python_attr == "_used_model_type":
                setattr(self, python_attr, INT_TO_PROBLEM[int(data)])
            else:
                setattr(self, python_attr, data)

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
            infeas_rows = utils._calculate_infeasibilities(equation)
            infeas_dict[equation.name] = infeas_rows

            if equation._definition is not None:
                names = equation._definition._find_all_symbols()
                var_names = [
                    name
                    for name in names
                    if name in self.container._data
                    and isinstance(self.container._data[name], gp.Variable)
                ]

                for name in var_names:
                    variable: Variable = self.container._data[name]
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


        Convert the model into scalar GAMS format.


        >>> my_model.convert("tmp", gp.FileFormat.GAMS) # doctest: +SKIP


        Add conversion options


        >>> options = gp.ConvertOptions(GDXNames=False)
        >>> my_model.convert("jacobian", file_format=gp.FileFormat.GDXJacobian, options=options) # doctest: +SKIP

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

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, records=["item1", "item2"])
        >>> v = gp.Variable(m, domain=i)
        >>> z = gp.Variable(m)
        >>> e = gp.Equation(m, domain=i)
        >>> e[i] = v[i] * z >= 5
        >>> e2 = gp.Equation(m, domain=i)
        >>> e2[i] = v[i] - z >= 10
        >>> model = gp.Model(m, "test", equations=[e, e2], problem="NLP", sense="MIN", objective=z)
        >>> summary = model.solve(options=gp.Options(equation_listing_limit=10))
        >>> print(model.getEquationListing())
        e(item1)..  (0)*v(item1) + (0)*z =G= 5 ; (LHS = 0, INFES = 5 ****)
        e(item2)..  (0)*v(item2) + (0)*z =G= 5 ; (LHS = 0, INFES = 5 ****)
        e2(item1)..  v(item1) - z =G= 10 ; (LHS = 0, INFES = 10 ****)
        e2(item2)..  v(item2) - z =G= 10 ; (LHS = 0, INFES = 10 ****)

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

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, records=["item1", "item2"])
        >>> v = gp.Variable(m, domain=i)
        >>> z = gp.Variable(m)
        >>> e = gp.Equation(m, domain=i)
        >>> e[i] = v[i] * z >= 5
        >>> model = gp.Model(m, "test", equations=[e], problem="NLP", sense="MIN", objective=z)
        >>> summary = model.solve(options=gp.Options(variable_listing_limit=10))
        >>> print(model.getVariableListing())
        v(item1)
                        (.LO, .L, .UP, .M = -INF, 0, +INF, 0)
               (0)      e(item1)
        <BLANKLINE>
        v(item2)
                        (.LO, .L, .UP, .M = -INF, 0, +INF, 0)
               (0)      e(item2)
        <BLANKLINE>
        z
                        (.LO, .L, .UP, .M = -INF, 0, +INF, 0)
               (0)      e(item1)
               (0)      e(item2)
        <BLANKLINE>

        """
        if self._variable_names is None:
            raise ValidationError(
                "The model must be solved with `variable_listing_limit` option for this functionality to work."
            )

        listings = []
        for name in self._variable_names:
            variable = cast("Variable", self.container._data[name])
            listings.append(variable.getVariableListing())

        return "\n".join(listings)

    def interrupt(self) -> None:
        """
        Sends interrupt signal to the running job.

        Examples
        --------
        >>> import gamspy as gp
        >>> import threading
        >>> m = gp.Container()
        >>> # ... define model ...
        >>> model = gp.Model(m, "my_model", problem="LP", equations=[])
        >>> # In a separate thread or signal handler:
        >>> model.interrupt() # doctest: +SKIP

        """
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

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> p = gp.Parameter(m, "p", records=1)
        >>> v = gp.Variable(m, "v")
        >>> e = gp.Equation(m, "e", definition=v >= p)
        >>> model = gp.Model(m, "test", equations=[e], problem="LP", sense="MIN", objective=v)
        >>> model.freeze(modifiables=[p])
        >>> p.setRecords(2)  # Modify parameter without regenerating the model
        >>> summary = model.solve()
        >>> model.unfreeze()

        """
        self._is_frozen = True
        if options is None:
            options = self.container._options

        self.instance = ModelInstance(
            self.container, self, modifiables, options, self.container.output
        )

    def unfreeze(self) -> None:
        """
        Unfreezes the model

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> p = gp.Parameter(m, "p", records=1)
        >>> v = gp.Variable(m, "v")
        >>> e = gp.Equation(m, "e", definition=v >= p)
        >>> model = gp.Model(m, "test", equations=[e], problem="LP", sense="MIN", objective=v)
        >>> model.freeze(modifiables=[p])
        >>> p.setRecords(2)  # Modify parameter without regenerating the model
        >>> summary = model.solve()
        >>> model.unfreeze()

        """
        self._is_frozen = False
        self.instance.close_license_session()

    def solve(
        self,
        solver: str | None = None,
        options: Options | None = None,
        solver_options: dict | str | Path | None = None,
        freeze_options: FreezeOptions | None = None,
        output: TextIO | None = None,
        backend: Literal["local", "engine", "neos"] = "local",
        client: EngineClient | NeosClient | None = None,
        load_symbols: list[SymbolType] | None = None,
        scenario: GUSSScenarioDict | None = None,
    ) -> pd.DataFrame | None:
        """
        Solves the model with given options.
        https://gamspy.readthedocs.io/en/latest/user/basics/model.html

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
        output : TextIO, optional
            Output redirection target. Set this to sys.stdout to see the output in your terminal.
        backend : str, optional
            Backend to run on
        client : EngineClient, NeosClient, optional
            EngineClient to communicate with GAMS Engine or NEOS Client to communicate with NEOS Server
        load_symbols : list[Symbol], optional
            Specifies the symbols that need to be loaded. If not given, all symbols are loaded after solve.
        scenario : GUSSScenarioDict, optional
            GUSS scenario dictionary to use for this solve.

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
        >>> import sys
        >>> import os
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, description="items")
        >>> p = gp.Parameter(m, description="profits", domain=i)
        >>> w = gp.Parameter(m, description="weights", domain=i)
        >>> c = gp.Parameter(m, description="capacity")
        >>> x = gp.Variable(m, domain=i, type=gp.VariableType.BINARY)
        >>> capacity_restriction = gp.Equation(m, definition=gp.Sum(i, w[i] * x[i]) <= c)
        >>> knapsack = gp.Model(
        ...     m,
        ...     equations=m.getEquations(), # Automatically grabs all equations in the container
        ...     problem=gp.Problem.MIP,     # Mixed Integer Program
        ...     sense=gp.Sense.MAX,         # Maximize profit
        ...     objective=gp.Sum(i, p[i] * x[i]),
        ... )


        Basic usage


        >>> knapsack.solve() # doctest: +SKIP


        Change solver


        >>> knapsack.solve(solver="CPLEX") # doctest: +SKIP


        Redirect output


        >>> knapsack.solve(output=sys.stdout) # doctest: +SKIP


        Add generic solve options


        >>> knapsack.solve(options=gp.Options(iteration_limit=2)) # doctest: +SKIP


        Add solver-specific options


        >>> knapsack.solve(solver="CPLEX", solver_options={"preind": "off"}) # doctest: +SKIP


        Solve on your own machine


        >>> knapsack.solve() # doctest: +SKIP


        Solve on GAMS Engine


        >>> client = gp.EngineClient(
        ...    host=os.getenv("ENGINE_URL", "https://<host_link>"),
        ...    username=os.getenv("ENGINE_USER"),
        ...    password=os.getenv("ENGINE_PASSWORD"),
        ...    namespace=os.getenv("ENGINE_NAMESPACE"),
        ... )
        >>> knapsack.solve(backend="engine", client=client) # doctest: +SKIP


        Solve on NEOS Server


        >>> client = gp.NeosClient(
        ...    email=os.getenv("NEOS_EMAIL"),
        ...    username=os.getenv("NEOS_USER"),
        ...    password=os.getenv("NEOS_PASSWORD"),
        ... )
        >>> knapsack.solve(backend="neos", client=client) # doctest: +SKIP

        """
        if load_symbols is not None:
            warnings.warn(
                "`load_symbols` argument has no effect and will be deprecated in a future release.",
                category=DeprecationWarning,
                stacklevel=2,
            )

        if output is None:
            output = self.container.output

        if solver is None:
            solver = self._default_solver
        else:
            if not isinstance(solver, str):
                raise TypeError(
                    f"`solver` argument must be of type `str` but given `{type(solver)}`"
                )
            solver = solver.lower()

        if solver == "conopt":
            solver = "conopt4"

        self._validate_scenario(scenario, freeze_options)

        validation.validate_solver_args(
            self.container.system_directory,
            backend,
            solver,
            self.problem,
            options,
            output,
        )
        validation.validate_equations(self)

        if options is None:
            options = self.container._options

        if isinstance(solver_options, (str, Path)):
            solver_options = Path(solver_options)

        # Only for local until GAMS Engine and NEOS Server backends adopt the new GP_SolveLine option.
        if backend == "local":
            frame = inspect.currentframe()
            options._frame = frame.f_back if frame is not None else None

        if solver_options is not None:
            write_solver_options(self.container, solver, solver_options)

        if self._is_frozen:
            instance_options = FreezeOptions()

            if freeze_options is not None:
                instance_options = freeze_options

            summary = self.instance.solve(
                solver, instance_options, solver_options, output
            )
            return summary

        self._add_runtime_options(options, backend)
        self.container._add_statement(
            self._generate_solve_string(scenario=scenario) + ";\n"
        )
        if self.container._in_loop:
            self._assign_model_attributes()
        options._set_model_info(solver, self.problem, solver_options)

        if self.container._in_loop:
            return None

        runner = backend_factory(
            self.container,
            options,
            solver,
            solver_options,
            output,
            backend,
            client,
            self,
        )

        summary = runner.run()

        if client is not None and not client.is_blocking:
            return

        symbol_names = gdxio._get_symbol_names_from_gdx(
            self.container.system_directory, self.container._gdx_out
        )
        self.container._should_load_from_gams(symbol_names)

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

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> v = gp.Variable(m, "v")
        >>> e = gp.Equation(m, "e", definition= v == 5)
        >>> my_model = gp.Model(m, "my_model", problem="LP", equations=[e])
        >>> my_model.toGams("tmp")  # doctest: +SKIP
        ================================================================================
        GAMS (.gms) file has been generated under ...
        ================================================================================

        """
        if options is not None and not isinstance(options, gp.Options):
            raise ValidationError(
                f"`options` must be of type gp.Options of found {type(options)}"
            )

        path = Path(path)
        converter = GamsConverter(self, path, options, dump_gams_state)
        converter.convert()

    def toLatex(
        self,
        path: str | Path,
        rename: dict[str, str] | None = None,
        *,
        generate_pdf: bool = False,
    ) -> None:
        """
        Generates a latex file that contains the model definition under path/<model_name>.tex

        Parameters
        ----------
        path : str | Path
            Path to the directory which will contain the .tex file.
        rename: dict[str, str], optional
            A dictionary to rename symbols in the LaTeX file. Keys are GAMSPy symbol names
            and values are the names that will be used in the LaTeX file.
        generate_pdf: bool, False by default
            Generates a pdf file if it is set. Requires pdflatex to be installed.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> v = gp.Variable(m, "v")
        >>> e = gp.Equation(m, "e", definition= v == 5)
        >>> my_model = gp.Model(m, "my_model", problem="LP", equations=[e])
        >>> my_model.toLatex("tmp")  # doctest: +SKIP
        ================================================================================
        LaTeX (.tex) file has been generated under ...
        ================================================================================

        """
        path = Path(path)
        converter = LatexConverter(self, path, rename)
        converter.convert()

        if generate_pdf:
            converter.to_pdf()
