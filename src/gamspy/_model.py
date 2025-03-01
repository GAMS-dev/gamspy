from __future__ import annotations

import inspect
import io
import logging
import os
import threading
import uuid
from collections.abc import Iterable
from enum import Enum
from typing import TYPE_CHECKING

import gamspy as gp
import gamspy._algebra.expression as expression
import gamspy._algebra.operation as operation
import gamspy._miro as miro
import gamspy._symbols.implicits as implicits
import gamspy._validation as validation
import gamspy.utils as utils
from gamspy._backend.backend import backend_factory
from gamspy._convert import GamsConverter, LatexConverter
from gamspy._model_instance import ModelInstance
from gamspy._options import (
    EXECUTION_OPTIONS,
    MODEL_ATTR_OPTION_MAP,
    ModelInstanceOptions,
    Options,
)
from gamspy.exceptions import GamspyException, ValidationError

if TYPE_CHECKING:
    from typing import Literal

    import pandas as pd

    from gamspy import Container, Equation, Parameter, Variable
    from gamspy._algebra.expression import Expression
    from gamspy._algebra.operation import Operation
    from gamspy._backend.engine import EngineClient
    from gamspy._backend.neos import NeosClient
    from gamspy._symbols.implicits import ImplicitParameter, ImplicitVariable
    from gamspy._symbols.symbol import Symbol

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
    NLP = "NLP"
    QCP = "QCP"
    DNLP = "DNLP"
    MIP = "MIP"
    RMIP = "RMIP"
    MINLP = "MINLP"
    RMINLP = "RMINLP"
    MIQCP = "MIQCP"
    RMIQCP = "RMIQCP"
    MCP = "MCP"
    CNS = "CNS"
    MPEC = "MPEC"
    RMPEC = "RMPEC"
    EMP = "EMP"
    MPSGE = "MPSGE"

    @classmethod
    def values(cls):
        """Convenience function to return all values of enum"""
        return list(cls._value2member_map_.keys())

    def __str__(self) -> str:
        return self.value


class Sense(Enum):
    """An enumeration for sense types"""

    MIN = "MIN"
    MAX = "MAX"
    FEASIBILITY = "FEASIBILITY"

    @classmethod
    def values(cls):
        """Convenience function to return all values of enum"""
        return list(cls._value2member_map_.keys())

    def __str__(self) -> str:
        return self.value


class ModelStatus(Enum):
    """An enumeration for model status types"""

    OptimalGlobal = 1
    OptimalLocal = 2
    Unbounded = 3
    InfeasibleGlobal = 4
    InfeasibleLocal = 5
    InfeasibleIntermed = 6
    Feasible = 7
    Integer = 8
    NonIntegerIntermed = 9
    IntegerInfeasible = 10
    LicenseError = 11
    ErrorUnknown = 12
    ErrorNoSolution = 13
    NoSolutionReturned = 14
    SolvedUnique = 15
    Solved = 16
    SolvedSingular = 17
    UnboundedNoSolution = 18
    InfeasibleNoSolution = 19


class SolveStatus(Enum):
    """An enumeration for solve status types"""

    NormalCompletion = 1
    IterationInterrupt = 2
    ResourceInterrupt = 3
    TerminatedBySolver = 4
    EvaluationInterrupt = 5
    CapabilityError = 6
    LicenseError = 7
    UserInterrupt = 8
    SetupError = 9
    SolverError = 10
    InternalError = 11
    Skipped = 12
    SystemError = 13


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
    equations : Iterable[Equation]
        Iterable of Equation objects.
    problem : Problem | str, optional
        'LP', 'NLP', 'QCP', 'DNLP', 'MIP', 'RMIP', 'MINLP', 'RMINLP', 'MIQCP', 'RMIQCP', 'MCP', 'CNS', 'MPEC', 'RMPEC', 'EMP', or 'MPSGE',
        by default Problem.LP.
    sense : Sense | str, optional
        "MIN", "MAX", or "FEASIBILITY". By default, Sense.FEASIBILITY
    objective : Variable | Expression, optional
        Objective variable to minimize or maximize or objective itself.
    matches : dict[Equation, Variable]
        Equation - Variable matches for MCP models.
    limited_variables : Iterable[ImplicitVariable], optional
        Allows limiting the domain of variables used in a model.
    external_module: str, optional
        The name of the external module in which the external equations are implemented

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> v = gp.Variable(m, "v")
    >>> e = gp.Equation(m, "e", definition= v == 5)
    >>> my_model = gp.Model(m, "my_model", "LP", [e])

    """

    # Prefix for auto-generated symbols
    _generate_prefix = "autogenerated_"
    _autogen_symbols: list[Variable | Equation] = []

    def __init__(
        self,
        container: Container | None = None,
        name: str | None = None,
        problem: Problem | str = Problem.MIP,
        equations: Iterable[Equation] = [],
        sense: Sense | str = Sense.FEASIBILITY,
        objective: Variable | Expression | None = None,
        matches: dict[Equation, Variable] | None = None,
        limited_variables: Iterable[ImplicitVariable] | None = None,
        external_module: str | None = None,
    ):
        self._auto_id = "m" + str(uuid.uuid4()).replace("-", "_")

        if name is not None:
            name = validation.validate_name(name)
            self.name = validation.validate_model_name(name)
        else:
            self.name = self._auto_id

        if container is not None:
            self.container = container
        else:
            self.container = gp._ctx_managers[
                (os.getpid(), threading.get_native_id())
            ]

        assert self.container is not None
        self._matches = matches
        self.problem, self.sense = validation.validate_model(
            equations, problem, sense
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
            limited_variables, Iterable
        ):
            raise ValidationError(
                f"`limited_variables must be an Iterable of ImplicitVariable objects but found {limited_variables}`"
            )

        self._limited_variables = limited_variables

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
        self._num_domain_violations = None
        self._algorithm_time = None
        self._total_solve_time = None
        self._total_solver_time = None
        self._num_iterations = None
        self._marginals = None
        self._max_infeasibility = None
        self._mean_infeasibility = None
        self._status: ModelStatus | None = None
        self._num_nodes_used = None
        self._solve_number = None
        self._num_dependencies = None
        self._num_discrete_variables = None
        self._num_equations = None
        self._num_infeasibilities = None
        self._num_nonlinear_insts = None
        self._num_nonlinear_zeros = None
        self._num_nonoptimalities = None
        self._num_nonzeros = None
        self._num_mcp_redefinitions = None
        self._num_variables = None
        self._num_bound_projections = None
        self._objective_estimation = None
        self._objective_value = None
        self._used_model_type = None
        self._model_generation_time = None
        self._solve_model_time = None
        self._sum_infeasibilities = None
        self._solve_status: SolveStatus | None = None
        self._solver_version = None

        self.container.models.append(self)
        self.container._synch_with_gams()

    def _serialize(self) -> dict:
        info = {
            "name": self.name,
            "problem": str(self.problem),
            "sense": str(self.sense),
        }

        # equations
        equations = [equation.name for equation in self.equations]
        info["equations"] = equations

        # matches
        if self._matches is not None:
            matches = {
                key.name: value.name for key, value in self._matches.items()
            }
            info["_matches"] = matches

        # objective variable
        if self._objective_variable is not None:
            info["_objective_variable"] = self._objective_variable.name

        # attributes
        for attribute in ATTRIBUTE_MAP.values():
            if attribute in ["_status", "_solve_status"]:
                value = getattr(self, attribute, None)
                if value is not None:
                    value = value.value

                info[attribute] = value
            else:
                info[attribute] = getattr(self, attribute, None)

        return info

    def __repr__(self) -> str:
        return f"Model(name='{self.name}', problem='{str(self.problem)}', equations={self.equations}, sense='{str(self.sense)}', objective={self._objective_variable}, matches={self._matches}, limited_variables={self._limited_variables}"

    def __str__(self) -> str:
        return (
            f"Model {self.name}:\n  Problem Type: {self.problem}\n  Sense:"
            f" {self.sense}\n  Equations: {self.equations}"
        )

    @property
    def num_domain_violations(self) -> float | None:
        """
        Number of domain violations.

        Returns
        -------
        float | None
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
    def num_iterations(self) -> float | None:
        """
        Number of iterations used.

        Returns
        -------
        float | None
        """
        return self._num_iterations

    @property
    def marginals(self) -> float | None:
        """
        Indicates whether there are marginals.

        Returns
        -------
        float | None
        """
        return self._marginals

    @property
    def max_infeasibility(self) -> float | None:
        """
        Maximum of infeasibilities

        Returns
        -------
        float | None
        """
        return self._max_infeasibility

    @property
    def mean_infeasibility(self) -> float | None:
        """
        Mean of infeasibilities

        Returns
        -------
        float | None
        """
        return self._mean_infeasibility

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
    def num_equations(self) -> float | None:
        """
        Number of equations.

        Returns
        -------
        float | None
        """
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
    def num_nonzeros(self) -> float | None:
        """
        Number of nonzero entries in the model coefficient matrix.

        Returns
        -------
        float | None
        """
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
    def num_variables(self) -> float | None:
        """
        Number of variables.

        Returns
        -------
        float | None
        """
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
    def objective_value(self) -> float | None:
        """
        Objective function value

        Returns
        -------
        float | None
        """
        return self._objective_value

    @property
    def used_model_type(self) -> float | None:
        """
        Integer number that indicates the used model type.

        Returns
        -------
        float | None
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
    def solver_version(self) -> float | None:
        """
        Solver version.

        Returns
        -------
        float | None
        """
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
            self.container._add_statement(
                f"putclose {self._external_module_file};"
            )

        write_model_statement = value != self._external_module
        self._external_module = None
        self._external_module_file = None

        if value is not None:
            filename = "f" + str(uuid.uuid4()).replace("-", "_")
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
        self._autogen_symbols.append(variable.name)
        self._autogen_symbols.append(equation.name)

        return variable, equation

    def _set_objective_variable(
        self,
        assignment: None | Variable | Operation | Expression = None,
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
            (gp.Variable, expression.Expression, operation.Operation),
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

            if self.problem in [Problem.CNS, Problem.MCP, Problem.EMP]:
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

            if equation.name not in [symbol.name for symbol in self.equations]:
                self.equations.append(equation)

            return variable

        if isinstance(
            assignment, (expression.Expression, operation.Operation)
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
            if equation.name not in [symbol.name for symbol in self.equations]:
                self.equations.append(equation)

            return variable

        return assignment

    def _generate_solve_string(self) -> str:
        solve_string = f"solve {self.name} using {self.problem}"

        if self.sense == gp.Sense.FEASIBILITY:
            # Set sense as min or max for feasibility
            self.sense = gp.Sense("MIN")

        if self.problem not in [Problem.MCP, Problem.CNS, Problem.EMP]:
            solve_string += f" {self.sense}"

        if self._objective_variable is not None:
            solve_string += f" {self._objective_variable.gamsRepr()}"

        return solve_string + ";"

    def _add_runtime_options(
        self, options: Options, backend: str = "local"
    ) -> None:
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
                if backend == "engine" and key == "loadpoint":
                    value = os.path.relpath(
                        value, self.container.working_directory
                    )

                self.container._add_statement(
                    f"{EXECUTION_OPTIONS[key]} '{value}';\n"
                )

    def _append_solve_string(self) -> None:
        solve_string = self._generate_solve_string()
        self.container._add_statement(solve_string + "\n")

    def _create_model_attributes(self) -> None:
        self.container._add_statement("$offListing")
        for attr_name in ATTRIBUTE_MAP:
            symbol_name = f"{self._generate_prefix}{attr_name}_{self._auto_id}"
            _ = gp.Parameter._constructor_bypass(self.container, symbol_name)

            self.container._add_statement(
                f"{symbol_name} = {self.name}.{attr_name};"
            )
        self.container._add_statement("$onListing")

    def _update_model_attributes(self) -> None:
        container = self.container._temp_container
        gdx_handle = utils._open_gdx_file(
            self.container.system_directory, self.container._gdx_out
        )

        for gams_attr, python_attr in ATTRIBUTE_MAP.items():
            symbol_name = f"{self._generate_prefix}{gams_attr}_{self._auto_id}"
            data = utils._get_scalar_data(
                container._gams2np, gdx_handle, symbol_name
            )

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
            else:
                setattr(self, python_attr, data)

        utils._close_gdx_handle(gdx_handle)
        self.container._temp_container.data = {}

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
        self._is_frozen = True
        if options is None:
            options = Options()

        self.instance = ModelInstance(
            self.container, self, modifiables, options
        )

    def unfreeze(self) -> None:
        """Unfreezes the model"""
        self._is_frozen = False
        self.instance.close_license_session()

    def solve(
        self,
        solver: str | None = None,
        options: Options | None = None,
        solver_options: dict | None = None,
        model_instance_options: ModelInstanceOptions | None = None,
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
            GAMS options
        solver_options : dict, optional
            Solver options
        model_instance_options : ModelInstanceOptions, optional
            Model instance options
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
        >>> my_model = gp.Model(m, "my_model", "LP", [e], "max", v)
        >>> solved = my_model.solve()

        """
        if solver is None:
            solver = utils.getDefaultSolvers()[str(self.problem).upper()]

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

        # Only for local until GAMS Engine and NEOS Server backends adopt the new GP_SolveLine option.
        if solver == "local":
            frame = inspect.currentframe().f_back
            options._frame = frame

        if self._is_frozen:
            if model_instance_options is None:
                model_instance_options = ModelInstanceOptions()

            summary = self.instance.solve(
                solver, model_instance_options, solver_options, output
            )
            return summary

        self.container._add_statement(self.getDeclaration())
        self._add_runtime_options(options, backend)
        self._append_solve_string()
        self._create_model_attributes()
        options._set_solver_options(
            working_directory=self.container.working_directory,
            solver=solver,
            problem=self.problem,
            solver_options=solver_options,
        )

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
        >>> my_model = gp.Model(m, "my_model", "LP", [e])
        >>> my_model.getDeclaration()
        'Model my_model / e,my_model_objective /;'

        """
        equations = []
        for equation in self.equations:
            if self._matches:
                if equation not in self._matches:
                    equations.append(equation.name)
            else:
                equations.append(equation.name)

        equations_str = ",".join(equations)

        if self._matches:
            matches_str = ",".join(
                [
                    f"{equation.name}.{variable.name}"
                    for equation, variable in self._matches.items()
                ]
            )

            equations_str = (
                ",".join([equations_str, matches_str])
                if equations
                else matches_str
            )

        if self._external_module_file:
            equations_str = ",".join(
                [equations_str, self._external_module_file]
            )

        if self._limited_variables:
            limited_variables_str = ",".join(
                [variable.gamsRepr() for variable in self._limited_variables]
            )
            equations_str += "," + limited_variables_str

        model_str = f"Model {self.name}"
        if equations_str != "":
            model_str += f" / {equations_str} /"
        model_str += ";"

        return model_str

    def toGams(
        self,
        path: str,
        options: Options | None = None,
        dump_gams_state: bool = False,
    ) -> None:
        """
        Generates GAMS model under path/<model_name>.gms

        Parameters
        ----------
        path : str
            Path to the directory which will contain the GAMS model.
        """
        if options is not None and not isinstance(options, gp.Options):
            raise ValidationError(
                f"`options` must be of type gp.Options of found {type(options)}"
            )

        converter = GamsConverter(self, path, options, dump_gams_state)
        converter.convert()

    def toLatex(self, path: str, generate_pdf: bool = False) -> None:
        """
        Generates a latex file that contains the model definition under path/<model_name>.tex

        Parameters
        ----------
        path : str
            Path to the directory which will contain the .tex file.
        """
        converter = LatexConverter(self, path)
        converter.convert()

        if generate_pdf:
            converter.to_pdf()
