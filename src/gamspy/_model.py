from __future__ import annotations

import io
import logging
import os
import platform
import signal
import uuid
from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING

from gams.core.gmd import gmdCloseLicenseSession

import gamspy as gp
import gamspy._algebra.expression as expression
import gamspy._algebra.operation as operation
import gamspy._symbols.implicits as implicits
import gamspy._validation as validation
import gamspy.utils as utils
from gamspy._backend.backend import backend_factory
from gamspy._model_instance import ModelInstance
from gamspy._symbols.symbol import Symbol
from gamspy.exceptions import GamspyException, ValidationError

if TYPE_CHECKING:
    from typing import Iterable, Literal

    import pandas as pd

    from gamspy import Container, Equation, Parameter, Variable
    from gamspy._algebra.expression import Expression
    from gamspy._algebra.operation import Operation
    from gamspy._backend.engine import EngineClient
    from gamspy._backend.neos import NeosClient
    from gamspy._options import ModelInstanceOptions, Options
    from gamspy._symbols.implicits import ImplicitParameter

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
attribute_map = {
    "domUsd": "num_domain_violations",
    "etAlg": "algorithm_time",
    "etSolve": "total_solve_time",
    "etSolver": "total_solver_time",
    "iterUsd": "num_iterations",
    "marginals": "marginals",
    "maxInfes": "max_infeasibility",
    "meanInfes": "mean_infeasibility",
    "modelStat": "status",
    "nodUsd": "num_nodes_used",
    "numDepnd": "num_dependencies",
    "numDVar": "num_discrete_variables",
    "numEqu": "num_equations",
    "numInfes": "num_infeasibilities",
    "numNLIns": "num_nonlinear_insts",
    "numNLNZ": "num_nonlinear_zeros",
    "numNOpt": "num_nonoptimalities",
    "numNZ": "num_nonzeros",
    "numRedef": "num_mcp_redefinitions",
    "numVar": "num_variables",
    "numVarProj": "num_bound_projections",
    "objEst": "objective_estimation",
    "objVal": "objective_value",
    "procUsed": "used_model_type",
    "resGen": "model_generation_time",
    "resUsd": "solve_model_time",
    "solveStat": "solve_status",
    "sumInfes": "sum_infeasibilities",
    "sysVer": "solver_version",
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
    equations : Sequence[Equation]
        Sequence of Equation objects.
    problem : Problem or str, optional
        'LP', 'NLP', 'QCP', 'DNLP', 'MIP', 'RMIP', 'MINLP', 'RMINLP', 'MIQCP', 'RMIQCP', 'MCP', 'CNS', 'MPEC', 'RMPEC', 'EMP', or 'MPSGE',
        by default Problem.LP.
    sense : Sense, optional
        "MIN", "MAX", or "FEASIBILITY".
    objective : Variable | Expression, optional
        Objective variable to minimize or maximize or objective itself.
    matches : dict[Equation, Variable]
        Equation - Variable matches for MCP models.
    limited_variables : Iterable, optional
        Allows limiting the domain of variables used in a model.

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

    def __init__(
        self,
        container: Container,
        name: str | None = None,
        problem: Problem | str = Problem.LP,
        equations: Sequence[Equation] = [],
        sense: Sense | str | None = None,
        objective: Variable | Expression | None = None,
        matches: dict[Equation, Variable] | None = None,
        limited_variables: Iterable[Variable] | None = None,
    ):
        self._auto_id = str(uuid.uuid4()).replace("-", "_")

        if name is not None:
            name = validation.validate_name(name)
            self.name = validation.validate_model_name(name)
        else:
            self.name = self._auto_id

        self.container = container
        self._matches = matches
        self.problem, self.sense = self._validate_model(
            equations, problem, sense
        )
        self.equations = list(equations)
        self._objective_variable = self._set_objective_variable(objective)
        self._limited_variables = limited_variables

        if not self.equations and not self._matches:
            raise ValidationError("Model requires at least one equation.")

        self.container._add_statement(self)

        # allow freezing
        self._is_frozen = False

        # Attributes
        self.num_domain_violations = None
        self.algorithm_time = None
        self.total_solve_time = None
        self.total_solver_time = None
        self.num_iterations = None
        self.marginals = None
        self.max_infeasibility = None
        self.mean_infeasibility = None
        self.status: ModelStatus | None = None
        self.num_nodes_used = None
        self.num_dependencies = None
        self.num_discrete_variables = None
        self.num_infeasibilities = None
        self.num_nonlinear_insts = None
        self.num_nonlinear_zeros = None
        self.num_nonoptimalities = None
        self.num_nonzeros = None
        self.num_mcp_redefinitions = None
        self.num_variables = None
        self.num_bound_projections = None
        self.objective_estimation = None
        self.objective_value = None
        self.used_model_type = None
        self.model_generation_time = None
        self.solve_model_time = None
        self.sum_infeasibilities = None
        self.solve_status: SolveStatus | None = None
        self.solver_version = None

        self._infeasibility_tolerance: float | None = None

        self.container._synch_with_gams()

    def __repr__(self) -> str:
        return f"<Model `{self.name}` ({hex(id(self))})>"

    def __str__(self) -> str:
        return (
            f"Model {self.name}:\n  Problem Type: {self.problem}\n  Sense:"
            f" {self.sense}\n  Equations: {self.equations}"
        )

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

            if self.problem in [gp.Problem.CNS, gp.Problem.MCP]:
                raise ValidationError(
                    "Problem type cannot be CNS or MCP when the sense is"
                    " FEASIBILITY"
                )

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
            self.equations.append(equation)

            return variable

        return assignment

    def _validate_model(
        self,
        equations: Sequence[Equation],
        problem: Problem | str,
        sense: str | Sense | None = None,
    ) -> tuple[Problem, Sense | None]:
        if isinstance(problem, str):
            if problem.upper() not in gp.Problem.values():
                raise ValueError(
                    f"Allowed problem types: {gp.Problem.values()} but found"
                    f" {problem}."
                )

            problem = gp.Problem(problem.upper())

        if isinstance(sense, str):
            if sense.upper() not in gp.Sense.values():
                raise ValueError(
                    f"Allowed sense values: {gp.Sense.values()} but found"
                    f" {sense}."
                )

            sense = gp.Sense(sense.upper())

        if (
            problem not in [Problem.CNS, Problem.MCP]
            and not isinstance(equations, Sequence)
            or any(
                not isinstance(equation, gp.Equation) for equation in equations
            )
        ):
            raise TypeError("equations must be list of Equation objects")

        return problem, sense  # type: ignore

    def _generate_solve_string(self) -> str:
        solve_string = f"solve {self.name} using {self.problem}"

        if self.sense:
            if self.sense == gp.Sense.FEASIBILITY:
                # Set sense as min or max for feasibility
                self.sense = gp.Sense.MIN

            solve_string += f" {self.sense}"

        if self._objective_variable is not None:
            solve_string += f" {self._objective_variable.gamsRepr()}"

        return solve_string + ";"

    def _append_solve_string(self) -> None:
        solve_string = self._generate_solve_string()
        self.container._add_statement(solve_string + "\n")

    def _create_model_attributes(self) -> None:
        self.container._add_statement("$offListing")
        for attr_name in attribute_map:
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

        for gams_attr, python_attr in attribute_map.items():
            symbol_name = f"{self._generate_prefix}{gams_attr}_{self._auto_id}"
            data = utils._get_scalar_data(
                container._gams2np, gdx_handle, symbol_name
            )

            if python_attr == "status":
                setattr(self, python_attr, ModelStatus(data))
            elif python_attr == "solve_status":
                status = SolveStatus(data)
                setattr(self, python_attr, status)

                if status in INTERRUPT_STATUS:
                    logger.warn(
                        f"The solve was interrupted! Solve status: {status.name}. "
                        "For further information, see https://www.gams.com/latest/docs/UG_GAMSOutput.html#UG_GAMSOutput_SolverStatus."
                    )
                elif status in ERROR_STATUS:
                    raise GamspyException(
                        f"The model `{self.name}` was not solved successfully!"
                        f" Solve status: {status.name}. {ERROR_STATUS[status]}",
                        status.value,
                    )
            else:
                setattr(self, python_attr, data)

        utils._close_gdx_handle(gdx_handle)
        self.container._temp_container.data = {}

    def compute_infeasibilities(self) -> dict[str, pd.DataFrame]:
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
        >>> infeasibilities = my_model.compute_infeasibilities()
        >>> infeasibilities["s"].infeasibility.item()
        320.0

        """
        infeas_dict = {}

        for equation in self.equations:
            if equation.records is None:
                continue

            infeas_rows = utils._calculate_infeasibilities(equation)
            infeas_dict[equation.name] = infeas_rows

        return infeas_dict

    @property
    def infeasibility_tolerance(self) -> float | None:
        """
        This option sets the tolerance for marking an equation infeasible in
        the equation listing. By default, 1.0e-13.

        Returns
        -------
        float | None
        """
        return self._infeasibility_tolerance

    @infeasibility_tolerance.setter
    def infeasibility_tolerance(self, value: float):
        self.container._add_statement(f"{self.name}.tolInfRep = {value};")
        self._infeasibility_tolerance = value

    def interrupt(self) -> None:
        """
        Sends interrupt signal to the running job.

        Raises
        ------
        ValidationError
            If the job is not initialized
        """
        if platform.system() == "Windows":
            self.container._process.send_signal(signal.SIGTERM)
        else:
            self.container._process.send_signal(signal.SIGINT)

        self.container._stop_socket()

    def freeze(
        self,
        modifiables: list[Parameter | ImplicitParameter],
        options: Options | None = None,
    ) -> None:
        """
        Freezes all symbols except modifiable symbols.

        Parameters
        ----------
        modifiables : List[Parameter | ImplicitParameter]
        freeze_options : dict, optional

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> a = gp.Parameter(m, name="a", records=10)
        >>> x = gp.Variable(m, name="x")
        >>> e = gp.Equation(m, name="e", definition= x <= a)
        >>> my_model = gp.Model(m, name="my_model", equations=m.getEquations(), problem="LP", sense="max", objective=x)
        >>> solved = my_model.solve()
        >>> float(x.toValue())
        10.0
        >>> my_model.freeze(modifiables=[a])
        >>> a.setRecords(35)
        >>> solved = my_model.solve()
        >>> float(x.toValue())
        35.0

        """
        self._is_frozen = True
        self.instance = ModelInstance(
            self.container, self, modifiables, options
        )

    def unfreeze(self) -> None:
        """Unfreezes the model"""
        self._is_frozen = False
        gmdCloseLicenseSession(self.instance.instance.sync_db._gmd)

    def solve(
        self,
        solver: str | None = None,
        options: Options | None = None,
        solver_options: dict | None = None,
        model_instance_options: ModelInstanceOptions | dict | None = None,
        output: io.TextIOWrapper | None = None,
        backend: Literal["local", "engine", "neos"] = "local",
        client: EngineClient | NeosClient | None = None,
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
        model_instance_options : optional
            Model instance options
        output : TextIOWrapper, optional
            Output redirection target
        backend : str, optional
            Backend to run on
        client : EngineClient, NeosClient, optional
            EngineClient to communicate with GAMS Engine or NEOS Client to communicate with NEOS Server

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
        validation.validate_solver_args(
            self.container.system_directory,
            solver,
            self.problem,
            options,
            output,
        )
        validation.validate_model(self)

        if options is None:
            options = (
                self.container._options
                if self.container._options
                else gp.Options()
            )

        options._set_solver_options(
            self.container.working_directory,
            solver=solver,
            problem=self.problem,
            solver_options=solver_options,
        )

        if self._is_frozen:
            self.instance.solve(solver, model_instance_options, output)
            return None

        self._append_solve_string()
        self._create_model_attributes()

        runner = backend_factory(
            self.container,
            options,
            output,
            backend,
            client,
            self,
        )

        summary = runner.run()

        if IS_MIRO_INIT:
            self.container._write_default_gdx_miro()

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
        'Model my_model / e /;'

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

    def toGams(self, path: str, skip_endogenous_records: bool = False) -> None:
        """
        Generates GAMS model under path/<model_name>.gms

        Parameters
        ----------
        path : str
            Path to the directory which will contain the GAMS model.
        """
        os.makedirs(path, exist_ok=True)

        def sort_names(name):
            PRECEDENCE = {
                gp.Set: 1,
                gp.Alias: 1,
                gp.Parameter: 3,
                gp.Variable: 4,
                gp.Equation: 5,
            }

            symbol = self.container[name]
            precedence = PRECEDENCE[type(symbol)]

            if isinstance(symbol, gp.Set) and any(
                not isinstance(elem, str) for elem in symbol.domain
            ):
                precedence = 2

            return precedence

        all_symbols = []
        definitions = []
        for equation in self.equations:
            definitions.append(equation._definition.getDeclaration())
            symbols = equation._definition._find_all_symbols()

            for symbol in symbols:
                if symbol not in all_symbols:
                    all_symbols.append(symbol)

        if self._matches:
            for equation, variable in self._matches.items():
                if (
                    equation.name not in all_symbols
                    and not skip_endogenous_records
                ):
                    symbols = equation._definition._find_all_symbols()
                    for symbol in symbols:
                        if symbol not in all_symbols:
                            all_symbols.append(symbol)

                    if equation.name not in all_symbols:
                        all_symbols.append(equation.name)

                    definitions.append(equation._definition.getDeclaration())

                if (
                    variable.name not in all_symbols
                    and not skip_endogenous_records
                ):
                    for elem in variable.domain:
                        if not isinstance(elem, Symbol):
                            continue

                        elem_path = validation.get_domain_path(elem)
                        for name in elem_path:
                            if name not in all_symbols:
                                all_symbols.append(name)

                    if variable.name not in all_symbols:
                        all_symbols.append(variable.name)

        all_needed_symbols = sorted(all_symbols, key=sort_names)
        gdx_path = os.path.join(path, self.name + "_data.gdx")
        self.container.write(gdx_path, all_needed_symbols)

        strings = [
            self.container[name].getDeclaration()
            for name in all_needed_symbols
        ]
        strings.append(f"$gdxLoadAll {os.path.abspath(gdx_path)}")
        strings += definitions
        strings.append(self.getDeclaration())
        solve_string = self._generate_solve_string()
        strings.append(solve_string)

        gams_string = "\n".join(strings)
        with open(os.path.join(path, self.name + ".gms"), "w") as file:
            file.write(gams_string)

        logger.info(
            f'GAMS model has been generated under {os.path.join(path, self.name + ".gms")}'
        )
