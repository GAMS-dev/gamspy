#
# GAMS - General Algebraic Modeling System Python API
#
# Copyright (c) 2023 GAMS Development Corp. <support@gams.com>
# Copyright (c) 2023 GAMS Software GmbH <support@gams.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
from __future__ import annotations

import io
import os
import uuid
from enum import Enum
from typing import TYPE_CHECKING, Iterable, Literal

from gams import GamsOptions

import gamspy as gp
import gamspy._algebra.expression as expression
import gamspy._algebra.operation as operation
import gamspy._validation as validation
import gamspy.utils as utils
from gamspy._backend.backend import backend_factory
from gamspy._model_instance import ModelInstance
from gamspy._options import _map_options
from gamspy.exceptions import GamspyException, ValidationError

if TYPE_CHECKING:
    import pandas as pd

    from gamspy import Container, Equation, Parameter, Variable
    from gamspy._algebra.expression import Expression
    from gamspy._algebra.operation import Operation
    from gamspy._backend.engine import EngineClient
    from gamspy._backend.neos import NeosClient
    from gamspy._options import ModelInstanceOptions, Options
    from gamspy._symbols.implicits import ImplicitParameter

IS_MIRO_INIT = os.getenv("MIRO", False)


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
    EvalError = 5
    CapabilityError = 6
    LicenseError = 7
    UserInterrupt = 8
    SetupError = 9
    SolverError = 10
    InternalError = 11
    Skipped = 12
    SystemError = 13


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
        The container that the model belongs to
    name : str
        Name of the model
    equations : str | Iterable
        List of Equation objects or str. ``all`` as a string represents
        all the equations specified before the creation of this model
    problem : Problem
        'LP', 'NLP', 'QCP', 'DNLP', 'MIP', 'RMIP', 'MINLP', 'RMINLP', 'MIQCP', 'RMIQCP', 'MCP', 'CNS', 'MPEC', 'RMPEC', 'EMP', or 'MPSGE'
    sense : Sense, optional
        "MIN", "MAX", or "FEASIBILITY"
    objective : Variable | Expression, optional
        Objective variable to minimize or maximize or objective itself
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
        name: str,
        problem: Problem | str,
        equations: list[Equation] = [],
        sense: Sense | str | None = None,
        objective: Variable | Expression | None = None,
        matches: dict | None = None,
        limited_variables: Iterable[Variable] | None = None,
    ):
        # check if the name is a reserved word
        name = validation.validate_name(name)

        self.name = validation.validate_model_name(name)
        self.container = container
        self.problem, self.sense = self._validate_model(
            equations, problem, sense
        )
        self.equations = list(equations)
        self._objective_variable = self._set_objective_variable(objective)
        self._matches = matches
        self._limited_variables = limited_variables
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

        self.container._run()

    def __repr__(self) -> str:
        return f"<Model `{self.name}` ({hex(id(self))})>"

    def __str__(self) -> str:
        return (
            f"Model {self.name}:\n  Problem Type: {self.problem}\n  Sense:"
            f" {self.sense}\n  Equations: {self.equations}"
        )

    def _generate_obj_var_and_equation(self):
        auto_id = str(uuid.uuid4()).replace("-", "_")
        variable = gp.Variable(
            self.container, f"{Model._generate_prefix}{auto_id}_variable"
        )
        equation = gp.Equation(
            self.container, f"{Model._generate_prefix}{auto_id}_equation"
        )

        return variable, equation

    def _set_objective_variable(
        self,
        assignment: None | (Variable | Operation | Expression) = None,
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
            equation[...] = variable == 0
            equation.modified = False
            equation._is_dirty = False
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
            equation[...] = assignment
            equation.modified = False
            equation._is_dirty = False
            variable.modified = False
            self.equations.append(equation)

            return variable

        return assignment

    def _prepare_gams_options(
        self,
        solver: str | None = None,
        backend: str = "local",
        options: Options | None = None,
        solver_options: dict | None = None,
        output: io.TextIOWrapper | None = None,
        create_log_file: bool = False,
    ) -> GamsOptions:
        gams_options = _map_options(
            self.container.workspace,
            backend=backend,
            options=options,
            global_options=self.container._options,
            is_seedable=False,
            output=output,
            create_log_file=create_log_file,
        )

        if solver:
            gams_options.all_model_types = solver

        if solver_options:
            if solver is None:
                raise ValidationError(
                    "You need to provide a 'solver' to apply solver options."
                )

            solver_file_name = (
                self.container.workspace.working_directory
                + os.sep
                + f"{solver.lower()}.123"
            )

            with open(solver_file_name, "w", encoding="utf-8") as solver_file:
                for key, value in solver_options.items():
                    solver_file.write(f"{key} {value}\n")

            gams_options.optfile = 123

        return gams_options

    def _validate_model(self, equations, problem, sense=None) -> tuple:
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
            and not isinstance(equations, (list, tuple))
            or any(
                not isinstance(equation, gp.Equation) for equation in equations
            )
        ):
            raise TypeError(
                "equations must be list or tuple of Equation objects"
            )

        return problem, sense

    def _append_solve_string(self) -> None:
        solve_string = f"solve {self.name} using {self.problem}"

        if self.sense:
            if self.sense == gp.Sense.FEASIBILITY:
                # Set sense as min or max for feasibility
                self.sense = gp.Sense.MIN

            solve_string += f" {self.sense}"

        if self._objective_variable is not None:
            solve_string += f" {self._objective_variable.gamsRepr()}"

        self.container._add_statement(solve_string + ";\n")

    def _create_model_attributes(self) -> None:
        for attr_name in attribute_map:
            symbol_name = f"{self._generate_prefix}{self.name}_{attr_name}"
            _ = gp.Parameter._constructor_bypass(self.container, symbol_name)

            self.container._add_statement(
                f"{symbol_name} = {self.name}.{attr_name};"
            )

    def _update_model_attributes(self) -> None:
        container = self.container._temp_container
        gdx_handle = utils._open_gdx_file(
            self.container.system_directory, self.container._gdx_out
        )

        for gams_attr, python_attr in attribute_map.items():
            symbol_name = f"{self._generate_prefix}{self.name}_{gams_attr}"
            data = utils._get_scalar_data(
                container._gams2np, gdx_handle, symbol_name
            )

            if python_attr == "status":
                setattr(self, python_attr, ModelStatus(data))
            elif python_attr == "solve_status":
                status = SolveStatus(data)
                setattr(self, python_attr, status)

                if status != SolveStatus.NormalCompletion:
                    raise GamspyException(
                        f"The model ({self.name}) was not solved successfully!"
                        f" Solve status: {status.name}",
                        status.value,
                    )
            else:
                setattr(self, python_attr, data)

        utils._close_gdx_handle(gdx_handle)
        self.container._temp_container.data = {}

    def _make_variable_and_equations_dirty(self):
        if (
            self._objective_variable is not None
            and not self._objective_variable.name.startswith(
                Model._generate_prefix
            )
        ):
            self._objective_variable._is_dirty = True

        for equation in self.equations:
            if not equation.name.startswith(Model._generate_prefix):
                equation._is_dirty = True

            if equation._assignment is not None:
                variables = equation._assignment._find_variables()
                for name in variables:
                    if not name.startswith(Model._generate_prefix):
                        self.container[name]._is_dirty = True

        if self._matches:
            for equation, variable in self._matches.items():
                equation._is_dirty = True
                variable._is_dirty = True

    def compute_infeasibilities(self) -> dict[str, pd.DataFrame]:
        """
        Computes infeasabilities for all equations of the model

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary of infeasibilities where equation names are keys and
            infeasibilities are values
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
        self.container._job.interrupt()

    def freeze(
        self,
        modifiables: list[Parameter | ImplicitParameter],
        freeze_options: dict | None = None,
    ) -> None:
        """
        Freezes all symbols except modifiable symbols.

        Parameters
        ----------
        modifiables : List[Parameter | ImplicitParameter]
        freeze_options : dict, optional
        """
        self.container._run()

        self.instance = ModelInstance(
            self.container, self, modifiables, freeze_options
        )
        self._is_frozen = True

    def unfreeze(self) -> None:
        """Unfreezes all symbols"""
        for symbol in self.container.data.values():
            if hasattr(symbol, "_is_frozen") and symbol._is_frozen:
                symbol._is_frozen = False

        self._is_frozen = False

    def solve(
        self,
        solver: str | None = None,
        options: Options | None = None,
        solver_options: dict | None = None,
        model_instance_options: ModelInstanceOptions | dict | None = None,
        output: io.TextIOWrapper | None = None,
        backend: Literal["local", "engine", "neos"] = "local",
        client: EngineClient | NeosClient | None = None,
        create_log_file: bool = False,
    ) -> pd.DataFrame | None:
        """
        Generates the gams string, writes it to a file and runs it

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
        create_log_file : bool
            Allows creating a log file

        Raises
        ------
        ValidationError
            In case engine_config is not provided for `engine` backend or
            neos_client is not provided for `neos` backend.
        ValueError
            In case problem is not in possible problem types
        ValueError
            In case sense is different than "MIN" or "MAX"
        """
        validation.validate_solver_args(solver, options, output)

        if self._is_frozen:
            self.instance.solve(model_instance_options, output)
            return None

        gams_options = self._prepare_gams_options(
            solver,
            backend,
            options,
            solver_options,
            output=output,
            create_log_file=create_log_file,
        )

        self._append_solve_string()
        self._create_model_attributes()
        self._make_variable_and_equations_dirty()

        runner = backend_factory(
            self.container,
            gams_options,
            output,
            backend,
            client,
            self,
        )

        summary = runner.solve()

        if IS_MIRO_INIT:
            self.container._write_default_gdx_miro()

        return summary

    def getStatement(self) -> str:
        """
        Statement of the Model definition

        Returns
        -------
        str
        """
        equations = []
        for equation in self.equations:
            if self._matches:
                if equation not in self._matches:
                    equations.append(equation.gamsRepr())
            else:
                equations.append(equation.gamsRepr())

        equations_str = ",".join(equations)

        if self._matches:
            matches_str = ",".join(
                [
                    f"{equation.gamsRepr()}.{variable.gamsRepr()}"
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
