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
from enum import Enum
from typing import Iterable
from typing import Literal
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union

from gams import GamsOptions

import gamspy as gp
import gamspy._algebra.expression as expression
import gamspy._algebra.operation as operation
import gamspy.utils as utils
from gamspy._engine import EngineConfig
from gamspy._model_instance import ModelInstance
from gamspy._neos import NeosClient
from gamspy._options import _map_options
from gamspy.exceptions import GamspyException

if TYPE_CHECKING:
    from gamspy import Parameter, Variable, Equation, Container
    from gamspy._algebra.expression import Expression
    from gamspy._algebra.operation import Operation
    from gamspy._symbols.implicits import ImplicitParameter
    from gamspy._options import Options
    import pandas as pd


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
        return list(cls._value2member_map_.keys())

    def __str__(self) -> str:
        return self.value


class ModelStatus(Enum):
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
    "solveStat": "solver_status",
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
    problem : str
        Problem type (e.g. LP, NLP etc.)
    sense : "MIN", "MAX", or "FEASIBILITY", optional
        Minimize or maximize
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
        problem: str,
        equations: list[Equation] = [],
        sense: Literal["MIN", "MAX", "FEASIBILITY"] | None = None,
        objective: Variable | Expression | None = None,
        matches: dict | None = None,
        limited_variables: Iterable[Variable] | None = None,
    ):
        # check if the name is a reserved word
        name = utils._reserved_check(name)

        self.name = name
        self.container = container
        self.problem, self.sense = self._validate_model(
            equations, problem, sense
        )
        self.equations = equations
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
        self.solver_status = None
        self.solver_version = None

        self._attr_symbol_names = [
            f"{self._generate_prefix}{self.name}_{attr_name}"
            for attr_name in attribute_map.keys()
        ]

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
                raise GamspyException(
                    "Cannot set an objective when the sense is FEASIBILITY!"
                )

            if self.problem in [gp.Problem.CNS, gp.Problem.MCP]:
                raise GamspyException(
                    "Problem type cannot be CNS or MCP when the sense is"
                    " FEASIBILITY"
                )

            # Generate an objective variable
            variable = gp.Variable(
                self.container,
                f"{self._generate_prefix}{self.name}_objective_variable",
            )

            # Generate an equation
            equation = gp.Equation(
                self.container,
                f"{self._generate_prefix}{self.name}_equation",
            )

            equation[...] = variable == 0
            self.equations.append(equation)

            return variable

        if isinstance(
            assignment, (expression.Expression, operation.Operation)
        ):
            # Generate an objective variable
            variable = gp.Variable(
                self.container,
                f"{self._generate_prefix}{self.name}_objective_variable",
            )

            # Generate an equation
            equation = gp.Equation(
                self.container,
                f"{self._generate_prefix}{self.name}_equation",
            )

            # Sum((i,j),c[i,j]*x[i,j])->Sum((i,j),c[i,j]*x[i,j]) =e= var
            assignment = assignment == variable

            # equation .. Sum((i,j),c[i,j]*x[i,j]) =e= var
            equation[...] = assignment
            self.equations.append(equation)

            return variable

        return assignment

    def _create_attribute_symbols(self) -> None:
        for symbol_name in self._attr_symbol_names:
            _ = gp.Parameter(self.container, symbol_name)

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
                raise GamspyException(
                    "You need to provide a 'solver' to apply solver options."
                )

            solver_file_name = (
                self.container.workspace.working_directory
                + os.sep
                + f"{solver.lower()}.123"
            )

            with open(solver_file_name, "w") as solver_file:
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
            else:
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
            and not isinstance(equations, list)
            or any(
                not isinstance(equation, gp.Equation) for equation in equations
            )
        ):
            raise TypeError("equations must be list of Equation objects")

        return problem, sense

    def _append_solve_string(self) -> None:
        solve_string = f"solve {self.name} using {self.problem}"

        if self.sense:
            if self.sense == gp.Sense.FEASIBILITY:
                # Set sense as min or max for feasibility
                self.sense = gp.Sense.MIN

            solve_string += f" {self.sense}"

        if self._objective_variable:
            solve_string += f" {self._objective_variable.gamsRepr()}"

        self.container._unsaved_statements.append(solve_string + ";\n")

    def _assign_model_attributes(self) -> None:
        for attr_name in attribute_map.keys():
            symbol_name = f"{self._generate_prefix}{self.name}_{attr_name}"

            self.container._unsaved_statements.append(
                f"{symbol_name} = {self.name}.{attr_name};"
            )

    def _update_model_attributes(self) -> None:
        temp_container = gp.Container(
            system_directory=self.container.system_directory
        )
        temp_container.read(
            self.container._gdx_out,
            [
                f"{self._generate_prefix}{self.name}_{gams_attr}"
                for gams_attr in attribute_map.keys()
            ],
            cast_to_gamspy=False,
        )

        for gams_attr, python_attr in attribute_map.items():
            symbol_name = f"{self._generate_prefix}{self.name}_{gams_attr}"

            if python_attr == "status":
                setattr(
                    self,
                    python_attr,
                    ModelStatus(temp_container[symbol_name].toValue()),
                )
            else:
                setattr(
                    self,
                    python_attr,
                    temp_container[symbol_name].toValue(),
                )

    def _delete_autogenerated_symbols(self):
        for name in self._attr_symbol_names:
            if name in self.container.data.keys():
                del self.container.data[name]

    def interrupt(self) -> None:
        """
        Sends interrupt signal to the running job.

        Raises
        ------
        GamspyException
            If the job is not initialized
        """
        self.container._interrupt()

    def freeze(
        self,
        modifiables: list[Parameter | ImplicitParameter],
        freeze_options: dict | None = None,
    ) -> None:
        """
        Freezes all symbols except modifiable symbols.

        Parameters
        ----------
        modifiables : List[Union[Parameter, ImplicitParameter]]
        freeze_options : Optional[dict], optional
        """
        self.container._run(is_implicit=True)

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

    def _make_variable_and_equations_dirty(self):
        if self._objective_variable is not None:
            self._objective_variable._is_dirty = True

        for equation in self.equations:
            equation._is_dirty = True

            if equation._definition is not None:
                variables = equation._definition.find_variables()
                for name in variables:
                    self.container[name]._is_dirty = True

        if self._matches:
            for equation, variable in self._matches.items():
                equation._is_dirty = True
                variable._is_dirty = True

    def _verify_backend_args(
        self,
        backend: str,
        engine_config: Union[EngineConfig, None],
        neos_client: Union[NeosClient, None],
    ):
        if backend not in ["local", "engine", "neos"]:
            raise GamspyException(
                f"`{backend}` is not a valid backend. Possible backends:"
                " local, engine, and neos"
            )

        if backend == "neos" and not isinstance(neos_client, NeosClient):
            raise GamspyException(
                "`neos_client` must be provided to solve on NEOS Server"
            )

        if backend == "engine" and not isinstance(engine_config, EngineConfig):
            raise GamspyException(
                "`engine_config` must be provided to solve on GAMS Engine"
            )

    def solve(
        self,
        solver: str | None = None,
        options: Options | None = None,
        solver_options: dict | None = None,
        model_instance_options: dict | None = None,
        output: io.TextIOWrapper | None = None,
        backend: Literal["local", "engine", "neos"] = "local",
        engine_config: EngineConfig | None = None,
        neos_client: Optional[NeosClient] = None,
        create_log_file: bool = False,
    ) -> Union[pd.DataFrame, None]:
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
        engine_config : EngineConfig, optional
            GAMS Engine configuration

        Raises
        ------
        GamspyException
            In case engine_config is not provided for `engine` backend or
            neos_client is not provided for `neos` backend.
        ValueError
            In case problem is not in possible problem types
        ValueError
            In case sense is different than "MIN" or "MAX"
        """
        self._verify_backend_args(backend, engine_config, neos_client)

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
        self._create_attribute_symbols()
        self._assign_model_attributes()
        self._make_variable_and_equations_dirty()

        summary = self.container._run(
            gams_options, output, backend, engine_config, neos_client
        )

        if backend == "neos" and not neos_client.is_blocking:  # type: ignore
            return None

        self._update_model_attributes()
        self._delete_autogenerated_symbols()

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
                if equation not in self._matches.keys():
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

        model_str = f"Model {self.name} / {equations_str} /;"

        return model_str
