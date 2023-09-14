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
from typing import Dict
from typing import Iterable
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union

from gams import (
    GamsOptions,
)

import gamspy as gp
import gamspy._algebra.expression as expression
import gamspy._algebra.operation as operation
import gamspy.utils as utils
from gamspy._engine import EngineConfig
from gamspy._model_instance import ModelInstance
from gamspy.exceptions import GamspyException

if TYPE_CHECKING:
    from gamspy import Parameter, Variable, Equation, Container
    from gamspy._algebra.expression import Expression
    from gamspy._algebra.operation import Operation
    from gamspy._symbols.implicits import ImplicitParameter


class Problem(Enum):
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
    MIN = "MIN"
    MAX = "MAX"

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


attribute_map = {
    "domUsd": "num_domain_violations",
    "etAlg": "algorithm_time",
    "etSolve": "solve_time",
    "etSolver": "solver_time",
    "iterUsd": "num_iterations",
    "marginals": "marginals",
    "maxInfes": "max_infeasibility",
    "meanInfes": "mean_infeasibility",
    "modelStat": "status",
    "nodUsd": "num_nodes_used",
    "numDepnd": "num_dependencies",
    "numDVar": "num_discrete_variables",
    "numInfes": "num_infeasibilities",
    "numNLIns": "num_nonlinear_insts",
    "numNLNZ": "num_nonlinear_zeros",
    "numNOpt": "num_nonoptimalities",
    "numNZ": "num_nonzeros",
    "numVar": "num_variables",
    "numVarProj": "num_bound_projections",
    "objEst": "objective_estimation",
    "objVal": "objective_value",
    "resGen": "model_generation_time",
    "sumInfes": "sum_infeasibilities",
    "sysVer": "solver_version",
}


class Model:
    """
    Represents a list of equations to be solved.
    https://www.gams.com/latest/docs/UG_ModelSolve.html#UG_ModelSolve_ModelStatement

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
    sense : "MIN" or "MAX", optional
        Minimize or maximize
    objective_variable : Variable, optional
        Objective variable to minimize or maximize
    limited_variables : Iterable, optional
        Allows limiting the domain of variables used in a model.

    Examples
    --------
    >>> transport = Model(m, "transport", equations=[cost,supply,demand])
    """

    def __init__(
        self,
        container: "Container",
        name: str,
        equations: Iterable["Equation"],
        problem: str,
        sense: Optional[Literal["MIN", "MAX"]] = None,
        objective: Optional[Union["Variable", "Expression"]] = None,
        matches: Optional[dict] = None,
        limited_variables: Optional[Iterable["Variable"]] = None,
    ):
        self.name = name
        self.container = container
        self._equations, self.problem, self.sense = self._validate_model(
            equations, problem, sense
        )
        self._objective_variable = self._set_objective_variable(objective)
        self._matches = matches
        self._limited_variables = limited_variables
        self.container._addStatement(self)
        self._generate_attribute_symbols()

        # allow freezing
        self._is_frozen = False

        # Attributes
        self.num_domain_violations = None
        self.algorithm_time = None
        self.solve_time = None
        self.solver_time = None
        self.num_iterations = None
        self.marginals = None
        self.max_infeasibility = None
        self.mean_infeasibility = None
        self.status: Optional[ModelStatus] = None
        self.num_nodes_used = None
        self.num_dependencies = None
        self.num_discrete_variables = None
        self.num_infeasibilities = None
        self.num_nonlinear_insts = None
        self.num_nonlinear_zeros = None
        self.num_nonoptimalities = None
        self.num_nonzeros = None
        self.num_variables = None
        self.num_bound_projections = None
        self.objective_estimation = None
        self.objective_value = None
        self.model_generation_time = None
        self.sum_infeasibilities = None
        self.solver_version = None
        self.solver_name = ""

    def _set_objective_variable(
        self,
        assignment: Optional[
            Union["Variable", "Operation", "Expression"]
        ] = None,
    ) -> Optional["Variable"]:
        """
        Returns objective variable. If the assignment is an Expression
        or an Operation (Sum, Product etc.), it creates a dummy variable
        and a dummy equation.

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

        if isinstance(
            assignment, (expression.Expression, operation.Operation)
        ):
            # Create a dummy objective variable
            variable = gp.Variable(
                self.container, f"dummy_objective{utils._getUniqueName()}"
            )

            # Create a dummy equation
            equation = gp.Equation(
                self.container, f"dummy_equation{utils._getUniqueName()}"
            )

            # Sum((i,j),c[i,j]*x[i,j])->Sum((i,j),c[i,j]*x[i,j]) =e= var
            assignment = assignment == variable

            # equation .. Sum((i,j),c[i,j]*x[i,j]) =e= var
            equation.expr = assignment
            self._equations.append(equation)

            return variable

        return assignment

    def _generate_attribute_symbols(self) -> None:
        for attr_name in attribute_map.keys():
            symbol_name = f"{self.name}_{attr_name}"
            self.container._unsaved_statements[symbol_name] = (
                f"Scalar {symbol_name};"
            )

    def _prepare_gams_options(
        self,
        commandline_options: Optional[dict] = None,
        solver_options: Optional[dict] = None,
    ) -> GamsOptions:
        options = GamsOptions(self.container.workspace)
        options.gdx = self.container._gdx_path

        if commandline_options:
            if not isinstance(commandline_options, dict):
                raise GamspyException("commandline_options must be a dict")

            for option, value in commandline_options.items():
                if option.lower() not in utils.COMMANDLINE_OPTIONS:
                    raise GamspyException(
                        f"Invalid commandline option: {option}"
                    )

                setattr(options, option, value)

        if solver_options:
            if (
                not commandline_options
                or "solver" not in commandline_options.keys()
            ):
                raise GamspyException(
                    "You need to provide 'solver' in commandline_options to"
                    " apply solver options."
                )

            solver_name = commandline_options["solver"]

            solver_file_name = (
                self.container.workspace.working_directory
                + os.sep
                + f"{solver_name}.123"
            )

            with open(solver_file_name, "w") as solver_file:
                for key, value in solver_options.items():
                    solver_file.write(f"{key} {value}\n")

            options.optfile = 123

        return options

    def _validate_model(self, equations, problem, sense=None) -> Tuple:
        if not isinstance(equations, list) or any(
            not isinstance(equation, gp.Equation) for equation in equations
        ):
            raise TypeError("equations must be list of Equation objects")

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

        return equations, problem, sense

    def _append_solve_string(self) -> None:
        solve_string = f"solve {self.name} using {self.problem}"

        if self.sense:
            solve_string += f" {self.sense}"

        if self._objective_variable:
            solve_string += f" {self._objective_variable.gamsRepr()}"

        self.container._unsaved_statements[utils._getUniqueName()] = (
            solve_string + ";\n"
        )

    def _assign_model_attributes(self) -> None:
        for attr_name in attribute_map.keys():
            symbol_name = f"{self.name}_{attr_name}"

            self.container._unsaved_statements[utils._getUniqueName()] = (
                f"{symbol_name} = {self.name}.{attr_name};"
            )

    def _update_model_attributes(self) -> None:
        temp_container = gp.Container(
            system_directory=self.container.system_directory
        )
        temp_container.read(
            self.container._gdx_path,
            [f"{self.name}_{gams_attr}" for gams_attr in attribute_map.keys()],
        )

        for gams_attr, python_attr in attribute_map.items():
            symbol_name = f"{self.name}_{gams_attr}"

            if python_attr == "status":
                setattr(
                    self,
                    python_attr,
                    ModelStatus(
                        temp_container[symbol_name].records.values[0][0]
                    ),
                )
            else:
                setattr(
                    self,
                    python_attr,
                    temp_container[symbol_name].records.values[0][0],
                )

    def _remove_dummy_symbols(self):
        """
        Removes model attributes, dummy variable and dummy equation from
        the container
        """
        attribute_names = [
            f"{self.name}_{attr_name}" for attr_name in attribute_map.keys()
        ]

        dummy_symbol_names = [
            name for name in self.container.data.keys() if "dummy_" in name
        ] + attribute_names

        for name in dummy_symbol_names:
            if name in self.container.data.keys():
                del self.container.data[name]

    @property
    def equations(self) -> Iterable["Equation"]:
        return self._equations

    @equations.setter
    def equations(self, new_equations) -> None:
        self._equations = new_equations

    def interrupt(self):
        self.container.interrupt()

    def freeze(
        self,
        modifiables: List[Union["Parameter", "ImplicitParameter"]],
        freeze_options: Optional[dict] = None,
    ):
        self.container._run()

        self.instance = ModelInstance(
            self.container, self, modifiables, freeze_options
        )
        self._is_frozen = True

    def unfreeze(self):
        for symbol in self.container.data.values():
            if hasattr(symbol, "_is_frozen") and symbol._is_frozen:
                symbol._is_frozen = False

    def solve(
        self,
        commandline_options: Optional[Dict[str, str]] = None,
        solver_options: Optional[dict] = None,
        model_instance_options: Optional[dict] = None,
        output: Optional[io.TextIOWrapper] = None,
        backend: Literal["local", "engine-one", "engine-sass"] = "local",
        engine_config: Optional["EngineConfig"] = None,
    ):
        """
        Generates the gams string, writes it to a file and runs it

        Parameters
        ----------
        commandline_options : dict, optional
        output : TextIOWrapper, optional

        Raises
        ------
        ValueError
            In case problem is not in possible problem types
        ValueError
            In case sense is different than "MIN" or "MAX"
        """
        if not self._is_frozen:
            options = self._prepare_gams_options(
                commandline_options, solver_options
            )

            self._append_solve_string()
            self._assign_model_attributes()

            self.container._run(options, output, backend, engine_config)

            self._update_model_attributes()
        else:
            self.instance.solve(model_instance_options, output)

        self._remove_dummy_symbols()

    def getStatement(self) -> str:
        """
        Statement of the Model definition

        Returns
        -------
        str
        """
        equations = []
        for equation in self._equations:
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
