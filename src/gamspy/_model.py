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
import gamspy.utils as utils
import gamspy._symbols._implicits as implicits
import gamspy as gp
from gamspy.exceptions import GamspyException
from gams import (
    GamsOptions,
    GamsModifier,
    UpdateAction,
    VarType,
    EquType,
)
from gamspy._engine import EngineConfig
import gamspy._algebra.expression as expression
import gamspy._algebra.operation as operation
import math

from typing import (
    Dict,
    Iterable,
    Literal,
    List,
    Optional,
    Union,
    Tuple,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from gamspy import Parameter, Variable, Equation, Container
    from gamspy._symbols._implicits import ImplicitParameter
    from gamspy._algebra.expression import Expression
    from gamspy._algebra.operation import Operation


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


variable_map = {
    "binary": VarType.Binary,
    "integer": VarType.Integer,
    "positive": VarType.Positive,
    "negative": VarType.Negative,
    "free": VarType.Free,
    "sos1": VarType.SOS1,
    "sos2": VarType.SOS2,
    "semicont": VarType.SemiCont,
    "semiint": VarType.SemiInt,
}


equation_map = {
    "eq": EquType.E,
    "leq": EquType.L,
    "geq": EquType.G,
    "nonbinding": EquType.N,
    "external": EquType.X,
    "cone": EquType.C,
}

update_map = {
    "l": UpdateAction.Primal,
    "m": UpdateAction.Dual,
    "up": UpdateAction.Upper,
    "lo": UpdateAction.Lower,
    "fx": UpdateAction.Fixed,
}


class ModelInstance:
    def __init__(
        self,
        container: "Container",
        model: "Model",
        modifiables: List[Union["Parameter", "ImplicitParameter"]],
    ) -> None:
        self.modifiables = modifiables
        self.main_container = container

        self.checkpoint = self.main_container._restart_from
        self.instance = self.checkpoint.add_modelinstance()
        self.instantiate(model)

    def update_sync_db(self):
        self.main_container.write(self.instance.sync_db._gmd)

    def _create_modifiers(self):
        modifiers = []

        for modifiable in self.modifiables:
            if isinstance(modifiable, gp.Parameter):
                modifiers.append(
                    GamsModifier(
                        self.instance.sync_db.add_parameter(
                            modifiable.name,
                            modifiable.dimension,
                            modifiable.description,
                        )
                    )
                )
            elif isinstance(modifiable, implicits.ImplicitParameter):
                attribute = modifiable.name.split(".")[-1]
                update_action = update_map[attribute]

                if isinstance(modifiable.parent, gp.Variable):
                    sync_db_symbol = self.instance.sync_db.add_variable(
                        modifiable.parent.name,
                        modifiable.parent.dimension,
                        variable_map[modifiable.parent.type],
                    )

                elif isinstance(modifiable.parent, gp.Equation):
                    sync_db_symbol = self.instance.sync_db.add_equation(
                        modifiable.parent.name,
                        modifiable.parent.dimension,
                        equation_map[modifiable.parent.type],
                    )

                attr_name = "_".join(modifiable.name.split("."))

                attr_param = gp.Parameter(
                    self.main_container,
                    attr_name,
                    domain=modifiable.parent.domain,
                )

                def value_func(seed=None, size=None):
                    return math.inf

                attr_param.generateRecords(density=1.0, func=value_func)
                print(attr_param.records)

                data_symbol = self.instance.sync_db.add_parameter(
                    attr_name,
                    modifiable.parent.dimension,
                )

                modifiers.append(
                    GamsModifier(sync_db_symbol, update_action, data_symbol)
                )
            else:
                raise GamspyException(
                    f"Symbol type {type(modifiable)} cannot be modified in a"
                    " frozen solve"
                )

        return modifiers

    def instantiate(self, model: "Model"):
        solve_string = (
            f"{model.name} use"  # type: ignore
            f" {model.problem} {model.sense} {model._objective_variable.name}"
        )

        modifiers = self._create_modifiers()

        self.instance.instantiate(solve_string, modifiers)

    def solve(self):
        self.instance.solve()

    @property
    def model_status(self):
        return self.instance.model_status

    @property
    def solver_status(self):
        return self.instance.solver_status

    def update_main_container(self):
        instance_container = gp.Container(name="instance_container")
        instance_container.read(self.instance.sync_db._gmd)

        for name in instance_container.data.keys():
            if name in self.main_container.data.keys():
                self.main_container[name].setRecords(
                    instance_container[name].records
                )


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
        self.ref_container = container
        self._equations, self.problem, self.sense = self._validate_model(
            equations, problem, sense
        )
        self._objective_variable = self._set_objective_variable(objective)
        self._matches = matches
        self._limited_variables = limited_variables
        self.ref_container._addStatement(self)
        self._generate_attribute_symbols()

        # allow freezing
        self.is_frozen = False

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
                self.ref_container, f"dummy_objective{utils._getUniqueName()}"
            )

            # Create a dummy equation
            equation = gp.Equation(
                self.ref_container, f"dummy_equation{utils._getUniqueName()}"
            )

            # Sum((i,j),c[i,j]*x[i,j])->Sum((i,j),c[i,j]*x[i,j]) =e= var
            assignment = assignment == variable

            # equation .. Sum((i,j),c[i,j]*x[i,j]) =e= var
            equation.definition = assignment
            self._equations.append(equation)

            return variable

        return assignment

    def _generate_attribute_symbols(self) -> None:
        for attr_name in self._get_attribute_names():
            symbol_name = f"{self.name}_{attr_name}"
            self.ref_container._unsaved_statements[symbol_name] = (
                f"Scalar {symbol_name};"
            )

    def _get_attribute_names(self) -> Dict[str, str]:
        attributes = {
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

        return attributes

    def _prepare_gams_options(
        self,
        commandline_options: Optional[dict] = None,
        solver_options: Optional[dict] = None,
    ) -> GamsOptions:
        options = GamsOptions(self.ref_container.workspace)
        options.gdx = self.ref_container._gdx_path

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
                self.ref_container.workspace.working_directory
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

        self.ref_container._unsaved_statements[utils._getUniqueName()] = (
            solve_string + ";\n"
        )

    def _assign_model_attributes(self) -> None:
        for attr_name in self._get_attribute_names().keys():
            symbol_name = f"{self.name}_{attr_name}"

            self.ref_container._unsaved_statements[utils._getUniqueName()] = (
                f"{symbol_name} = {self.name}.{attr_name};"
            )

    def _update_model_attributes(self) -> None:
        gdxHandle = utils._openGdxFile(
            self.ref_container.system_directory, self.ref_container._gdx_path
        )

        for gams_attr, python_attr in self._get_attribute_names().items():
            symbol_name = f"{self.name}_{gams_attr}"

            records = utils._getSymbolData(
                self.ref_container._gams2np, gdxHandle, symbol_name
            )

            if python_attr == "status":
                setattr(
                    self,
                    python_attr,
                    ModelStatus(records.values[0][0]),
                )
            else:
                setattr(
                    self,
                    python_attr,
                    records.values[0][0],
                )

    def _remove_dummy_symbols(self):
        """
        Removes model attributes, dummy variable and dummy equation from
        the container
        """
        attribute_names = [
            f"{self.name}_{attr_name}"
            for attr_name in self._get_attribute_names()
        ]

        dummy_symbol_names = [
            name for name in self.ref_container.data.keys() if "dummy_" in name
        ] + attribute_names

        for name in dummy_symbol_names:
            del self.ref_container.data[name]

    @property
    def equations(self) -> Iterable["Equation"]:
        return self._equations

    @equations.setter
    def equations(self, new_equations) -> None:
        self._equations = new_equations

    def freeze(self, modifiables):
        self.ref_container._run()

        self.instance = ModelInstance(self.ref_container, self, modifiables)
        self.is_frozen = True

    def unfreeze(self):
        self.is_frozen = False

    def solve(
        self,
        commandline_options: Optional[Dict[str, str]] = None,
        solver_options: Optional[dict] = None,
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
        if not self.is_frozen:
            self._append_solve_string()
            self._assign_model_attributes()

            options = self._prepare_gams_options(
                commandline_options, solver_options
            )

            self.ref_container._run(options, output, backend, engine_config)

            self._update_model_attributes()
            self._remove_dummy_symbols()
        else:
            self.instance.update_sync_db()
            self.instance.solve()
            self.instance.update_main_container()

            self.status = ModelStatus(self.instance.model_status)

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
