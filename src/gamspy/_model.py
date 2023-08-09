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
import os
import io
from enum import Enum
import gamspy.utils as utils
import gamspy as gp
from gams import GamsJob, GamsOptions
from typing import Dict, List, Literal, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from gamspy import Variable, Equation, Container


class ModelStatus(Enum):
    # Optimal solution achieved
    OptimalGlobal = 1
    # Local optimal solution achieved
    OptimalLocal = 2
    # Unbounded model found
    Unbounded = 3
    # Infeasible model found
    InfeasibleGlobal = 4
    # Locally infeasible model found
    InfeasibleLocal = 5
    # Solver terminated early and model was still infeasible
    InfeasibleIntermed = 6
    # Solver terminated early and model was feasible but not yet optimal
    Feasible = 7
    # Integer solution found
    Integer = 8
    # Solver terminated early with a non integer solution found
    NonIntegerIntermed = 9
    # No feasible integer solution could be found
    IntegerInfeasible = 10
    # Licensing problem
    LicenseError = 11
    # Error - No cause known
    ErrorUnknown = 12
    # Error - No solution attained
    ErrorNoSolution = 13
    # No solution returned
    NoSolutionReturned = 14
    # Unique solution in a CNS models
    SolvedUnique = 15
    # Feasible solution in a CNS models
    Solved = 16
    # Singular in a CNS models
    SolvedSingular = 17
    # Unbounded - no solution
    UnboundedNoSolution = 18
    # Infeasible - no solution
    InfeasibleNoSolution = 19


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
    equations : str | list
        List of Equation objects or str. ``all`` as a string represents
        all the equations specified before the creation of this model
    problem : str
        Problem type (e.g. LP, NLP etc.)
    sense : "MIN" or "MAX", optional
        Minimize or maximize
    objective_variable : Variable, optional
        Objective variable to minimize or maximize
    limited_variables : list, optional
        Allows limiting the domain of variables used in a model.

    Examples
    --------
    >>> transport = Model(m, "transport", equations=[cost,supply,demand])
    """

    def __init__(
        self,
        container: "Container",
        name: str,
        equations: List["Equation"],
        problem: str,
        sense: Optional[Literal["MIN", "MAX"]] = None,
        objective_variable: Optional["Variable"] = None,
        limited_variables: Optional[list] = None,
    ):
        self.name = name
        self.ref_container = container
        self.problem, self.sense, self.objective_variable = (
            self._validate_model(problem, sense, objective_variable)
        )
        self._equations = equations
        self._limited_variables = limited_variables
        self.ref_container._addStatement(self)
        self._generate_attribute_symbols()

        # Attributes
        self.num_domain_violations = None
        self.algorithm_time = None
        self.solve_time = None
        self.solver_time = None
        self.num_iterations = None
        self.marginals = None
        self.max_infeasibility = None
        self.mean_infeasibility = None
        self.status = None
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

    @property
    def equations(self) -> Union[str, list]:
        return self._equations

    @equations.setter
    def equations(self, new_equations) -> None:
        self._equations = new_equations

    def _generate_attribute_symbols(self) -> None:
        for attr_name in self._getAttributeNames():
            symbol_name = f"{self.name}_{attr_name}"
            _ = gp.Parameter(self.ref_container, symbol_name)

    def _getAttributeNames(self) -> Dict[str, str]:
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

    def _prepare_gams_options(self, commandline_options) -> GamsOptions:
        options = None
        if commandline_options:
            if not isinstance(commandline_options, dict):
                raise Exception("commandline_options must be a dict")

            options = GamsOptions(self.ref_container.workspace)
            for option, value in commandline_options.items():
                if option.lower() not in utils.COMMANDLINE_OPTIONS:
                    raise Exception(f"Invalid commandline option: {option}")

                setattr(options, option, value)

        return options

    def solve(
        self,
        commandline_options: Optional[dict] = None,
        output: Optional[io.TextIOWrapper] = None,
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
        self._append_solve_string()
        self._assign_model_attributes()

        self.ref_container.write(self.ref_container._gdx_path)
        gams_string = self.ref_container.generateGamsString(
            self.ref_container._unsaved_statements
        )

        options = self._prepare_gams_options(commandline_options)

        checkpoint = (
            self.ref_container._restart_from
            if os.path.exists(
                self.ref_container._restart_from._checkpoint_file_name
            )
            else None
        )
        job = GamsJob(
            self.ref_container.workspace,
            source=gams_string,
            checkpoint=checkpoint,
        )

        job.run(
            gams_options=options,
            checkpoint=self.ref_container._save_to,
            create_out_db=True,
            output=output,
        )

        self.ref_container._swap_checkpoints()

        self.ref_container._gdx_path = (
            job.out_db.workspace.working_directory
            + os.sep
            + job.out_db.name
            + ".gdx"
        )

        self.ref_container.loadRecordsFromGdx(self.ref_container._gdx_path)
        self._update_model_attributes()

    def _validate_model(
        self, problem, sense=None, objective_variable=None
    ) -> Tuple:
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

        if objective_variable is not None and not isinstance(
            objective_variable, gp.Variable
        ):
            raise TypeError("Objective variable must be type Variable")

        return problem, sense, objective_variable

    def _append_solve_string(self) -> None:
        solve_string = f"solve {self.name} using {self.problem}"

        if self.sense:
            solve_string += f" {self.sense}"

        if self.objective_variable:
            solve_string += f" {self.objective_variable.gamsRepr()}"

        self.ref_container._unsaved_statements[utils._getUniqueName()] = (
            solve_string + ";\n"
        )

    def _assign_model_attributes(self) -> None:
        """
        Assign model attributes to parameters
        """
        for attr_name in self._getAttributeNames().keys():
            symbol_name = f"{self.name}_{attr_name}"

            self.ref_container._unsaved_statements[utils._getUniqueName()] = (
                f"{symbol_name} = {self.name}.{attr_name};"
            )

    def _update_model_attributes(self) -> None:
        for gams_attr, python_attr in self._getAttributeNames().items():
            symbol_name = f"{self.name}_{gams_attr}"

            if python_attr == "status":
                setattr(
                    self,
                    python_attr,
                    ModelStatus(
                        self.ref_container[symbol_name].records.values[0][0]
                    ),
                )
            else:
                setattr(
                    self,
                    python_attr,
                    self.ref_container[symbol_name].records.values[0][0],
                )

    def getStatement(self) -> str:
        """
        Statement of the Model definition

        Returns
        -------
        str
        """
        equations_str = ""
        if isinstance(self._equations, str):
            equations_str = self._equations  # pragma: no cover
        else:
            equations_str = ",".join(
                [equation.name for equation in self._equations]
            )

        if self._limited_variables:
            limited_variables_str = ",".join(
                [variable.gamsRepr() for variable in self._limited_variables]
            )
            equations_str = ",".join([equations_str, limited_variables_str])

        model_str = f"\nModel {self.name} / {equations_str} /;"

        return model_str
