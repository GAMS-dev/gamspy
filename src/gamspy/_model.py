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
from enum import Enum
from typing import Dict, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from gamspy import Container


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
        equations: Union[str, list],
        limited_variables: Optional[list] = None,
    ):
        self.name = name
        self.ref_container = container
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
        import gamspy as gp

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

    def getStatement(self) -> str:
        """
        Statement of the Model definition

        Returns
        -------
        str
        """
        equations_str = ""
        if isinstance(self._equations, str):
            equations_str = self._equations
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
