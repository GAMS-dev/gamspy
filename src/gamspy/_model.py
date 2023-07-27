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
from typing import Optional, Union, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from gamspy import Container


class ModelStatus(Enum):
    # Not solved yet
    NotSolved = 0
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
        self._status = ModelStatus.NotSolved

    @property
    def status(self) -> ModelStatus:
        return self._status

    @status.setter
    def status(self, new_status):
        self._status = ModelStatus(new_status)

    @property
    def equations(self) -> Union[str, list]:
        return self._equations

    @equations.setter
    def equations(self, new_equations) -> None:
        self._equations = new_equations

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
