#
# GAMS - General Algebraic Modeling System Python API
#
# Copyright (c) 2017-2023 GAMS Development Corp. <support@gams.com>
# Copyright (c) 2017-2023 GAMS Software GmbH <support@gams.com>
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

from typing import Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from gamspy import Container


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
    equations : Union[str, list]
        List of Equation objects or str. ``all`` as a string represents
        all the equations specified before the creation of this model
    limited_variables : Optional[list], optional
        Allows limiting the domain of variables used in a model.

    Example
    ----------
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

    @property
    def equations(self) -> Union[str, list]:
        return self._equations

    @equations.setter
    def equations(self, new_equations) -> None:
        self._equations = new_equations

    def getStatement(self) -> str:
        """Statement of the Model definition

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
