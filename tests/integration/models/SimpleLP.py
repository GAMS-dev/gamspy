"""
## GAMSSOURCE: https://www.gams.com/latest/psoptlib_ml/libhtml/psoptlib_SimpleLP.html
## LICENSETYPE: Demo
## MODELTYPE: LP


Simple linear programming model

For more details please refer to Chapter 2 (Gcode2.1), of the following book:
Soroudi, Alireza. Power System Optimization Modeling in GAMS. Springer, 2017.
--------------------------------------------------------------------------------
Model type: LP
--------------------------------------------------------------------------------
Contributed by
Dr. Alireza Soroudi
IEEE Senior Member
email: alireza.soroudi@gmail.com
We do request that publications derived from the use of the developed GAMS code
explicitly acknowledge that fact by citing
Soroudi, Alireza. Power System Optimization Modeling in GAMS. Springer, 2017.
DOI: doi.org/10.1007/978-3-319-62350-4
"""

from __future__ import annotations

import os

from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Variable


def main():
    m = Container(
        system_directory=os.getenv("SYSTEM_DIRECTORY", None),
    )

    # VARIABLES #
    x1 = Variable(m, name="x1")
    x2 = Variable(m, name="x2")
    x3 = Variable(m, name="x3")

    # EQUATIONS #
    eq1 = Equation(m, name="eq1", type="regular")
    eq2 = Equation(m, name="eq2", type="regular")
    eq3 = Equation(m, name="eq3", type="regular")

    eq1[...] = x1 + 2 * x2 >= 3
    eq2[...] = x3 + x2 >= 5
    eq3[...] = x1 + x3 == 4
    eq4 = x1 + 3 * x2 + 3 * x3  # Objective Function

    LP1 = Model(
        m,
        name="LP1",
        equations=m.getEquations(),
        problem="lp",
        sense="min",
        objective=eq4,
    )
    LP1.solve()

    print("Objective Function Value:  ", round(LP1.objective_value, 4), "\n")
    print("x1:  ", round(x1.toValue(), 4))
    print("x2:  ", round(x2.toValue(), 4))
    print("x3:  ", round(x3.toValue(), 4))


if __name__ == "__main__":
    main()
