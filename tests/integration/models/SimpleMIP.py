"""
## GAMSSOURCE: https://www.gams.com/latest/psoptlib_ml/libhtml/psoptlib_SimpleMIP.html
## LICENSETYPE: Demo
## MODELTYPE: MIP


Simple Mixed Integer Linear Programming model

For more details please refer to Chapter 2 (Gcode2.3), of the following book:
Soroudi, Alireza. Power System Optimization Modeling in GAMS. Springer, 2017.
--------------------------------------------------------------------------------
Model type: MIP
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

from gamspy import Container, Equation, Model, Variable


def main():
    m = Container(
        system_directory=os.getenv("SYSTEM_DIRECTORY", None),
    )

    # VARIABLES #
    x = Variable(m, name="x", type="free")
    y = Variable(m, name="y", type="binary")

    # EQUATIONS #
    eq1 = Equation(m, name="eq1", type="regular")
    eq2 = Equation(m, name="eq2", type="regular")

    eq1[...] = -3 * x + 2 * y >= 1
    eq2[...] = -8 * x + 10 * y <= 10
    eq3 = x + y  # Objective Function

    MIP1 = Model(
        m,
        name="MIP1",
        equations=m.getEquations(),
        problem="mip",
        sense="max",
        objective=eq3,
    )

    x.up[...] = 0.3
    MIP1.solve()

    print("Objective Function Value:  ", round(MIP1.objective_value, 4))
    print("y:  ", round(y.toValue(), 4))
    print("x:  ", round(x.toValue(), 4))


if __name__ == "__main__":
    main()
