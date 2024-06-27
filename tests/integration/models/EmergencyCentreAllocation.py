"""
## GAMSSOURCE: https://www.gams.com/latest/psoptlib_ml/libhtml/psoptlib_EmergencyCentreAllocation.html
## LICENSETYPE: Demo
## MODELTYPE: MIP


Mixed integer linear programming model for optimal allocation of Emergency Centres

For more details please refer to Chapter 2 (Gcode2.5), of the following book:
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
        system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None),
    )

    # Binary Variables
    x1 = Variable(m, name="x1", type="binary")
    x2 = Variable(m, name="x2", type="binary")
    x3 = Variable(m, name="x3", type="binary")
    x4 = Variable(m, name="x4", type="binary")
    x5 = Variable(m, name="x5", type="binary")
    x6 = Variable(m, name="x6", type="binary")

    # Equations
    eq1 = Equation(m, name="eq1", type="regular")
    eq2 = Equation(m, name="eq2", type="regular")
    eq3 = Equation(m, name="eq3", type="regular")
    eq4 = Equation(m, name="eq4", type="regular")
    eq5 = Equation(m, name="eq5", type="regular")
    eq6 = Equation(m, name="eq6", type="regular")

    eq1[...] = x1 + x6 >= 1
    eq2[...] = x2 >= 1
    eq3[...] = x3 + x5 >= 1
    eq4[...] = x4 + x5 >= 1
    eq5[...] = x3 + x4 + x5 + x6 >= 1
    eq6[...] = x1 + x5 + x6 >= 1
    of = x1 + x2 + x3 + x4 + x5 + x6

    emergency = Model(
        m,
        name="emergency",
        equations=m.getEquations(),
        problem="mip",
        sense="min",
        objective=of,
    )
    emergency.solve()

    import math

    assert math.isclose(emergency.objective_value, 3.0000, rel_tol=0.001)
    print("Objective Function Value:\t", emergency.objective_value)


if __name__ == "__main__":
    main()
