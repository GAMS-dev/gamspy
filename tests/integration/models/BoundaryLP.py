"""
## GAMSSOURCE: https://www.gams.com/latest/psoptlib_ml/libhtml/psoptlib_BoundaryLP.html
## LICENSETYPE: Demo
## MODELTYPE: LP


Simple linear programming model for determination of boundary values of an objective function

For more details please refer to Chapter 2 (Gcode2.2), of the following book:
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

from gamspy import Container, Equation, Model, Variable


def main():
    m = Container()

    # VARIABLES #
    x1 = Variable(m, name="x1")
    x2 = Variable(m, name="x2")
    x3 = Variable(m, name="x3")

    # EQUATIONS #
    eq1 = Equation(m, name="eq1", type="regular")
    eq2 = Equation(m, name="eq2", type="regular")
    eq3 = Equation(m, name="eq3", type="regular")

    eq1[...] = x1 + 2 * x2 <= 3
    eq2[...] = x2 + x3 <= 2
    eq3[...] = x1 + x2 + x3 == 4
    of = x1 + 2 * x2 - 3 * x3

    LP1 = Model(
        m,
        name="LP1",
        equations=m.getEquations(),
        problem="lp",
        sense="max",
        objective=of,
    )

    x1.lo = 0
    x1.up = 5
    x2.lo = 0
    x2.up = 3
    x3.lo = 0
    x3.up = 2

    LP1.solve()

    print(" * Model LP1 *")
    print("x1:  ", round(x1.toValue(), 3))
    print("x2:  ", round(x2.toValue(), 3))
    print("x3:  ", round(x3.toValue(), 3))
    print("of:  ", round(LP1.objective_value, 3))

    LP2 = Model(
        m,
        name="LP2",
        equations=m.getEquations(),
        problem="lp",
        sense="min",
        objective=of,
    )
    LP2.solve()
    print("\n * Model LP2 *")
    print("x1:  ", round(x1.toValue(), 3))
    print("x2:  ", round(x2.toValue(), 3))
    print("x3:  ", round(x3.toValue(), 3))
    print("of:  ", round(LP2.objective_value, 3))

    import math

    assert math.isclose(LP2.objective_value, -4.000000, rel_tol=0.001)


if __name__ == "__main__":
    main()
