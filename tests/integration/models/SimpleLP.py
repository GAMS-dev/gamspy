"""
*** Simple linear programming model

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
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Variable


def main():
    m = Container()

    # VARIABLES #
    x1 = Variable(m, name="x1")
    x2 = Variable(m, name="x2")
    x3 = Variable(m, name="x3")
    of = Variable(m, name="of")

    # EQUATIONS #
    eq1 = Equation(m, name="eq1", type="regular")
    eq2 = Equation(m, name="eq2", type="regular")
    eq3 = Equation(m, name="eq3", type="regular")
    eq4 = Equation(m, name="eq4", type="regular")

    eq1.expr = x1 + 2 * x2 >= 3
    eq2.expr = x3 + x2 >= 5
    eq3.expr = x1 + x3 == 4
    eq4.expr = x1 + 3 * x2 + 3 * x3 == of

    LP1 = Model(
        m,
        name="LP1",
        equations=m.getEquations(),
        problem="lp",
        sense="min",
        objective=of,
    )
    LP1.solve()

    print("Objective Function Value:  ", round(of.toValue(), 4), "\n")
    print("x1:  ", round(x1.toValue(), 4))
    print("x2:  ", round(x2.toValue(), 4))
    print("x3:  ", round(x3.toValue(), 4))


if __name__ == "__main__":
    main()
