"""
*** Simple Mixed Integer Linear Programming model

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
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Variable


def main():
    m = Container()

    # VARIABLES #
    x = Variable(m, name="x", type="free")
    of = Variable(m, name="of", type="free")
    y = Variable(m, name="y", type="binary")

    # EQUATIONS #
    eq1 = Equation(m, name="eq1", type="regular")
    eq2 = Equation(m, name="eq2", type="regular")
    eq3 = Equation(m, name="eq3", type="regular")

    eq1.expr = -3 * x + 2 * y >= 1
    eq2.expr = -8 * x + 10 * y <= 10
    eq3.expr = x + y == of

    MIP1 = Model(
        m,
        name="MIP1",
        equations=m.getEquations(),
        problem="mip",
        sense="max",
        objective=of,
    )

    x.up.assign = 0.3
    MIP1.solve()

    print("Objective Function Value:  ", round(of.toValue(), 4))
    print("y:  ", round(y.toValue(), 4))
    print("x:  ", round(x.toValue(), 4))


if __name__ == "__main__":
    main()
