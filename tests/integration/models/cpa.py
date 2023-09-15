"""
Combustion of propan in air.

Hiebert's (1983) reduced variant.
Hiebert, K.L., An evaluation of mathematical software that solves systems
of nonlinear equations. ACM Transactions on Mathematical Software,
vol. 8, 1983, pp.5-20.

Shacham, M., Brauner, N., Cutlip, M.B., (2002) A web-based library for
testing performance of numerical software for solving nonlinear algebraic
equations. Computers & Chemical Engineering, vol. 26, 2002, pp.547-554.

Meintjes, K., Morgan, A.P., (1990) Chemical equilibrium systems as
numerical test problems. ACM Trans. Math. Software, 16, 1990, pp. 143-151.
"""
import gamspy.math as gams_math
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Parameter
from gamspy import Variable


def sqr(x):
    return gams_math.power(x, 2)


def main():
    m = Container()

    # SCALAR #
    R = Parameter(m, name="R", records=40)

    # VARIABLES #
    x1 = Variable(m, name="x1")
    x2 = Variable(m, name="x2")
    x3 = Variable(m, name="x3")
    x4 = Variable(m, name="x4")
    x5 = Variable(m, name="x5")
    x6 = Variable(m, name="x6")
    x7 = Variable(m, name="x7")
    x8 = Variable(m, name="x8")
    x9 = Variable(m, name="x9")
    x10 = Variable(m, name="x10")
    obj = Variable(m, name="obj")

    # EQUATIONS #
    e1 = Equation(m, name="e1", type="regular")
    e2 = Equation(m, name="e2", type="regular")
    e3 = Equation(m, name="e3", type="regular")
    e4 = Equation(m, name="e4", type="regular")
    e5 = Equation(m, name="e5", type="regular")
    e6 = Equation(m, name="e6", type="regular")
    e7 = Equation(m, name="e7", type="regular")
    e8 = Equation(m, name="e8", type="regular")
    e9 = Equation(m, name="e9", type="regular")
    e10 = Equation(m, name="e10", type="regular")
    eobj = Equation(m, name="eobj", type="regular")

    # CONSTRAINTS
    e1.expr = x1 + x4 - 3 == 0
    e2.expr = 2 * x1 + x2 + x4 + x7 + x8 + x9 + 2 * x10 - R == 0
    e3.expr = 2 * x2 + 2 * x5 + x6 + x7 - 8 == 0
    e4.expr = 2 * x3 + x5 - 4 * R == 0
    e5.expr = x1 * x5 - 0.193 * x2 * x4 == 0
    e6.expr = (
        x6 * gams_math.sqrt(x2)
        - 0.002597
        * gams_math.sqrt(
            x2 * x4 * (x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10)
        )
        == 0
    )
    e7.expr = (
        x7 * gams_math.sqrt(x4)
        - 0.003448
        * gams_math.sqrt(
            x1 * x4 * (x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10)
        )
        == 0
    )
    e8.expr = (
        x4 * x8
        - 1.799
        * x2
        * (x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10)
        / 100000
        == 0
    )
    e9.expr = (
        x4 * x9
        - 0.0002155
        * x1
        * gams_math.sqrt(
            x3 * (x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10)
        )
        == 0
    )
    e10.expr = (
        x10 * sqr(x4)
        - 3.84
        * sqr(x4)
        * (x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10)
        / 100000
        == 0
    )

    # OBJECTIVE
    eobj.expr = obj == 1

    # Bound on variables:
    x1.lo.assign = 0.000001
    x1.up.assign = 100
    x2.lo.assign = 0.000001
    x2.up.assign = 100
    x3.lo.assign = 0.000001
    x3.up.assign = 100
    x4.lo.assign = 0.000001
    x4.up.assign = 100
    x5.lo.assign = 0.000001
    x5.up.assign = 100
    x6.lo.assign = 0.000001
    x6.up.assign = 100
    x7.lo.assign = 0.000001
    x7.up.assign = 100
    x8.lo.assign = 0.000001
    x8.up.assign = 100
    x9.lo.assign = 0.000001
    x9.up.assign = 100
    x10.lo.assign = 0.000001
    x10.up.assign = 100

    # Initial point:
    x1.l.assign = 2
    x2.l.assign = 5
    x3.l.assign = 40
    x4.l.assign = 1
    x5.l.assign = 0
    x6.l.assign = 0
    x7.l.assign = 0
    x8.l.assign = 0
    x9.l.assign = 0
    x10.l.assign = 5

    cpa = Model(
        m,
        name="cpa",
        equations=m.getEquations(),
        problem="nlp",
        sense="min",
        objective=obj,
    )

    cpa.solve()

    print("x1:  ", round(x1.toValue(), 3))
    print("x2:  ", round(x2.toValue(), 3))
    print("x3:  ", round(x3.toValue(), 3))
    print("x4:  ", round(x4.toValue(), 3))
    print("x5:  ", round(x5.toValue(), 3))
    print("x6:  ", round(x6.toValue(), 3))
    print("x7:  ", round(x7.toValue(), 3))
    print("x8:  ", round(x8.toValue(), 3))
    print("x9:  ", round(x9.toValue(), 3))
    print("x10:  ", round(x10.toValue(), 3))

    # End cpa


if __name__ == "__main__":
    main()
