"""
Optimal design of industrial refrigeration system.

Paul H and Tay, Optimal design of an industrial refrigeration system.
Proc. of Int. Conf. on Optimization Techniques and Applications, 1987,
pp.427-435.

Pant, M., Thangaraj, R., Singh, V.P., (2009) Optimization of mechanical
design problems using improved differential evolution algorithm.
International Journal of Recent Trends in Engineering, vol.1, No.5,
May 2009, pp.21-25.
"""
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Parameter
from gamspy import Variable


def main():
    m = Container()

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
    x11 = Variable(m, name="x11")
    x12 = Variable(m, name="x12")
    x13 = Variable(m, name="x13")
    x14 = Variable(m, name="x14")
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
    e11 = Equation(m, name="e11", type="regular")
    e12 = Equation(m, name="e12", type="regular")
    e13 = Equation(m, name="e13", type="regular")
    e14 = Equation(m, name="e14", type="regular")
    e15 = Equation(m, name="e15", type="regular")
    eobj = Equation(m, name="eobj", type="regular")

    # Objective function to be minimized:
    eobj.expr = obj == (
        63098.88 * x2 * x4 * x12
        + 5441.5 * x12 * x2**2
        + 115055.5 * x6 * (x2**1.664)
        + 6172.27 * x6 * x2**2
        + 63098.88 * x1 * x3 * x11
        + 5441.5 * x11 * x1**2
        + 115055.5 * x5 * (x1**1.664)
        + 6172.27 * x5 * x1**2
        + 140.53 * x1 * x11
        + 281.29 * x3 * x11
        + 70.26 * x1**2
        + 281.29 * x1 * x3
        + 281.29 * x3**2
        + 14437
        * (x8**1.8812)
        * (x12**0.3424)
        * x7
        * x10
        * (x1**2)
        / (x9 * x14)
        + 20470.2 * (x7**2.893) * (x11 * 0.316) * (x1 * 82)
    )

    # Constaints:
    e1.expr = 1.524 / x7 <= 1
    e2.expr = 1.524 / x8 <= 1
    e3.expr = 0.07789 * x1 - 2 * x9 / x7 <= 1
    e4.expr = 7.05305 * (x1**2) * x10 / (x2 * x8 * x9 * x14) <= 1
    e5.expr = 0.0833 * x14 / x13 <= 1
    e6.expr = (
        47.136 * x12 * (x2**0.333) / x10
        - 1.333 * x8 * (x13**2.1195)
        + 62.08 * (x13**2.1195) * (x8**0.2) / (x10 * x12)
        <= 1
    )
    e7.expr = 0.04771 * x10 * (x8**1.8812) * (x12**0.3424) <= 1
    e8.expr = 0.0488 * x9 * (x7**1.893) * (x11**0.316) <= 1
    e9.expr = 0.0099 * x1 / x3 <= 1
    e10.expr = 0.0193 * x2 / x4 <= 1
    e11.expr = 0.0298 * x1 / x5 <= 1
    e12.expr = 0.056 * x2 / x6 <= 1
    e13.expr = 2 / x9 <= 1
    e14.expr = 2 / x10 <= 1
    e15.expr = x12 / x11 <= 1

    # Bounds on variables:
    x1.lo.assign = 0.001
    x1.up.assign = 5
    x2.lo.assign = 0.001
    x2.up.assign = 5
    x3.lo.assign = 0.001
    x3.up.assign = 5
    x4.lo.assign = 0.001
    x4.up.assign = 5
    x5.lo.assign = 0.001
    x5.up.assign = 5
    x6.lo.assign = 0.001
    x6.up.assign = 5
    x7.lo.assign = 0.001
    x7.up.assign = 5
    x8.lo.assign = 0.001
    x8.up.assign = 5
    x9.lo.assign = 0.001
    x9.up.assign = 5
    x10.lo.assign = 0.001
    x10.up.assign = 5
    x11.lo.assign = 0.001
    x11.up.assign = 5
    x12.lo.assign = 0.001
    x12.up.assign = 5
    x13.lo.assign = 0.001
    x13.up.assign = 5
    x14.lo.assign = 0.001
    x14.up.assign = 5

    refrigeration = Model(
        m,
        name="refrigeration",
        equations=m.getEquations(),
        problem="nlp",
        sense="MIN",
        objective=obj,
    )
    refrigeration.solve()

    # REPORTING PARAMETER
    rep = Parameter(m, name="rep", domain=["*", "*"])
    rep["x1", "value"] = x1.l
    rep["x2", "value"] = x2.l
    rep["x3", "value"] = x3.l
    rep["x4", "value"] = x4.l
    rep["x5", "value"] = x5.l
    rep["x6", "value"] = x6.l
    rep["x7", "value"] = x7.l
    rep["x8", "value"] = x8.l
    rep["x9", "value"] = x9.l
    rep["x10", "value"] = x10.l
    rep["x11", "value"] = x11.l
    rep["x12", "value"] = x12.l
    rep["x13", "value"] = x13.l
    rep["x14", "value"] = x14.l

    print("Objective Function Value: ", round(obj.toValue(), 4), "\n")
    print("Solution Summary:\n", rep.pivot().round(3))

    # End refrigeration


if __name__ == "__main__":
    main()
