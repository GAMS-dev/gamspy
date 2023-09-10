"""
Updating and Projecting Coefficients: The RAS Approach (IOBALANCE)

The RAS procedure (named after Richard A. Stone) is an iterative procedure to
update matrices. This numerical example is taken from chapter 7.4.2 of Miller
and Blair. Several additional optimization formulations will be applied to
this toy problem.


Miller R E, and Blair P D, Input-Output Analysis: Foundations and Extensions,
Cambridge University Press, New York, 2009.

Keywords: linear programming, nonlinear programming, quadratic constraints,
statistics,
          RAS approach
"""

from gamspy import (
    Set,
    Alias,
    Parameter,
    Variable,
    Equation,
    Container,
    Model,
    Sum,
    Smax,
    Card,
)
import gamspy.math as gams_math
import numpy as np
from gamspy import Problem, Sense


def main():
    m = Container()

    # Sets

    i = Set(m, name="i", records=list(range(1, 4)))
    j = Alias(m, name="j", alias_with=i)

    # Parameters

    a0 = Parameter(
        m,
        name="a0",
        domain=[i, j],
        records=np.array(
            [
                [0.120, 0.100, 0.049],
                [0.210, 0.247, 0.265],
                [0.026, 0.249, 0.145],
            ]
        ),
    )

    z1 = Parameter(
        m,
        name="z1",
        domain=[i, j],
        records=np.array(
            [
                [98, 72, 75],
                [65, 8, 63],
                [88, 27, 44],
            ]
        ),
    )

    x = Parameter(m, name="x", domain=[j], records=np.array([421, 284, 283]))
    u = Parameter(
        m,
        name="u",
        domain=[i],
    )
    v = Parameter(
        m,
        name="v",
        domain=[j],
    )
    a1 = Parameter(m, name="ai", domain=[i, j])

    u[i] = Sum(j, z1[i, j])
    v[j] = Sum(i, z1[i, j])

    a1[i, j] = z1[i, j] / x[j]

    print("Values of u \n", u.records)
    print("Values of v \n", v.records)
    print("Values of a1 \n", a1.records)

    # --- 1: RAS updating

    r = Parameter(m, name="r", domain=[i])
    s = Parameter(m, name="s", domain=[j])

    r[i] = 1
    s[j] = 1

    oldr = Parameter(m, name="oldr", domain=[i])
    olds = Parameter(m, name="olds", domain=[j])
    maxdelta = Parameter(m, name="maxdelta")
    maxdelta.assign = 1

    while True:
        oldr[i] = r[i]
        olds[j] = s[j]
        r[i] = r[i] * u[i] / Sum(j, r[i] * a0[i, j] * x[j] * s[j])
        s[j] = s[j] * v[j] / Sum(i, r[i] * a0[i, j] * x[j] * s[j])
        maxdelta.assign = max(
            Smax(i, gams_math.abs(oldr[i] - r[i])),
            Smax(j, gams_math.abs(olds[j] - s[j])),
        )
        print("In loop maxdelta: ", round(maxdelta.records.values[0][0], 3))
        if maxdelta.records.values[0][0] < 0.005:
            break

    # Parameter
    report = Parameter(m, name="report", domain=["*", i, j])
    # option report:3:1:2

    report["A0", i, j] = a0[i, j]
    report["A1", i, j] = a1[i, j]
    report["RAS", i, j] = r[i] * a0[i, j] * s[j]

    # --- 2: Entropy formulation   a*ln(a/a0)
    #        The RAS procedure gives the solution to the Entropy formulation
    # Variable
    obj = Variable(m, name="obj", type="free")
    a = Variable(m, name="a", type="positive", domain=[i, j])

    # Equation
    rowbal = Equation(m, name="rowbal", domain=[i])
    colbal = Equation(m, name="colbal", domain=[j])
    defobjent = Equation(m, name="defobjent")

    rowbal[i] = Sum(j, a[i, j] * x[j]) == u[i]

    colbal[j] = Sum(i, a[i, j] * x[j]) == v[j]

    defobjent.expr = obj == Sum(
        [i, j], x[j] * a[i, j] * gams_math.log(a[i, j] / a0[i, j])
    )

    mEntropy = Model(
        m,
        name="mEntropy",
        equations=[rowbal, colbal, defobjent],
        problem=Problem.NLP,
        sense=Sense.MIN,
        objective=obj,
    )

    # we need to exclude small values to avoid domain violations
    a.lo[i, j] = 1e-5

    mEntropy.solve()
    report["Entropy", i, j] = a.l[i, j]

    # --- 3: Entropy with flow variable
    #        we can balance the flow matrix instead of the A matrix
    zv = Variable(m, name="zv", type="free", domain=[i, j])

    rowbalz = Equation(m, name="rowbalz", domain=[i])
    colbalz = Equation(m, name="colbalz", domain=[j])
    defobjentz = Equation(m, name="defobjentz")

    rowbalz[i] = Sum(j, zv[i, j]) == u[i]

    colbalz[j] = Sum(i, zv[i, j]) == v[j]

    zbar = Parameter(m, name="zbar", domain=[i, j])

    zbar[i, j] = a0[i, j] * x[j]
    zv.lo[i, j] = 1

    defobjentz.expr = obj == Sum(
        [i, j], zv[i, j] * gams_math.log(zv[i, j] / zbar[i, j])
    )

    mEntropyz = Model(
        m,
        name="mEntropyz",
        equations=[rowbalz, colbalz, defobjentz],
        problem=Problem.NLP,
        sense=Sense.MIN,
        objective=obj,
    )

    # turn off detailed outputs
    m.addOptions({"limRow": 0, "limCol": 0, "solPrint": "off"})

    mEntropyz.solve()
    report["EntropyZ", i, j] = zv.l[i, j] / x[j]

    # --- 4. absolute deviation formulations result in LPs
    #        MAD Mean Absolute Deviations
    #        MAPE Mean absolute percentage error
    #        Linf Infinity norm
    # Positive Variable
    ap = Variable(m, name="ap", type="positive", domain=[i, j])
    an = Variable(m, name="an", type="positive", domain=[i, j])
    amax = Variable(m, name="amax", type="positive")

    # Equation
    defabs = Equation(m, name="defabs", domain=[i, j])
    defmaxp = Equation(m, name="defmaxp", domain=[i, j])
    defmaxn = Equation(m, name="defmaxn", domain=[i, j])
    defmad = Equation(m, name="defmad")
    defmade = Equation(m, name="defmade")
    deflinf = Equation(m, name="deflinf")

    defabs[i, j] = a[i, j] - a0[i, j] == ap[i, j] - an[i, j]

    defmaxp[i, j] = a[i, j] - a0[i, j] <= amax

    defmaxn[i, j] = a[i, j] - a0[i, j] >= -amax

    defmad.expr = obj == 1 / gams_math.power(Card(i), 2) * Sum(
        [i, j], ap[i, j] + an[i, j]
    )

    defmade.expr = obj == 100 / gams_math.power(Card(i), 2) * Sum(
        [i, j], (ap[i, j] + an[i, j]) / a0[i, j]
    )

    deflinf.expr = obj == amax

    # Model
    mMAD = Model(
        m,
        name="mMAD",
        equations=[rowbal, colbal, defabs, defmad],
        problem="LP",
        sense=Sense.MIN,
        objective=obj,
    )
    mMADE = Model(
        m,
        name="mMADE",
        equations=[rowbal, colbal, defabs, defmade],
        problem="LP",
        sense=Sense.MIN,
        objective=obj,
    )
    mLinf = Model(
        m,
        name="mLinf",
        equations=[rowbal, colbal, defmaxp, defmaxn, deflinf],
        problem="LP",
        sense=Sense.MIN,
        objective=obj,
    )

    mMAD.solve()
    report["MAD", i, j] = a.l[i, j]
    mMADE.solve()
    report["MADE", i, j] = a.l[i, j]
    mLinf.solve()
    report["Linf", i, j] = a.l[i, j]

    # --- 5. Squared Deviations can be solved with powerful QP codes
    #        SD     squared deviations
    #        RSD    relative squared deviations
    # Equation
    defsd = Equation(m, name="defsd")
    defrsd = Equation(m, name="defrsd")

    defsd.expr = obj == Sum([i, j], gams_math.power(a[i, j] + a0[i, j], 2))

    defrsd.expr = obj == Sum(
        [i, j], gams_math.power(a[i, j] + a0[i, j], 2) / a0[i, j]
    )

    # Model
    mSD = Model(
        m,
        name="mSD",
        equations=[rowbal, colbal, defsd],
        problem=Problem.QCP,
        sense=Sense.MIN,
        objective=obj,
    )
    mRSD = Model(
        m,
        name="mRSD",
        equations=[rowbal, colbal, defrsd],
        problem=Problem.QCP,
        sense=Sense.MIN,
        objective=obj,
    )

    mSD.solve()
    report["SD", i, j] = a.l[i, j]
    mRSD.solve()
    report["RSD", i, j] = a.l[i, j]

    print()
    print("\t SOLUTION REPORT: \n", report.pivot())


if __name__ == "__main__":
    main()
