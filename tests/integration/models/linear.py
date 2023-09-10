"""
Linear Regression with Various Criteria (LINEAR)

This example solves linear models with differing objective functions.
Absolute deviations cannot be solved in a reliable manner with
most NLP systems and one has to resort to a formulation with
negative and positive deviations (models ending with the letter a).


Bracken, J, and McCormick, G P, Chapter 8.2. In Selected Applications of
Nonlinear Programming. John Wiley and Sons, New York, 1968, pp. 86-88.

Keywords: linear programming, nonlinear programming, discontinuous derivatives,
          linear regression, econometrics
"""

from gamspy import Set, Parameter, Variable, Equation, Container, Model, Sum
import gamspy.math as gams_math
import pandas as pd
from gamspy import Problem, Sense


def main():
    m = Container()

    # Data
    dat_df = pd.DataFrame(
        [
            ["1", "y", 99],
            ["1", "a", 1],
            ["1", "b", 85],
            ["1", "c", 76],
            ["1", "d", 44],
            ["2", "y", 93],
            ["2", "a", 1],
            ["2", "b", 82],
            ["2", "c", 78],
            ["2", "d", 42],
            ["3", "y", 99],
            ["3", "a", 1],
            ["3", "b", 75],
            ["3", "c", 73],
            ["3", "d", 42],
            ["4", "y", 97],
            ["4", "a", 1],
            ["4", "b", 74],
            ["4", "c", 72],
            ["4", "d", 44],
            ["5", "y", 90],
            ["5", "a", 1],
            ["5", "b", 76],
            ["5", "c", 73],
            ["5", "d", 43],
            ["6", "y", 96],
            ["6", "a", 1],
            ["6", "b", 74],
            ["6", "c", 69],
            ["6", "d", 46],
            ["7", "y", 93],
            ["7", "a", 1],
            ["7", "b", 73],
            ["7", "c", 69],
            ["7", "d", 46],
            ["8", "y", 130],
            ["8", "a", 1],
            ["8", "b", 96],
            ["8", "c", 80],
            ["8", "d", 36],
            ["9", "y", 118],
            ["9", "a", 1],
            ["9", "b", 93],
            ["9", "c", 78],
            ["9", "d", 36],
            ["10", "y", 88],
            ["10", "a", 1],
            ["10", "b", 70],
            ["10", "c", 73],
            ["10", "d", 37],
            ["11", "y", 89],
            ["11", "a", 1],
            ["11", "b", 82],
            ["11", "c", 71],
            ["11", "d", 46],
            ["12", "y", 93],
            ["12", "a", 1],
            ["12", "b", 80],
            ["12", "c", 72],
            ["12", "d", 45],
            ["13", "y", 94],
            ["13", "a", 1],
            ["13", "b", 77],
            ["13", "c", 76],
            ["13", "d", 42],
            ["14", "y", 75],
            ["14", "a", 1],
            ["14", "b", 67],
            ["14", "c", 76],
            ["14", "d", 50],
            ["15", "y", 84],
            ["15", "a", 1],
            ["15", "b", 82],
            ["15", "c", 70],
            ["15", "d", 48],
            ["16", "y", 91],
            ["16", "a", 1],
            ["16", "b", 76],
            ["16", "c", 76],
            ["16", "d", 41],
            ["17", "y", 100],
            ["17", "a", 1],
            ["17", "b", 74],
            ["17", "c", 78],
            ["17", "d", 31],
            ["18", "y", 98],
            ["18", "a", 1],
            ["18", "b", 71],
            ["18", "c", 80],
            ["18", "d", 29],
            ["19", "y", 101],
            ["19", "a", 1],
            ["19", "b", 70],
            ["19", "c", 83],
            ["19", "d", 39],
            ["20", "y", 80],
            ["20", "a", 1],
            ["20", "b", 64],
            ["20", "c", 79],
            ["20", "d", 38],
        ]
    )

    # Sets
    i = Set(m, name="i", records=list(range(1, 21)))
    n = Set(m, name="n", records=["a", "b", "c", "d"])

    # Parameters
    dat = Parameter(m, name="dat", domain=[i, "*"], records=dat_df)

    # Variables
    obj = Variable(m, name="obj", type="free")
    dev = Variable(m, name="dev", type="free", domain=[i])
    devp = Variable(m, name="devp", type="positive", domain=[i])
    devn = Variable(m, name="devn", type="positive", domain=[i])
    b = Variable(m, name="b", type="free", domain=[n])

    # Equations
    ddev = Equation(m, name="ddev", domain=[i])
    ddeva = Equation(m, name="ddeva", domain=[i])
    ls1 = Equation(m, name="ls1")
    ls1a = Equation(m, name="ls1a")
    ls2 = Equation(m, name="ls2")
    ls3 = Equation(m, name="ls3")
    ls4 = Equation(m, name="ls4")
    ls5 = Equation(m, name="ls5")
    ls5a = Equation(m, name="ls5a")
    ls6 = Equation(m, name="ls6")
    ls7 = Equation(m, name="ls7")
    ls8 = Equation(m, name="ls8")

    ddev[i] = dev[i] == dat[i, "y"] - Sum(n, b[n] * dat[i, n])

    ddeva[i] = devp[i] - devn[i] == dat[i, "y"] - Sum(n, b[n] * dat[i, n])

    ls1.expr = obj == Sum(i, gams_math.abs(dev[i]))

    ls1a.expr = obj == Sum(i, devp[i] + devn[i])

    ls2.expr = obj == Sum(i, gams_math.power(dev[i], 2))

    ls3.expr = obj == Sum(i, gams_math.power(gams_math.abs(dev[i]), 3))

    ls4.expr = obj == Sum(i, gams_math.power(dev[i], 4))

    ls5.expr = obj == Sum(i, gams_math.abs(dev[i] / dat[i, "y"]))

    ls5a.expr = obj == Sum(i, (devp[i] + devn[i]) / dat[i, "y"])

    ls6.expr = obj == Sum(i, gams_math.power(dev[i] / dat[i, "y"], 2))

    ls7.expr = obj == Sum(
        i, gams_math.power(gams_math.abs(dev[i] / dat[i, "y"]), 3)
    )

    ls8.expr = obj == Sum(i, gams_math.power(dev[i] / dat[i, "y"], 4))

    # Models
    mod1 = Model(
        m,
        name="mod1",
        equations=[ddev, ls1],
        problem="dnlp",
        sense=Sense.MIN,
        objective=obj,
    )
    mod1a = Model(
        m,
        name="mod1a",
        equations=[ddeva, ls1a],
        problem="lp",
        sense=Sense.MIN,
        objective=obj,
    )
    mod2 = Model(
        m,
        name="mod2",
        equations=[ddev, ls2],
        problem=Problem.NLP,
        sense=Sense.MIN,
        objective=obj,
    )
    mod3 = Model(
        m,
        name="mod3",
        equations=[ddev, ls3],
        problem="dnlp",
        sense=Sense.MIN,
        objective=obj,
    )
    mod4 = Model(
        m,
        name="mod4",
        equations=[ddev, ls4],
        problem=Problem.NLP,
        sense=Sense.MIN,
        objective=obj,
    )
    mod5 = Model(
        m,
        name="mod5",
        equations=[ddev, ls5],
        problem="dnlp",
        sense=Sense.MIN,
        objective=obj,
    )
    mod5a = Model(
        m,
        name="mod5a",
        equations=[ddeva, ls5a],
        problem="lp",
        sense=Sense.MIN,
        objective=obj,
    )
    mod6 = Model(
        m,
        name="mod6",
        equations=[ddev, ls6],
        problem=Problem.NLP,
        sense=Sense.MIN,
        objective=obj,
    )
    mod7 = Model(
        m,
        name="mod7",
        equations=[ddev, ls7],
        problem="dnlp",
        sense=Sense.MIN,
        objective=obj,
    )
    mod8 = Model(
        m,
        name="mod8",
        equations=[ddev, ls8],
        problem=Problem.NLP,
        sense=Sense.MIN,
        objective=obj,
    )

    # Reporting Parameter
    result = Parameter(m, name="result", domain=["*", "*"])

    b.l[n] = 1
    dev.l[i] = dat[i, "y"] - Sum(n, b.l[n] * dat[i, n])
    dev.up[i] = 100
    dev.lo[i] = -100
    devp.up[i] = 100
    devn.up[i] = 100

    m.addOptions({"limRow": 0, "limCol": 0})

    mod1.solve()
    result["mod1", n] = b.l[n]
    result["mod1", "obj"] = obj.l

    mod1a.solve()
    result["mod1a", n] = b.l[n]
    result["mod1a", "obj"] = obj.l

    mod2.solve()
    result["mod2", n] = b.l[n]
    result["mod2", "obj"] = obj.l

    mod3.solve()
    result["mod3", n] = b.l[n]
    result["mod3", "obj"] = obj.l

    mod4.solve()
    result["mod4", n] = b.l[n]
    result["mod4", "obj"] = obj.l

    mod5.solve()
    result["mod5", n] = b.l[n]
    result["mod5", "obj"] = obj.l

    mod5a.solve()
    result["mod5a", n] = b.l[n]
    result["mod5a", "obj"] = obj.l

    mod6.solve()
    result["mod6", n] = b.l[n]
    result["mod6", "obj"] = obj.l

    mod7.solve()
    result["mod7", n] = b.l[n]
    result["mod7", "obj"] = obj.l

    mod8.solve()
    result["mod8", n] = b.l[n]
    result["mod8", "obj"] = obj.l

    print(result.pivot())


if __name__ == "__main__":
    main()
