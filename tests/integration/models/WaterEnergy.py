"""
*** Water-Energy Nexus

For more details please refer to Chapter 10 (Gcode10.1), of the following book:
Soroudi, Alireza. Power System Optimization Modeling in GAMS. Springer, 2017.
--------------------------------------------------------------------------------
Model type: MINLP
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
import pandas as pd

import gamspy.math as gams_math
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Parameter
from gamspy import Set
from gamspy import Sum
from gamspy import Variable


def sqr(x):
    return gams_math.power(x, 2)


def reformat_df(dataframe):
    return dataframe.reset_index().melt(
        id_vars="index", var_name="Category", value_name="Value"
    )


def data_records():
    # gendata records table
    cols = ["a", "b", "c", "Pmax", "Pmin"]
    inds = [f"p{p}" for p in range(1, 5)]
    data = [
        [0.0002069, -0.1483, 57.11, 500, 0],
        [0.0003232, -0.1854, 57.11, 400, 0],
        [0.001065, -0.6026, 126.8, 400, 0],
        [0.0004222, -0.2119, 57.11, 350, 0],
    ]
    gendata_recs = reformat_df(pd.DataFrame(data, columns=cols, index=inds))

    # Coproduct records table
    cols = [
        "Pmax",
        "Pmin",
        "Wmax",
        "Wmin",
        "rmin",
        "rmax",
        "A11",
        "A12",
        "A22",
        "b1",
        "b2",
        "C",
    ]
    inds = [f"c{c}" for c in range(1, 4)]
    data = [
        [
            800,
            160,
            200,
            30,
            4,
            9,
            0.0004433,
            0.003546,
            0.007093,
            -1.106,
            -4.426,
            737.4,
        ],
        [
            600,
            120,
            150,
            23,
            4,
            9,
            0.0007881,
            0.006305,
            0.01261,
            -1.475,
            -5.901,
            737.4,
        ],
        [
            400,
            80,
            100,
            15,
            4,
            9,
            0.001773,
            0.01419,
            0.02837,
            -2.213,
            -8.851,
            737.4,
        ],
    ]
    Coproduct_recs = reformat_df(pd.DataFrame(data, columns=cols, index=inds))

    # Waterdata records table
    cols = ["a", "b", "c", "Wmax", "Wmin"]
    inds = ["w1"]
    data = [[1.82e-02, -7.081e-1, 7.374, 250, 0]]
    waterdata_recs = reformat_df(pd.DataFrame(data, columns=cols, index=inds))

    # pwdata records table
    cols = ["Pd", "water"]
    inds = [f"t{t}" for t in range(1, 11)] + [f"t{t}" for t in range(16, 25)]
    data = [
        [1250, 150],
        [1125, 130],
        [875, 100],
        [750, 150],
        [950, 200],
        [1440, 350],
        [1500, 300],
        [1750, 200],
        [2000, 300],
        [2250, 400],
        [2500, 550],
        [2125, 550],
        [2375, 500],
        [2250, 400],
        [1975, 350],
        [1750, 300],
        [1625, 250],
        [1500, 200],
        [1376, 150],
    ]
    pwdata_recs = reformat_df(pd.DataFrame(data, columns=cols, index=inds))

    return gendata_recs, Coproduct_recs, waterdata_recs, pwdata_recs


def main():
    m = Container()

    # SETS #
    t = Set(m, name="t", records=[f"t{t}" for t in range(1, 25)])
    i = Set(m, name="i", records=[f"p{p}" for p in range(1, 5)])
    c = Set(m, name="c", records=[f"c{c}" for c in range(1, 4)])
    w = Set(m, name="w", records=["w1"])

    # PARAMETERS #
    gendata = Parameter(
        m,
        name="gendata",
        domain=[i, "*"],
        records=data_records()[0],
        description="generator cost characteristics and limits",
    )
    Coproduct = Parameter(
        m, name="Coproduct", domain=[c, "*"], records=data_records()[1]
    )
    waterdata = Parameter(
        m, name="waterdata", domain=[w, "*"], records=data_records()[2]
    )
    PWdata = Parameter(
        m, name="PWdata", domain=[t, "*"], records=data_records()[3]
    )

    # FREE VARIABLES #
    of = Variable(m, name="of")
    TC = Variable(m, name="TC")
    CC = Variable(m, name="CC")
    WaterCost = Variable(m, name="WaterCost")

    # BINARY VARIABLES #
    Up = Variable(m, name="Up", type="binary", domain=[i, t])
    Uc = Variable(m, name="Uc", type="binary", domain=[c, t])
    Uw = Variable(m, name="Uw", type="binary", domain=[w, t])

    # POSITIVE VARIABLES #
    p = Variable(m, name="p", type="positive", domain=[i, t])
    Pc = Variable(m, name="Pc", type="positive", domain=[c, t])
    Wc = Variable(m, name="Wc", type="positive", domain=[c, t])
    Water = Variable(m, name="Water", type="positive", domain=[w, t])

    p.up[i, t] = gendata[i, "Pmax"]
    Pc.up[c, t] = Coproduct[c, "Pmax"]
    Wc.up[c, t] = Coproduct[c, "Wmax"]
    Water.up[w, t] = waterdata[w, "Wmax"]

    # EQUATIONS #
    costThermal = Equation(m, name="costThermal", type="regular")
    balanceP = Equation(m, name="balanceP", type="regular", domain=[t])
    balanceW = Equation(m, name="balanceW", type="regular", domain=[t])
    costCoprodcalc = Equation(m, name="costCoprodcalc", type="regular")
    Objective = Equation(m, name="Objective", type="regular")
    costwatercalc = Equation(m, name="costwatercalc", type="regular")
    ratio1 = Equation(m, name="ratio1", type="regular", domain=[c, t])
    ratio2 = Equation(m, name="ratio2", type="regular", domain=[c, t])
    eq1 = Equation(m, name="eq1", type="regular", domain=[w, t])
    eq2 = Equation(m, name="eq2", type="regular", domain=[w, t])
    eq3 = Equation(m, name="eq3", type="regular", domain=[c, t])
    eq4 = Equation(m, name="eq4", type="regular", domain=[c, t])
    eq5 = Equation(m, name="eq5", type="regular", domain=[c, t])
    eq6 = Equation(m, name="eq6", type="regular", domain=[c, t])
    eq7 = Equation(m, name="eq7", type="regular", domain=[i, t])
    eq8 = Equation(m, name="eq8", type="regular", domain=[i, t])

    costThermal.expr = TC == Sum(
        [t, i],
        gendata[i, "a"] * sqr(p[i, t])
        + gendata[i, "b"] * p[i, t]
        + gendata[i, "c"] * Up[i, t],
    )
    balanceP[t] = Sum(i, p[i, t]) + Sum(c, Pc[c, t]) == PWdata[t, "Pd"]
    balanceW[t] = Sum(w, Water[w, t]) + Sum(c, Wc[c, t]) == PWdata[t, "water"]
    costCoprodcalc.expr = CC == (
        Sum(
            [c, t],
            Coproduct[c, "A11"] * sqr(Pc[c, t])
            + 2 * Coproduct[c, "A12"] * Pc[c, t] * Wc[c, t]
            + Coproduct[c, "A22"] * sqr(Wc[c, t])
            + Coproduct[c, "B1"] * Pc[c, t]
            + Coproduct[c, "B2"] * Wc[c, t]
            + Coproduct[c, "C"] * Uc[c, t],
        )
    )
    costwatercalc.expr = WaterCost == Sum(
        [t, w],
        waterdata[w, "a"] * sqr(Water[w, t])
        + waterdata[w, "b"] * Water[w, t]
        + waterdata[w, "c"] * Uw[w, t],
    )
    Objective.expr = of == TC + CC + WaterCost
    ratio1[c, t] = Pc[c, t] <= Wc[c, t] * Coproduct[c, "Rmax"]
    ratio2[c, t] = Pc[c, t] >= Wc[c, t] * Coproduct[c, "Rmin"]
    eq1[w, t] = Water[w, t] <= Uw[w, t] * waterdata[w, "Wmax"]
    eq2[w, t] = Water[w, t] >= Uw[w, t] * waterdata[w, "Wmin"]
    eq3[c, t] = Wc[c, t] <= Uc[c, t] * Coproduct[c, "Wmax"]
    eq4[c, t] = Wc[c, t] >= Uc[c, t] * Coproduct[c, "Wmin"]
    eq5[c, t] = Pc[c, t] <= Uc[c, t] * Coproduct[c, "Pmax"]
    eq6[c, t] = Pc[c, t] >= Uc[c, t] * Coproduct[c, "Pmin"]
    eq7[i, t] = p[i, t] <= Up[i, t] * gendata[i, "Pmax"]
    eq8[i, t] = p[i, t] >= Up[i, t] * gendata[i, "Pmin"]

    DEDcostbased = Model(
        m,
        name="DEDcostbased",
        equations=m.getEquations(),
        problem="minlp",
        sense="min",
        objective=of,
    )
    DEDcostbased.solve()

    print("Objective Function Value:  ", round(of.toValue(), 4))


if __name__ == "__main__":
    main()
