"""
Cost based Dynamic Economic Dispatch integrated with Energy Storage and Wind


For more details please refer to Chapter 7 (Gcode7.2), of the following book:
Soroudi, Alireza. Power System Optimization Modeling in GAMS. Springer, 2017.
--------------------------------------------------------------------------------
Model type: QCP
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
from gamspy import Ord
from gamspy import Parameter
from gamspy import Set
from gamspy import Sum
from gamspy import Variable


def reformat_df(dataframe):
    return dataframe.reset_index().melt(
        id_vars="index", var_name="Category", value_name="Value"
    )


def data_records():
    # gendata records table
    cols = ["a", "b", "c", "d", "e", "f", "Pmin", "Pmax", "RU0", "RD0"]
    inds = [f"p{i}" for i in range(1, 5)]
    data = [
        [0.12, 14.80, 89, 1.2, -5.0, 3.0, 28, 200, 40, 40],
        [0.17, 16.57, 83, 2.3, -4.24, 6.09, 20, 290, 30, 30],
        [0.15, 15.55, 100, 1.1, -2.15, 5.69, 30, 190, 30, 30],
        [0.19, 16.21, 70, 1.1, -3.99, 6.2, 20, 260, 50, 50],
    ]
    gen_recs = reformat_df(pd.DataFrame(data, columns=cols, index=inds))

    # data records table
    cols = ["lamda", "load", "wind"]
    inds = [f"t{i}" for i in range(1, 25)]
    data = [
        [32.71, 510, 44.1],
        [34.72, 530, 48.5],
        [32.71, 516, 65.7],
        [32.74, 510, 144.9],
        [32.96, 515, 202.3],
        [34.93, 544, 317.3],
        [44.9, 646, 364.4],
        [52.0, 686, 317.3],
        [53.03, 741, 271.0],
        [47.26, 734, 306.9],
        [44.07, 748, 424.1],
        [38.63, 760, 398.0],
        [39.91, 754, 487.6],
        [39.45, 700, 521.9],
        [41.14, 686, 541.3],
        [39.23, 720, 560.0],
        [52.12, 714, 486.8],
        [40.85, 761, 372.6],
        [41.2, 727, 367.4],
        [41.15, 714, 314.3],
        [45.76, 618, 316.6],
        [45.59, 584, 311.4],
        [45.56, 578, 405.4],
        [34.72, 544, 470.4],
    ]
    data_recs = reformat_df(pd.DataFrame(data, columns=cols, index=inds))

    return gen_recs, data_recs


def main():
    m = Container(delayed_execution=True)

    # SETS #
    t = Set(
        m,
        name="t",
        records=[f"t{i}" for i in range(1, 25)],
        description="hours",
    )
    g = Set(
        m,
        name="g",
        records=[f"p{i}" for i in range(1, 5)],
        description="thermal units",
    )

    # PARAMETERS #
    gendata = Parameter(
        m,
        name="gendata",
        domain=[g, "*"],
        records=data_records()[0],
        description="generator cost characteristics and limits",
    )
    data = Parameter(
        m, name="data", domain=[t, "*"], records=data_records()[1]
    )

    # VARIABLES #
    cost = Variable(
        m, name="cost", type="free", description="cost of thermal units"
    )
    p = Variable(
        m,
        name="p",
        type="free",
        domain=[g, t],
        description="power generated by thermal power plant",
    )
    SOC = Variable(m, name="SOC", type="free", domain=[t])
    Pd = Variable(m, name="Pd", type="free", domain=[t])
    Pc = Variable(m, name="Pc", type="free", domain=[t])
    Pw = Variable(m, name="Pw", type="free", domain=[t])
    PWC = Variable(m, name="PWC", type="free", domain=[t])

    p.up[g, t] = gendata[g, "Pmax"]
    p.lo[g, t] = gendata[g, "Pmin"]

    # SCALARS #
    SOC0 = Parameter(m, name="SOC0", records=100)
    SOCmax = Parameter(m, name="SOCmax", records=300)
    eta_c = Parameter(m, name="eta_c", records=0.95)
    eta_d = Parameter(m, name="eta_d", records=0.9)
    VWC = Parameter(m, name="VWC", records=50)

    SOC.up[t] = SOCmax
    SOC.lo[t] = 0.2 * SOCmax
    SOC.fx["t24"] = SOC0

    Pc.up[t] = 0.2 * SOCmax
    Pc.lo[t] = 0
    Pd.up[t] = 0.2 * SOCmax
    Pd.lo[t] = 0
    Pw.up[t] = data[t, "wind"]
    Pw.lo[t] = 0
    PWC.up[t] = data[t, "wind"]
    PWC.lo[t] = 0

    # EQUATIONS #
    Genconst3 = Equation(m, name="Genconst3", type="regular", domain=[g, t])
    Genconst4 = Equation(m, name="Genconst4", type="regular", domain=[g, t])
    costThermalcalc = Equation(m, name="costThermalcalc", type="regular")
    constESS = Equation(m, name="constESS", type="regular", domain=[t])
    balance = Equation(m, name="balance", type="regular", domain=[t])
    wind = Equation(m, name="wind", type="regular", domain=[t])

    costThermalcalc.definition = cost == Sum(t, VWC * PWC[t]) + Sum(
        [t, g],
        gendata[g, "a"] * gams_math.power(p[g, t], 2)
        + gendata[g, "b"] * p[g, t]
        + gendata[g, "c"],
    )

    Genconst3[g, t] = p[g, t.lead(1)] - p[g, t] <= gendata[g, "RU0"]

    Genconst4[g, t] = p[g, t.lag(1)] - p[g, t] <= gendata[g, "RD0"]

    constESS[t] = (
        SOC[t]
        == SOC0.where[Ord(t) == 1]
        + SOC[t.lag(1)].where[Ord(t) > 1]
        + Pc[t] * eta_c
        - Pd[t] / eta_d
    )

    balance[t] = Pw[t] + Sum(g, p[g, t]) + Pd[t] >= data[t, "load"] + Pc[t]

    wind[t] = Pw[t] + PWC[t] == data[t, "wind"]

    DEDESScostbased = Model(
        m,
        name="DEDESScostbased",
        equations=m.getEquations(),
        problem="qcp",
        sense="min",
        objective=cost,
    )
    DEDESScostbased.solve()

    import math

    assert math.isclose(
        DEDESScostbased.objective_value, 223360.0645, rel_tol=0.001
    )

    # Reporting parameter
    rep = Parameter(m, name="rep", domain=[t, "*"])
    rep[t, "Pth"] = Sum(g, p.l[g, t])
    rep[t, "SOC"] = SOC.l[t]
    rep[t, "Pd"] = Pd.l[t]
    rep[t, "Pc"] = Pc.l[t]
    rep[t, "Pw"] = Pw.l[t]
    rep[t, "Pwc"] = PWC.l[t]
    rep[t, "Load"] = data[t, "load"]

    # Export results to an excel file
    writer = pd.ExcelWriter("DEDESScostbased.xlsx", engine="openpyxl")
    p.pivot().round(4).to_excel(writer, sheet_name="Pthermal")
    rep.pivot().round(4).to_excel(writer, sheet_name="rep")

    # close writer agent
    writer.close()


if __name__ == "__main__":
    main()
