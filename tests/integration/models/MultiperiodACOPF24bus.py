"""
*** Multi-period AC-OPF for IEEE 24-bus network considering wind and load shedding

For more details please refer to Chapter 6 (Gcode6.7), of the following book:
Soroudi, Alireza. Power System Optimization Modeling in GAMS. Springer, 2017.
--------------------------------------------------------------------------------
Model type: NLP
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
import math

import pandas as pd

import gamspy.math as gams_math
from gamspy import Alias
from gamspy import Card
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Ord
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
    # GenD records table
    cols = ["pmax", "pmin", "b", "Qmax", "Qmin", "Vg", "RU", "RD"]
    inds = ["1", "2", "7", "13", "15", "16", "18", "21", "22", "23"]
    data = [
        [152, 30.4, 13.32, 192, -50, 1.035, 21, 21],
        [152, 30.4, 13.32, 192, -50, 1.035, 21, 21],
        [350, 75.0, 20.7, 300, 0, 1.025, 43, 43],
        [591, 206.85, 20.93, 591, 0, 1.02, 31, 31],
        [215, 66.25, 21.0, 215, -100, 1.014, 31, 31],
        [155, 54.25, 10.52, 155, -50, 1.017, 31, 31],
        [400, 100.0, 5.47, 400, -50, 1.05, 70, 70],
        [400, 100.0, 5.47, 400, -50, 1.05, 70, 70],
        [300, 0.0, 0.0, 300, -60, 1.05, 53, 53],
        [360, 248.5, 10.52, 310, -125, 1.05, 31, 31],
    ]
    GenD_recs = reformat_df(pd.DataFrame(data, columns=cols, index=inds))

    # BD records table
    cols = ["Pd", "Qd"]
    inds = [str(ii) for ii in range(1, 25)]
    data = [
        [108, 22],
        [97, 20],
        [180, 37],
        [74, 15],
        [71, 14],
        [136, 28],
        [125, 25],
        [171, 35],
        [175, 36],
        [195, 40],
        [0, 0],
        [0, 0],
        [265, 54],
        [194, 39],
        [317, 64],
        [100, 20],
        [0, 0],
        [333, 68],
        [181, 37],
        [128, 26],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
    ]
    BD_recs = reformat_df(pd.DataFrame(data, columns=cols, index=inds))

    # LN records table
    cols = ["r", "x", "b", "limit"]
    inds = [
        ("1", "2"),
        ("1", "3"),
        ("1", "5"),
        ("2", "4"),
        ("2", "6"),
        ("3", "9"),
        ("3", "24"),
        ("4", "9"),
        ("5", "10"),
        ("6", "10"),
        ("7", "8"),
        ("8", "9"),
        ("8", "10"),
        ("9", "11"),
        ("9", "12"),
        ("10", "11"),
        ("10", "12"),
        ("11", "13"),
        ("11", "14"),
        ("12", "13"),
        ("12", "23"),
        ("13", "23"),
        ("14", "16"),
        ("15", "16"),
        ("15", "21"),
        ("15", "24"),
        ("16", "17"),
        ("16", "19"),
        ("17", "18"),
        ("17", "22"),
        ("18", "21"),
        ("19", "20"),
        ("20", "23"),
        ("21", "22"),
    ]
    data = [
        [0.0026, 0.0139, 0.4611, 175],
        [0.0546, 0.2112, 0.0572, 175],
        [0.0218, 0.0845, 0.0229, 175],
        [0.0328, 0.1267, 0.0343, 175],
        [0.0497, 0.192, 0.052, 175],
        [0.0308, 0.119, 0.0322, 175],
        [0.0023, 0.0839, 0.0, 400],
        [0.0268, 0.1037, 0.0281, 175],
        [0.0228, 0.0883, 0.0239, 175],
        [0.0139, 0.0605, 2.459, 175],
        [0.0159, 0.0614, 0.0166, 175],
        [0.0427, 0.1651, 0.0447, 175],
        [0.0427, 0.1651, 0.0447, 175],
        [0.0023, 0.0839, 0.0, 400],
        [0.0023, 0.0839, 0.0, 400],
        [0.0023, 0.0839, 0.0, 400],
        [0.0023, 0.0839, 0.0, 400],
        [0.0061, 0.0476, 0.0999, 500],
        [0.0054, 0.0418, 0.0879, 500],
        [0.0061, 0.0476, 0.0999, 500],
        [0.0124, 0.0966, 0.203, 500],
        [0.0111, 0.0865, 0.1818, 500],
        [0.005, 0.0389, 0.0818, 500],
        [0.0022, 0.0173, 0.0364, 500],
        [0.00315, 0.0245, 0.206, 1000],
        [0.0067, 0.0519, 0.1091, 500],
        [0.0033, 0.0259, 0.0545, 500],
        [0.003, 0.0231, 0.0485, 500],
        [0.0018, 0.0144, 0.0303, 500],
        [0.0135, 0.1053, 0.2212, 500],
        [0.00165, 0.01295, 0.109, 1000],
        [0.00255, 0.0198, 0.1666, 1000],
        [0.0014, 0.0108, 0.091, 1000],
        [0.0087, 0.0678, 0.1424, 500],
    ]
    inds = pd.MultiIndex.from_tuples(inds, names=["Index1", "Index2"])
    LN_recs = pd.DataFrame(data, columns=cols, index=inds)
    LN_recs.reset_index(inplace=True)
    LN_recs = LN_recs.melt(
        id_vars=["Index1", "Index2"], value_vars=["r", "x", "b", "limit"]
    )

    # WD records table
    cols = ["w", "d"]
    inds = [f"t{tt}" for tt in range(1, 25)]
    data = [
        [0.0786666666666667, 0.684511335492475],
        [0.0866666666666667, 0.644122690036197],
        [0.117333333333333, 0.6130691560297],
        [0.258666666666667, 0.599733282530006],
        [0.361333333333333, 0.588874071251667],
        [0.566666666666667, 0.5980186702229],
        [0.650666666666667, 0.626786054486569],
        [0.566666666666667, 0.651743189178891],
        [0.484, 0.706039245570585],
        [0.548, 0.787007048961707],
        [0.757333333333333, 0.839016955610593],
        [0.710666666666667, 0.852733854067441],
        [0.870666666666667, 0.870642027052772],
        [0.932, 0.834254143646409],
        [0.966666666666667, 0.816536483139646],
        [1.0, 0.819394170318156],
        [0.869333333333333, 0.874071251666984],
        [0.665333333333333, 1.0],
        [0.656, 0.983615926843208],
        [0.561333333333333, 0.936368832158506],
        [0.565333333333333, 0.887597637645266],
        [0.556, 0.809297008954087],
        [0.724, 0.74585635359116],
        [0.84, 0.733473042484283],
    ]
    WD_recs = reformat_df(pd.DataFrame(data, columns=cols, index=inds))

    return GenD_recs, BD_recs, LN_recs, WD_recs


def main():
    m = Container()

    # SETS #
    i = Set(
        m,
        name="i",
        records=[str(ii) for ii in range(1, 25)],
        description="network buses",
    )
    slack = Set(m, name="slack", domain=[i], records=[13])
    t = Set(m, name="t", records=[f"t{tt}" for tt in range(1, 25)])

    # ALIAS #
    j = Alias(m, name="j", alias_with=i)

    # SCALARS #
    Sbase = Parameter(m, name="Sbase", records=100)

    # PARAMETERS
    GenD = Parameter(
        m,
        name="GenD",
        domain=[i, "*"],
        records=data_records()[0],
        description="generating units characteristics",
    )
    BD = Parameter(
        m,
        name="BD",
        domain=[i, "*"],
        records=data_records()[1],
        description="demands of each bus in MW",
    )
    LN = Parameter(
        m,
        name="LN",
        domain=[i, j, "*"],
        records=data_records()[2],
        description="network technical characteristics",
    )
    WD = Parameter(m, name="WD", domain=[t, "*"], records=data_records()[3])
    Wcap = Parameter(
        m,
        name="Wcap",
        domain=[i],
        records=[("8", 200), ("19", 150), ("21", 100)],
    )
    cx = Parameter(m, name="cx", domain=[i, j])

    LN[i, j, "x"].where[LN[i, j, "x"] == 0] = LN[j, i, "x"]
    LN[i, j, "r"].where[LN[i, j, "r"] == 0] = LN[j, i, "r"]
    LN[i, j, "b"].where[LN[i, j, "b"] == 0] = LN[j, i, "b"]
    LN[i, j, "Limit"].where[LN[i, j, "Limit"] == 0] = LN[j, i, "Limit"]
    LN[i, j, "bij"].where[LN[i, j, "Limit"]] = 1 / LN[i, j, "x"]
    LN[i, j, "z"].where[LN[i, j, "Limit"]] = gams_math.sqrt(
        sqr(LN[i, j, "x"]) + sqr(LN[i, j, "r"])
    )
    LN[j, i, "z"].where[LN[i, j, "z"] == 0] = LN[i, j, "z"]
    LN[i, j, "th"].where[
        (LN[i, j, "Limit"]) & (LN[i, j, "x"]) & (LN[i, j, "r"])
    ] = gams_math.atan(LN[i, j, "x"] / (LN[i, j, "r"]))
    LN[i, j, "th"].where[
        (LN[i, j, "Limit"]) & (LN[i, j, "x"]) & (LN[i, j, "r"] == 0)
    ] = (math.pi / 2)
    LN[i, j, "th"].where[
        (LN[i, j, "Limit"]) & (LN[i, j, "r"]) & (LN[i, j, "x"] == 0)
    ] = 0
    LN[j, i, "th"].where[LN[i, j, "Limit"]] = LN[i, j, "th"]

    cx[i, j].where[(LN[i, j, "limit"]) & (LN[j, i, "limit"])] = 1
    cx[i, j].where[cx[j, i]] = 1

    # VARIABLES #
    OF = Variable(m, name="OF")
    Pij = Variable(m, name="Pij", domain=[i, j, t])
    Qij = Variable(m, name="Qij", domain=[i, j, t])
    Pg = Variable(m, name="Pg", domain=[i, t])
    Qg = Variable(m, name="Qg", domain=[i, t])
    Va = Variable(m, name="Va", domain=[i, t])
    V = Variable(m, name="V", domain=[i, t])
    Pw = Variable(m, name="Pw", domain=[i, t])

    # EQUATIONS #
    eq1 = Equation(m, name="eq1", type="regular", domain=[i, j, t])
    eq2 = Equation(m, name="eq2", type="regular", domain=[i, j, t])
    eq3 = Equation(m, name="eq3", type="regular", domain=[i, t])
    eq4 = Equation(m, name="eq4", type="regular", domain=[i, t])
    eq5 = Equation(m, name="eq5", type="regular")
    eq6 = Equation(m, name="eq6", type="regular", domain=[i, t])
    eq7 = Equation(m, name="eq7", type="regular", domain=[i, t])

    eq1[i, j, t].where[cx[i, j]] = (
        Pij[i, j, t]
        == (
            V[i, t] * V[i, t] * gams_math.cos(LN[j, i, "th"])
            - V[i, t]
            * V[j, t]
            * gams_math.cos(Va[i, t] - Va[j, t] + LN[j, i, "th"])
        )
        / LN[j, i, "z"]
    )

    eq2[i, j, t].where[cx[i, j]] = (
        Qij[i, j, t]
        == (
            V[i, t] * V[i, t] * gams_math.sin(LN[j, i, "th"])
            - V[i, t]
            * V[j, t]
            * gams_math.sin(Va[i, t] - Va[j, t] + LN[j, i, "th"])
        )
        / LN[j, i, "z"]
        - LN[j, i, "b"] * V[i, t] * V[i, t] / 2
    )

    eq3[i, t] = Pw[i, t].where[Wcap[i]] + Pg[i, t].where[GenD[i, "Pmax"]] - WD[
        t, "d"
    ] * BD[i, "pd"] / Sbase == Sum(j.where[cx[j, i]], Pij[i, j, t])

    eq4[i, t] = Qg[i, t].where[GenD[i, "Qmax"]] - WD[t, "d"] * BD[
        i, "qd"
    ] / Sbase == Sum(j.where[cx[j, i]], Qij[i, j, t])

    eq5.expr = OF >= Sum(
        [i, t], Pg[i, t] * GenD[i, "b"] * Sbase.where[GenD[i, "Pmax"]]
    )

    eq6[i, t].where[(GenD[i, "Pmax"]) & (Ord(t) > 1)] = (
        Pg[i, t] - Pg[i, t.lag(1)] <= GenD[i, "RU"] / Sbase
    )

    eq7[i, t].where[(GenD[i, "Pmax"]) & (Ord(t) < Card(t))] = (
        Pg[i, t] - Pg[i, t.lead(1)] <= GenD[i, "RD"] / Sbase
    )

    loadflow = Model(
        m,
        name="loadflow",
        equations=[eq1, eq2, eq3, eq4, eq5, eq6, eq7],
        problem="nlp",
        sense="min",
        objective=OF,
    )

    Pg.lo[i, t] = GenD[i, "Pmin"] / Sbase
    Pg.up[i, t] = GenD[i, "Pmax"] / Sbase
    Qg.lo[i, t] = GenD[i, "Qmin"] / Sbase
    Qg.up[i, t] = GenD[i, "Qmax"] / Sbase

    Va.up[i, t] = math.pi / 2
    Va.lo[i, t] = -math.pi / 2
    Va.l[i, t] = 0
    Va.fx[slack, t] = 0

    Pij.up[i, j, t].where[cx[i, j]] = 1 * LN[i, j, "Limit"] / Sbase
    Pij.lo[i, j, t].where[cx[i, j]] = -1 * LN[i, j, "Limit"] / Sbase
    Qij.up[i, j, t].where[cx[i, j]] = 1 * LN[i, j, "Limit"] / Sbase
    Qij.lo[i, j, t].where[cx[i, j]] = -1 * LN[i, j, "Limit"] / Sbase

    V.lo[i, t] = 0.9
    V.up[i, t] = 1.1
    V.l[i, t] = 1
    Pw.up[i, t] = WD[t, "w"] * Wcap[i] / Sbase
    Pw.lo[i, t] = 0

    loadflow.solve()

    # Reporting Parameters
    report = Parameter(m, name="report", domain=[t, i, "*"])
    report2 = Parameter(m, name="report2", domain=[i, t])
    report3 = Parameter(m, name="report3", domain=[i, t])

    report[t, i, "V"] = V.l[i, t]
    report[t, i, "Angle"] = Va.l[i, t]
    report[t, i, "Pg"] = Pg.l[i, t] * Sbase
    report[t, i, "Gg"] = Qg.l[i, t] * Sbase
    report[t, i, "LMP_P"] = eq3.m[i, t] / Sbase
    report[t, i, "LMP_Q"] = eq4.m[i, t] / Sbase
    report2[i, t] = Pg.l[i, t] * Sbase
    report3[i, t] = Qg.l[i, t] * Sbase
    print("report  \n", report.pivot().round(4))

    writer = pd.ExcelWriter("results.xlsx", engine="openpyxl")

    report.pivot().round(4).to_excel(writer, sheet_name="classic")
    report2.pivot().round(4).to_excel(writer, sheet_name="classic2")
    report3.pivot().round(4).to_excel(writer, sheet_name="classic3")

    writer.close()


if __name__ == "__main__":
    main()
