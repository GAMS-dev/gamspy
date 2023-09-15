"""
*** Optimal operation of energy hub

For more details please refer to Chapter 10 (Gcode10.3), of the following book:
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
import pandas as pd

from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Parameter
from gamspy import Set
from gamspy import Sum
from gamspy import Variable


def reformat_df(dataframe):
    return dataframe.reset_index().melt(
        id_vars="index", var_name="Category", value_name="Value"
    )


def data_records():
    # data records table
    cols = ["Dh", "De", "Dc", "lamda"]
    inds = [f"t{i}" for i in range(1, 25)]
    data = [
        [21.4, 52.1, 11.5, 36.7],
        [23.2, 66.7, 13.7, 40.4],
        [26.1, 72.2, 16.0, 38.5],
        [26.7, 78.4, 21.4, 38.0],
        [25.6, 120.2, 22.0, 40.2],
        [26.4, 83.5, 30.8, 38.6],
        [39.5, 110.4, 38.9, 52.3],
        [47.3, 124.3, 46.8, 67.3],
        [52.1, 143.6, 51.0, 70.5],
        [49.1, 149.3, 48.9, 66.2],
        [69.3, 154.2, 34.8, 73.3],
        [62.0, 147.3, 32.7, 60.8],
        [68.0, 200.7, 27.8, 63.2],
        [68.6, 174.4, 32.0, 70.8],
        [56.4, 176.5, 33.2, 63.1],
        [41.3, 136.1, 34.1, 52.5],
        [37.4, 108.7, 40.8, 57.0],
        [25.4, 96.9, 43.6, 49.2],
        [25.7, 89.1, 51.5, 47.5],
        [21.9, 82.5, 43.1, 49.5],
        [22.4, 76.9, 36.5, 53.1],
        [24.6, 66.8, 27.7, 51.6],
        [22.7, 47.2, 19.1, 50.5],
        [22.6, 64.7, 11.0, 36.4],
    ]
    data_recs = reformat_df(pd.DataFrame(data, columns=cols, index=inds))

    return data_recs


def main():
    m = Container()

    # SET #
    t = Set(
        m,
        name="t",
        records=[f"t{i}" for i in range(1, 25)],
        description="hours",
    )

    # DATA PARAMETER #
    data = Parameter(m, name="data", domain=[t, "*"], records=data_records())

    # SCALARS #
    CBmax = Parameter(m, name="CBmax", records=500)
    eta_ee = Parameter(m, name="eta_ee", records=0.98)
    eta_ghf = Parameter(m, name="eta_ghf", records=0.9)
    eta_hc = Parameter(m, name="eta_hc", records=0.95)

    # VARIABLES #
    cost = Variable(m, name="cost", type="free")

    E = Variable(m, name="E", type="positive", domain=[t])
    G = Variable(m, name="G", type="positive", domain=[t])
    H1 = Variable(m, name="H1", type="positive", domain=[t])
    H2 = Variable(m, name="H2", type="positive", domain=[t])

    H2.up[t] = CBmax

    # EQUATIONS #
    eq1 = Equation(m, name="eq1", type="regular")
    eq2 = Equation(m, name="eq2", type="regular", domain=[t])
    eq3 = Equation(m, name="eq3", type="regular", domain=[t])
    eq4 = Equation(m, name="eq4", type="regular", domain=[t])
    eq5 = Equation(m, name="eq5", type="regular", domain=[t])

    eq1.expr = cost == Sum(t, data[t, "lamda"] * E[t] + 12 * G[t])
    eq2[t] = eta_ee * E[t] == data[t, "De"]
    eq3[t] = H1[t] == data[t, "Dh"]
    eq4[t] = eta_ghf * G[t] == H1[t] + H2[t]
    eq5[t] = eta_hc * H2[t] == data[t, "Dc"]

    hub = Model(
        m,
        name="hub",
        equations=m.getEquations(),
        problem="lp",
        sense="min",
        objective=cost,
    )
    hub.solve()

    # Reporting Parameter
    report = Parameter(m, name="report", domain=[t, "*"])
    report[t, "E"] = E.l[t]
    report[t, "G"] = G.l[t]
    report[t, "h1"] = H1.l[t]
    report[t, "h2"] = H2.l[t]

    print("report:  \n", report.pivot().round(4))


if __name__ == "__main__":
    main()
