"""
*** Environmental Load Dispatch

For more details please refer to Chapter 3 (Gcode3.3), of the following book:
Soroudi, Alireza. Power System Optimization Modeling in GAMS. Springer, 2017.
--------------------------------------------------------------------------------
Model type: QCP
--------------------------------------------------------------------------------
Contributed by
Dr. Alireza Soroudi
IEEE Senior Member
Email: alireza.soroudi@gmail.com
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
    cols = ["a", "b", "c", "d", "e", "f", "Pmin", "Pmax"]
    inds = [f"g{i}" for i in range(1, 6)]
    data = [
        [3.0, 20.0, 100.0, 2.0, -5.0, 3.0, 28, 206],
        [4.05, 18.07, 98.87, 3.82, -4.24, 6.09, 90, 284],
        [4.05, 15.55, 104.26, 5.01, -2.15, 5.69, 68, 189],
        [3.99, 19.21, 107.21, 1.1, -3.99, 6.2, 76, 266],
        [3.88, 26.18, 95.31, 3.55, -6.88, 5.57, 19, 53],
    ]
    data_recs = reformat_df(pd.DataFrame(data, columns=cols, index=inds))

    return data_recs


def main():
    m = Container()

    # SET #
    gen = Set(m, name="gen", records=[f"g{i}" for i in range(1, 6)])

    # PARAMETERS #
    report = Parameter(m, name="report", domain=[gen, "*"])
    data = Parameter(m, name="data", domain=[gen, "*"], records=data_records())

    # SCALARS #
    load = Parameter(m, name="load", records=400)
    Eprice = Parameter(m, name="Eprice", records=0.1)

    # VARIABLES #
    P = Variable(m, name="P", type="free", domain=[gen])
    OF = Variable(m, name="OF", type="free")
    TE = Variable(m, name="TE", type="free")
    TC = Variable(m, name="TC", type="free")

    # EQUATIONS #
    eq1 = Equation(m, name="eq1", type="regular")
    eq2 = Equation(m, name="eq2", type="regular")
    eq3 = Equation(m, name="eq3", type="regular")
    eq4 = Equation(m, name="eq4", type="regular")

    eq1.expr = TC == Sum(
        gen,
        data[gen, "a"] * P[gen] * P[gen]
        + data[gen, "b"] * P[gen]
        + data[gen, "c"],
    )
    eq2.expr = Sum(gen, P[gen]) >= load
    eq3.expr = TE == Sum(
        gen,
        data[gen, "d"] * P[gen] * P[gen]
        + data[gen, "e"] * P[gen]
        + data[gen, "f"],
    )
    eq4.expr = OF == TC + TE * Eprice

    P.lo[gen] = data[gen, "Pmin"]
    P.up[gen] = data[gen, "Pmax"]

    END1 = Model(
        m,
        name="END1",
        equations=[eq1, eq2, eq3, eq4],
        problem="qcp",
        sense="min",
        objective=TC,
    )
    END1.solve()
    report[gen, "ED"] = P.l[gen]

    END2 = Model(
        m,
        name="END2",
        equations=[eq1, eq2, eq3, eq4],
        problem="qcp",
        sense="min",
        objective=TE,
    )
    END2.solve()
    report[gen, "END"] = P.l[gen]

    END3 = Model(
        m,
        name="END3",
        equations=[eq1, eq2, eq3, eq4],
        problem="qcp",
        sense="min",
        objective=OF,
    )
    END3.solve()
    report[gen, "penalty"] = P.l[gen]

    TE.up.assign = 90000
    END1.solve()
    report[gen, "limit"] = P.l[gen]

    print("report  \n", report.pivot().round(4))


if __name__ == "__main__":
    main()
