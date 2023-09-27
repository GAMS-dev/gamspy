"""
*** Multi-objective Economic-Environmental Load Dispatch

For more details please refer to Chapter 3 (Gcode3.4), of the following book:
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

from gamspy import Card
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

    # SETS #
    gen = Set(m, name="gen", records=[f"g{i}" for i in range(1, 6)])
    counter = Set(m, name="counter", records=[f"c{i}" for i in range(1, 12)])

    # PARAMETERS
    report = Parameter(m, name="report", domain=["*"])
    rep = Parameter(m, name="rep", domain=[counter, "*"])
    rep2 = Parameter(m, name="rep2", domain=[counter, gen])
    data = Parameter(m, name="data", domain=[gen, "*"], records=data_records())

    # SCALARS
    load = Parameter(m, name="load", records=400)
    Elim = Parameter(m, name="Elim")

    # VARIABLES #
    P = Variable(m, name="P", domain=[gen])
    TE = Variable(m, name="TE")
    TC = Variable(m, name="TC")

    # EQUATIONS #
    eq1 = Equation(m, name="eq1", type="regular")
    eq2 = Equation(m, name="eq2", type="regular")
    eq3 = Equation(m, name="eq3", type="regular")

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

    P.lo[gen] = data[gen, "Pmin"]
    P.up[gen] = data[gen, "Pmax"]

    END1 = Model(
        m,
        name="END1",
        equations=[eq1, eq2, eq3],
        problem="qcp",
        sense="min",
        objective=TC,
    )
    END1.solve()
    report["maxTE"] = TE.l
    report["minTC"] = TC.l

    END2 = Model(
        m,
        name="END2",
        equations=[eq1, eq2, eq3],
        problem="qcp",
        sense="min",
        objective=TE,
    )
    END2.solve()
    report["maxTC"] = TC.l
    report["minTE"] = TE.l

    for idx, cc in enumerate(counter.toList()):
        Elim.assign = (report["maxTE"] - report["minTE"]) * (idx) / (
            Card(counter) - 1
        ) + report["minTE"]
        TE.up.assign = Elim
        END1.solve()
        rep[cc, "TC"] = TC.l
        rep[cc, "TE"] = TE.l
        rep2[cc, gen] = P.l[gen]

    print("rep  \n", rep.pivot().round(4))
    print("rep2  \n", rep2.pivot().round(4))


if __name__ == "__main__":
    main()
