"""
*** Sensitivity Analysis in Economic Load Dispatch

For more details please refer to Chapter 3 (Gcode3.2), of the following book:
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
    cols = ["a", "b", "c", "Pmin", "Pmax"]
    inds = [f"g{i}" for i in range(1, 6)]
    data = [
        [3.0, 20.0, 100.0, 28, 206],
        [4.05, 18.07, 98.87, 90, 284],
        [4.05, 15.55, 104.26, 68, 189],
        [3.99, 19.21, 107.21, 76, 266],
        [3.88, 26.18, 95.31, 19, 53],
    ]
    data_recs = reformat_df(pd.DataFrame(data, columns=cols, index=inds))

    return data_recs


def main():
    m = Container()

    # SETS #
    gen = Set(m, name="gen", records=[f"g{i}" for i in range(1, 6)])
    counter = Set(m, name="counter", records=[f"c{i}" for i in range(1, 12)])

    # PARAMETERS #
    report = Parameter(m, name="report", domain=[counter, "*"])
    repGen = Parameter(m, name="repGen", domain=[counter, gen])
    load = Parameter(m, name="load", records=400)
    data = Parameter(m, name="data", domain=[gen, "*"], records=data_records())

    # VARIABLES #
    P = Variable(m, name="P", domain=[gen])
    OF = Variable(m, name="OF")

    # EQUATIONS #
    eq1 = Equation(m, name="eq1", type="regular")
    eq2 = Equation(m, name="eq2", type="regular")

    eq1.expr = OF == Sum(
        gen,
        data[gen, "a"] * P[gen] * P[gen]
        + data[gen, "b"] * P[gen]
        + data[gen, "c"],
    )

    eq2.expr = Sum(gen, P[gen]) >= load

    P.lo[gen] = data[gen, "Pmin"]
    P.up[gen] = data[gen, "Pmax"]

    ECD = Model(
        m,
        name="ECD",
        equations=[eq1, eq2],
        problem="qcp",
        sense="min",
        objective=OF,
    )

    for idx, cc in enumerate(counter.toList()):
        load.assign = Sum(gen, data[gen, "Pmin"]) + (
            (idx) / (Card(counter) - 1)
        ) * Sum(gen, data[gen, "Pmax"] - data[gen, "Pmin"])
        ECD.solve()
        repGen[cc, gen] = P.l[gen]
        report[cc, "OF"] = OF.l
        report[cc, "load"] = load

    print("repgen:  \n", repGen.pivot().round(3))
    print("report:  \n", report.pivot().round(3))


if __name__ == "__main__":
    main()
