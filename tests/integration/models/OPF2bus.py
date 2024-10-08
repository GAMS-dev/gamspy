"""
## GAMSSOURCE: https://www.gams.com/latest/psoptlib_ml/libhtml/psoptlib_OPF2bus.html
## LICENSETYPE: Demo
## MODELTYPE: QCP


Optimal power flow for a simple two-bus system

For more details please refer to Chapter 6 (Gcode6.1), of the following book:
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

from __future__ import annotations

import pandas as pd

from gamspy import Container, Equation, Model, Parameter, Set, Sum, Variable


def reformat_df(dataframe):
    return dataframe.reset_index().melt(
        id_vars="index", var_name="Category", value_name="Value"
    )


def data_records():
    # data records table
    cols = ["a", "b", "c", "Pmin", "Pmax"]
    inds = ["G1", "G2"]
    data = [
        [3.0, 20.0, 100.0, 28, 206],
        [4.05, 18.07, 98.87, 90, 284],
    ]
    data_recs = reformat_df(pd.DataFrame(data, columns=cols, index=inds))
    return data_recs


def main():
    m = Container()

    # SETS #
    gen = Set(m, name="gen", records=["g1", "g2"])
    bus = Set(m, name="bus", records=["1", "2"])

    # SCALARS #
    L2 = Parameter(m, name="L2", records=400)
    X12 = Parameter(m, name="X12", records=0.2)
    Sbase = Parameter(m, name="Sbase", records=100)
    P12_max = Parameter(m, name="P12_max", records=1.5)

    # DATA PARAMETER #
    data = Parameter(m, name="data", domain=[gen, "*"], records=data_records())

    # VARIABLES #
    P = Variable(m, name="P", domain=gen)
    delta = Variable(m, name="delta", domain=bus)
    P12 = Variable(m, name="P12")

    # EQUATIONS #
    eq2 = Equation(m, name="eq2", type="regular")
    eq3 = Equation(m, name="eq3", type="regular")
    eq4 = Equation(m, name="eq4", type="regular")

    eq1 = Sum(
        gen,
        data[gen, "a"] * P[gen] * P[gen]
        + data[gen, "b"] * P[gen]
        + data[gen, "c"],
    )
    eq2[...] = P["g1"] == P12
    eq3[...] = P["g2"] + P12 == L2 / Sbase
    eq4[...] = (delta["1"] - delta["2"]) / X12 == P12

    P.lo[gen] = data[gen, "Pmin"] / Sbase
    P.up[gen] = data[gen, "Pmax"] / Sbase
    P12.lo = -P12_max
    P12.up = P12_max
    delta.fx["1"] = 0

    OPF = Model(
        m,
        name="OPF",
        equations=m.getEquations(),
        problem="qcp",
        sense="min",
        objective=eq1,
    )
    OPF.solve()

    import math

    assert math.isclose(OPF.objective_value, 306.1075, rel_tol=0.001)

    print("Objective Function Value:  ", OPF.objective_value)


if __name__ == "__main__":
    main()
