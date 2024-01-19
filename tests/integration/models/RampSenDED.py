"""
Ramp rate sensitivity analysis for Dynamic Economic Load Dispatch

For more details please refer to Chapter 4 (Gcode4.2), of the following book:
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

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Parameter
from gamspy import Set
from gamspy import Sum
from gamspy import Variable
from gamspy.math import sqr


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
    gendata_recs = reformat_df(pd.DataFrame(data, columns=cols, index=inds))

    # demand records list
    demands_recs = np.array([
        510,
        530,
        516,
        510,
        515,
        544,
        646,
        686,
        741,
        734,
        748,
        760,
        754,
        700,
        686,
        720,
        714,
        761,
        727,
        714,
        618,
        584,
        578,
        544,
    ])

    return gendata_recs, demands_recs


def main():
    m = Container(
        system_directory=os.getenv("SYSTEM_DIRECTORY", None),
        delayed_execution=int(os.getenv("DELAYED_EXECUTION", False)),
    )

    # SETS #
    t = Set(
        m,
        name="t",
        records=[f"t{t}" for t in range(1, 25)],
        description="hours",
    )
    i = Set(
        m,
        name="i",
        records=[f"p{i}" for i in range(1, 5)],
        description="thermal units",
    )

    # PARAMETERS #
    gendata = Parameter(
        m,
        name="gendata",
        domain=[i, "*"],
        records=data_records()[0],
        description="generator cost characteristics and limits",
    )
    demand = Parameter(m, name="demand", domain=t, records=data_records()[1])

    # VARIABLES #
    costThermal = Variable(
        m, name="costThermal", description="cost of thermal units"
    )
    p = Variable(
        m,
        name="p",
        domain=[i, t],
        description="power generated by thermal power plant",
    )
    EM = Variable(m, name="EM", description="emission calculation")

    p.up[i, t] = gendata[i, "Pmax"]
    p.lo[i, t] = gendata[i, "Pmin"]

    # EQUATIONS #
    Genconst3 = Equation(m, name="Genconst3", type="regular", domain=[i, t])
    Genconst4 = Equation(m, name="Genconst4", type="regular", domain=[i, t])
    costThermalcalc = Equation(m, name="costThermalcalc", type="regular")
    balance = Equation(m, name="balance", type="regular", domain=t)
    EMcalc = Equation(m, name="EMcalc", type="regular")

    costThermalcalc[...] = costThermal == Sum(
        [t, i],
        gendata[i, "a"] * sqr(p[i, t])
        + gendata[i, "b"] * p[i, t]
        + gendata[i, "c"],
    )
    Genconst3[i, t] = p[i, t.lead(1)] - p[i, t] <= gendata[i, "RU"]
    Genconst4[i, t] = p[i, t.lag(1)] - p[i, t] <= gendata[i, "RD"]
    balance[t] = Sum(i, p[i, t]) >= demand[t]
    EMcalc[...] = EM == Sum(
        [t, i],
        gendata[i, "d"] * sqr(p[i, t])
        + gendata[i, "e"] * p[i, t]
        + gendata[i, "f"],
    )

    DEDcostbased = Model(
        m,
        name="DEDcostbased",
        equations=m.getEquations(),
        problem="qcp",
        sense="min",
        objective=costThermal,
    )

    # COUNTER SET
    counter = Set(m, name="counter", records=[f"c{c}" for c in range(1, 22)])

    # SCALAR
    Rscale = Parameter(m, name="Rscale", records=1)

    # REPORTING PARAMETER
    report1 = Parameter(m, name="report1", domain=[counter, "*"])

    for idx, c in enumerate(counter.toList()):
        Rscale[...] = 1 - (idx) * 0.02
        gendata[i, "RU"] = gendata[i, "RU0"] * Rscale
        gendata[i, "RD"] = gendata[i, "RD0"] * Rscale
        DEDcostbased.solve()
        report1[c, "Scale"] = Rscale
        report1[c, "TC"] = costThermal.l
        report1[c, "EM"] = EM.l

    print("report1:  \n", report1.pivot().round(4))

    report1.pivot().round(4).to_excel(
        "DEDcostbased.xlsx", sheet_name="Pthermal"
    )


if __name__ == "__main__":
    main()
