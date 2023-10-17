"""
Price based Dynamic Economic Load Dispatch

For more details please refer to Chapter 4 (Gcode4.5), of the following book:
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
import numpy as np
import pandas as pd

import gamspy.math as gams_math
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
    # gendata records table
    cols = ["a", "b", "c", "d", "e", "f", "Pmin", "Pmax", "RU0", "RD0"]
    inds = ["p1", "p2", "p3", "p4"]
    data = [
        [0.12, 14.80, 89, 1.2, -5, 3, 28, 200, 40, 40],
        [0.17, 16.57, 83, 2.3, -4.24, 6.09, 20, 290, 30, 30],
        [0.15, 15.55, 100, 1.1, -2.15, 5.69, 30, 190, 30, 30],
        [0.19, 16.21, 70, 1.1, -3.99, 6.2, 20, 260, 50, 50],
    ]
    gen_recs = reformat_df(pd.DataFrame(data, columns=cols, index=inds))

    # data records table
    cols = ["lamda", "load"]
    inds = [f"t{i}" for i in range(1, 25)]
    data = [
        [32.71, 510],
        [34.72, 530],
        [32.71, 516],
        [32.74, 510],
        [32.96, 515],
        [34.93, 544],
        [44.9, 646],
        [52.0, 686],
        [53.03, 741],
        [47.26, 734],
        [44.07, 748],
        [38.63, 760],
        [39.91, 754],
        [39.45, 700],
        [41.14, 686],
        [39.23, 720],
        [52.12, 714],
        [40.85, 761],
        [41.2, 727],
        [41.15, 714],
        [45.76, 618],
        [45.59, 584],
        [45.56, 578],
        [34.72, 544],
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
    i = Set(
        m,
        name="i",
        records=[f"p{i}" for i in range(1, 5)],
        description="thermal units",
    )

    # SCALAR #
    lim = Parameter(m, name="lim", records=np.inf)

    # PARAMETERS #
    gendata = Parameter(
        m,
        name="gendata",
        domain=[i, "*"],
        records=data_records()[0],
        description="generator cost characteristics and limits",
    )
    data = Parameter(
        m, name="data", domain=[t, "*"], records=data_records()[1]
    )

    # VARIABLES #
    OF = Variable(m, name="OF", type="free", description="objective (revenue)")
    costThermal = Variable(
        m, name="costThermal", type="free", description="cost of thermal units"
    )
    p = Variable(
        m,
        name="p",
        type="free",
        domain=[i, t],
        description="power generated by thermal power plant",
    )
    EM = Variable(
        m, name="EM", type="free", description="emission calculation"
    )

    p.up[i, t] = gendata[i, "Pmax"]
    p.lo[i, t] = gendata[i, "Pmin"]

    # EQUATIONS #
    Genconst3 = Equation(m, name="Genconst3", type="regular", domain=[i, t])
    Genconst4 = Equation(m, name="Genconst4", type="regular", domain=[i, t])
    costThermalcalc = Equation(m, name="costThermalcalc", type="regular")
    balance = Equation(m, name="balance", type="regular", domain=[t])
    EMcalc = Equation(m, name="EMcalc", type="regular")
    EMlim = Equation(m, name="EMlim", type="regular")
    benefitcalc = Equation(m, name="benefitcalc", type="regular")

    costThermalcalc.definition = costThermal == Sum(
        [t, i],
        gendata[i, "a"] * gams_math.power(p[i, t], 2)
        + gendata[i, "b"] * p[i, t]
        + gendata[i, "c"],
    )

    Genconst3[i, t] = p[i, t.lead(1)] - p[i, t] <= gendata[i, "RU0"]

    Genconst4[i, t] = p[i, t.lag(1)] - p[i, t] <= gendata[i, "RD0"]

    balance[t] = Sum(i, p[i, t]) <= data[t, "load"]

    EMcalc.definition = EM == Sum(
        [t, i],
        gendata[i, "d"] * gams_math.power(p[i, t], 2)
        + gendata[i, "e"] * p[i, t]
        + gendata[i, "f"],
    )

    EMlim.definition = EM <= lim

    benefitcalc.definition = (
        OF == Sum([i, t], 1 * data[t, "lamda"] * p[i, t]) - costThermal
    )

    DEDPB = Model(
        m,
        name="DEDPB",
        equations=m.getEquations(),
        problem="qcp",
        sense="max",
        objective=OF,
    )
    DEDPB.solve()

    import math

    assert math.isclose(DEDPB.objective_value, 99552.6661, rel_tol=0.001)

    # Export results to an excel file
    p.pivot().round(3).to_excel("DEDPB.xlsx")


if __name__ == "__main__":
    main()
