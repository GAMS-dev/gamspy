"""
## GAMSSOURCE: https://www.gams.com/latest/psoptlib_ml/libhtml/psoptlib_TransportationOn-Off.html
## LICENSETYPE: Demo
## MODELTYPE: MINLP


Transportation model with On/off state modeling of production side

For more details please refer to Chapter 2 (Gcode2.12), of the following book:
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
    # data records table
    cols = ["Pmin", "Pmax"]
    inds = [f"s{s}" for s in range(1, 4)]
    data = [
        [100, 450],
        [50, 350],
        [30, 500],
    ]
    data_recs = reformat_df(pd.DataFrame(data, columns=cols, index=inds))

    # c records list
    c_recs = np.array([
        [0.0755, 0.0655, 0.0498, 0.0585],
        [0.0276, 0.0163, 0.096, 0.0224],
        [0.068, 0.0119, 0.034, 0.0751],
    ])

    return c_recs, data_recs


def main():
    m = Container(
        system_directory=os.getenv("SYSTEM_DIRECTORY", None),
    )

    # SETS #
    i = Set(m, name="i", records=[f"s{s}" for s in range(1, 4)])
    j = Set(m, name="j", records=[f"d{d}" for d in range(1, 5)])

    # PARAMETERS #
    demand = Parameter(
        m, name="demand", domain=j, records=np.array([217, 150, 145, 244])
    )
    c = Parameter(m, name="c", domain=[i, j], records=data_records()[0])
    data = Parameter(
        m, name="data", domain=[i, "*"], records=data_records()[1]
    )

    # VARIABLES #
    x = Variable(m, name="x", type="free", domain=[i, j])
    P = Variable(m, name="P", type="free", domain=i)
    U = Variable(m, name="U", type="binary", domain=i)

    # EQUATIONS #

    # Objective Function
    eq1 = Sum([i, j], c[i, j] * sqr(x[i, j]))

    # Constraints
    eq2 = Equation(m, name="eq2", type="regular", domain=i)
    eq3 = Equation(m, name="eq3", type="regular", domain=i)
    eq4 = Equation(m, name="eq4", type="regular", domain=j)
    eq5 = Equation(m, name="eq5", type="regular", domain=i)

    eq2[i] = P[i] <= data[i, "Pmax"] * U[i]
    eq3[i] = P[i] >= data[i, "Pmin"] * U[i]
    eq4[j] = Sum(i, x[i, j]) >= demand[j]
    eq5[i] = Sum(j, x[i, j]) == P[i]

    P.lo[i] = 0
    P.up[i] = data[i, "Pmax"]
    x.lo[i, j] = 0
    x.up[i, j] = 100

    minlp1 = Model(
        m,
        name="minlp1",
        equations=m.getEquations(),
        problem="minlp",
        sense="min",
        objective=eq1,
    )
    minlp1.solve()

    print("Objective Function Value:  ", round(minlp1.objective_value, 4))


if __name__ == "__main__":
    main()
