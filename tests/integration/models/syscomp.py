"""
Solving complex linear algebraic systems of equations.

References
Kalvelagen, E., (2002) Solving systems of linear equations with GAMS.
http://www.gams.com/~erwin/lineq.pdf
"""
import numpy as np
import pandas as pd

from gamspy import Alias
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Parameter
from gamspy import Set
from gamspy import Sum
from gamspy import Variable


def data_records():
    cols = ["i1", "i2", "rhs"]
    idxs = [("real", "i1"), ("real", "i2"), ("imag", "i1"), ("imag", "i2")]

    data = np.array([[30, 20, 14], [15, 8, 11], [10, -15, 5], [0, -4, -7]])

    idxs = pd.MultiIndex.from_tuples(idxs, names=["Index1", "Index2"])
    data = pd.DataFrame(data, columns=cols, index=idxs)
    data.reset_index(inplace=True)
    melted_data = data.melt(
        id_vars=["Index1", "Index2"], value_vars=["i1", "i2", "rhs"]
    )
    return melted_data


def main():
    m = Container()

    # SET #
    i = Set(m, name="i", records=["i1", "i2"])

    # ALIAS #
    j = Alias(m, name="j", alias_with=i)

    # PARAMETER #
    data = Parameter(
        m, name="data", domain=["*", "*", "*"], records=data_records()
    )

    # VARIABLES #
    rx = Variable(
        m, name="rx", domain=[i], description="real part of the solution"
    )
    ix = Variable(
        m, name="ix", domain=[i], description="imaginary part of the solution"
    )
    obj = Variable(
        m, name="obj", description="variable of a virtual objective"
    )

    # EQUATIONS #
    real = Equation(
        m,
        name="real",
        type="regular",
        domain=[i],
        description="real part of the system",
    )
    imag = Equation(
        m,
        name="imag",
        type="regular",
        domain=[i],
        description="imaginary part of the system",
    )
    eobj = Equation(
        m, name="eobj", type="regular", description="name of the objective"
    )

    eobj.expr = obj == 0
    real[i] = (
        Sum(j, data["real", i, j] * rx[j] - data["imag", i, j] * ix[j])
        == data["real", i, "rhs"]
    )
    imag[i] = (
        Sum(j, data["imag", i, j] * rx[j] + data["real", i, j] * ix[j])
        == data["imag", i, "rhs"]
    )

    syscomp = Model(
        m,
        name="syscomp",
        equations=m.getEquations(),
        problem="lp",
        sense="MIN",
        objective=obj,
    )
    syscomp.solve()

    # REPORTING PARAMETER
    rep = Parameter(m, name="rep", domain=["*", i])

    rep["rx", i] = rx.l[i]
    rep["ix", i] = ix.l[i]

    print("Objective Function Value:  ", round(obj.toValue(), 4), "\n")
    print("Solution Summary:\n", rep.pivot().round(3))

    # End of SysComp


if __name__ == "__main__":
    main()
