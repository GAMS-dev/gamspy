"""
A small linear dynamic macroeconomic model of U.S. economy in which
both monetary and fiscal policy variables are used.

Linear Quadratic Riccati Equations are solved as a General Nonlinear
Programming Problem instead of the usual Matrix Recursion.

Please see:
Kendrick, D, Caution and Probing in a Macroeconomic Model.
Journal of Economic Dynamics and Control 4, 2 (1982) pp.149-170.
"""
import numpy as np

import gamspy.math as gams_math
from gamspy import Alias
from gamspy import Card
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Number
from gamspy import Ord
from gamspy import Parameter
from gamspy import Set
from gamspy import Sum
from gamspy import Variable


def main():
    cont = Container()

    # SETS #
    n = Set(
        cont, name="n", records=["consumpt", "invest"], description="states"
    )
    m = Set(
        cont, name="m", records=["gov-expend", "money"], description="controls"
    )
    k = Set(
        cont,
        name="k",
        records=[
            "1964-i",
            "1964-ii",
            "1964-iii",
            "1964-iv",
            "1965-i",
            "1965-ii",
            "1965-iii",
            "1965-iv",
        ],
        description="horizon",
    )
    ku = Set(cont, name="ku", domain=[k], description="control horizon")
    ki = Set(cont, name="ki", domain=[k], description="initial period")
    kt = Set(cont, name="kt", domain=[k], description="terminal period")

    # ALIASES #
    nn = Alias(cont, name="nn", alias_with=n)
    mp = Alias(cont, name="mp", alias_with=m)

    ku[k] = Number(1).where[Ord(k) < Card(k)]
    ki[k] = Number(1).where[Ord(k) == 1]
    kt[k] = ~ku[k]

    # PARAMETERS #
    a = Parameter(
        cont,
        name="a",
        domain=[n, nn],
        records=np.array([[0.914, -0.016], [0.097, 0.424]]),
        description="state vector matrix",
    )
    b = Parameter(
        cont,
        name="b",
        domain=[n, m],
        records=np.array([[0.305, 0.424], [-0.101, 1.459]]),
        description="control vector matrix",
    )
    wk = Parameter(
        cont,
        name="wk",
        domain=[n, nn],
        records=np.array([[0.0625, 0], [0, 1]]),
        description="penalty matrix for states - input",
    )
    rk = Parameter(
        cont,
        name="rk",
        domain=[m, mp],
        records=np.array([[1, 0], [0, 0.444]]),
        description="penalty matrix for controls",
    )

    c = Parameter(
        cont,
        name="c",
        domain=[n],
        records=[("consumpt", -59.4), ("invest", -184.7)],
        description="constant term",
    )
    xinit = Parameter(
        cont,
        name="xinit",
        domain=[n],
        records=[("consumpt", 387.9), ("invest", 85.3)],
        description="initial value",
    )
    uinit = Parameter(
        cont,
        name="uinit",
        domain=[m],
        records=[("gov-expend", 110.5), ("money", 147.1)],
        description="initial controls",
    )
    xtilde = Parameter(
        cont, name="xtilde", domain=[n, k], description="desired path for x"
    )
    utilde = Parameter(
        cont, name="utilde", domain=[m, k], description="desired path for u"
    )
    w = Parameter(
        cont,
        name="w",
        domain=[n, nn, k],
        description="penalty matrix on states",
    )

    w[n, nn, ku] = wk[n, nn]
    w[n, nn, kt] = 10000 * wk[n, nn]
    xtilde[n, k] = xinit[n] * gams_math.power(1.0075, Ord(k) - 1)
    utilde[m, k] = uinit[m] * gams_math.power(1.0075, Ord(k) - 1)

    # VARIABLES #
    x = Variable(cont, name="x", domain=[n, k], description="state variable")
    u = Variable(cont, name="u", domain=[m, k], description="control variable")
    j = Variable(cont, name="j", description="criterion")

    # EQUATIONS #

    criterion = Equation(
        cont,
        name="criterion",
        type="regular",
        description="criterion definition",
    )
    stateq = Equation(
        cont,
        name="stateq",
        type="regular",
        domain=[n, k],
        description="state equation",
    )

    criterion.expr = j == 0.5 * Sum(
        [k, n, nn],
        (x[n, k] - xtilde[n, k]) * w[n, nn, k] * (x[nn, k] - xtilde[nn, k]),
    ) + 0.5 * Sum(
        [ku, m, mp],
        (u[m, ku] - utilde[m, ku]) * rk[m, mp] * (u[mp, ku] - utilde[mp, ku]),
    )

    stateq[n, k.lead(1)] = (
        x[n, k.lead(1)]
        == Sum(nn, a[n, nn] * x[nn, k]) + Sum(m, b[n, m] * u[m, k]) + c[n]
    )

    macro = Model(
        cont,
        name="macro",
        equations=cont.getEquations(),
        problem="nlp",
        sense="MIN",
        objective=j,
    )

    x.l[n, k] = xinit[n]
    u.l[m, k] = uinit[m]
    x.fx[n, ki] = xinit[n]

    macro.solve()

    # REPORTING PARAMETER
    rep = Parameter(cont, name="rep", domain=["*", k])

    rep["xtilde_consumpt", k] = xtilde["consumpt", k]
    rep["xtilde_invest", k] = xtilde["invest", k]
    rep["x_consumpt", k] = x.l["consumpt", k]
    rep["x_invest", k] = x.l["invest", k]
    rep["utilde_gov-expend", k] = utilde["gov-expend", k]
    rep["utilde_money", k] = utilde["money", k]
    rep["u_gov-expend", k] = u.l["gov-expend", k]
    rep["u_money", k] = u.l["money", k]

    print("Objective Function Value: ", round(j.toValue(), 4))
    print("Solution Summary:\n", rep.pivot().round(3))

    # End Macro


if __name__ == "__main__":
    main()
