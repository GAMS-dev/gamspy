"""
## GAMSSOURCE: https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_qp6.html
## LICENSETYPE: Demo
## MODELTYPE: MCP
## DATAFILES: qp6.gdx
## KEYWORDS: mixed complementarity problem, quadratic programming, finance


Standard QP Model - QP4 expressed as MCP (QP6)

Formulate the QP as an LCP, ie write down the first order
conditions of QP4 and solve.


Kalvelagen, E, Model Building with GAMS. forthcoming
de Wetering, A V, private communication.
"""

from __future__ import annotations

from pathlib import Path

from gamspy import (
    Card,
    Container,
    Equation,
    Model,
    Number,
    Ord,
    Parameter,
    Set,
    Sum,
    Variable,
)
from gamspy.math import sqr


def main():
    cont = Container(
        load_from=str(Path(__file__).parent.absolute()) + "/qp6.gdx",
    )

    # Sets
    days, stocks = cont.getSymbols(["days", "stocks"])

    # Parameters
    returns, _val = cont.getSymbols(["return", "val"])

    # Set
    d = Set(cont, name="d", domain=[days], description="selected days")
    s = Set(cont, name="s", domain=[stocks], description="selected stocks")

    # select subset of stocks and periods
    d[days] = (Ord(days) > 1) & (Ord(days) < 31)
    s[stocks] = Ord(stocks) < 51

    # Parameter
    mean = Parameter(
        cont, name="mean", domain=stocks, description="mean of daily return"
    )
    dev = Parameter(cont, name="dev", domain=[stocks, days], description="deviations")
    totmean = Parameter(cont, name="totmean", description="total mean return")

    mean[s] = Sum(d, returns[s, d]) / Card(d)
    dev[s, d] = returns[s, d] - mean[s]
    totmean[...] = Sum(s, mean[s]) / (Card(s))

    # Variable
    x = Variable(
        cont,
        name="x",
        type="positive",
        domain=stocks,
        description="investments",
    )
    w = Variable(
        cont,
        name="w",
        type="free",
        domain=days,
        description="intermediate variables",
    )

    # Equation
    budget = Equation(cont, name="budget")
    retcon = Equation(cont, name="retcon", description="returns constraint")
    wdef = Equation(cont, name="wdef", domain=days)

    wdef[d] = w[d] == Sum(s, x[s] * dev[s, d])

    budget[...] = Sum(s, x[s]) == Number(1.0)

    retcon[...] = Sum(s, mean[s] * x[s]) >= totmean * 1.25

    # Equation
    d_x = Equation(cont, name="d_x", domain=stocks)
    d_w = Equation(cont, name="d_w", domain=days)

    # Variable
    m_budget = Variable(cont, name="m_budget", type="free")
    m_wdef = Variable(cont, name="m_wdef", type="free", domain=days)

    # Positive Variable
    m_retcon = Variable(cont, name="m_retcon", type="positive")

    m_wdef.fx[days].where[~d[days]] = 0

    d_x[s] = Sum(d, m_wdef[d] * dev[s, d]) >= m_retcon * mean[s] + m_budget

    d_w[d] = 2 * w[d] / (Card(d) - 1) == m_wdef[d]

    qp6 = Model(
        cont,
        name="qp6",
        matches={
            d_x: x,
            d_w: w,
            retcon: m_retcon,
            budget: m_budget,
            wdef: m_wdef,
        },
        problem="mcp",
    )

    qp6.solve()

    z = Parameter(cont, name="z")
    z[...] = Sum(d, sqr(w.l[d])) / (Card(d) - 1)
    print(z.records)
    print("\ninvestments: ", x.records.set_index("stocks").level.round(3))


if __name__ == "__main__":
    main()
