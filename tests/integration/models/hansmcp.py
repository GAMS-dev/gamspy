"""
## GAMSSOURCE: https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_hansmcp.html
## LICENSETYPE: Demo
## MODELTYPE: MCP
## DATAFILES: hansmcp.gdx
## KEYWORDS: mixed complementarity problem, activity analysis, general equilibrium model, social accounting matrix, european regional policy, impact analysis


Hansen's Activity Analysis Example.


Scarf, H, and Hansen, T, The Computation of Economic Equilibria.
Yale University Press, 1973.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

from gamspy import Alias
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Number
from gamspy import Ord
from gamspy import Parameter
from gamspy import Problem
from gamspy import Set
from gamspy import Smax
from gamspy import Sum
from gamspy import Variable
from gamspy import VariableType


def main():
    m = Container(
        system_directory=os.getenv("SYSTEM_DIRECTORY", None),
        delayed_execution=int(os.getenv("DELAYED_EXECUTION", False)),
    )

    c = Set(m, "c", description="commodities")
    h = Set(m, "h", description="consumers")
    s = Set(m, "s", description="sectors")

    cc = Alias(m, "cc", c)

    e = Parameter(m, "e", domain=[c, h], description="commodity endowments")
    d = Parameter(m, "d", domain=[c, h], description="reference demands")
    esub = Parameter(m, "esub", domain=h, description="elasticities in demand")
    data = Parameter(
        m, "data", domain=["*", c, s], description="activity analysis matrix"
    )

    # Load the records of the symbols from a gdx file.
    m.loadRecordsFromGdx(
        str(Path(__file__).parent.absolute()) + "/hansmcp.gdx",
        ["c", "h", "s", "e", "d", "esub", "data"],
    )

    alpha = Parameter(
        m,
        "alpha",
        domain=[c, h],
        description="demand function share parameter",
    )
    a = Parameter(
        m, "a", domain=[c, s], description="activity analysis matrix"
    )

    alpha[c, h] = d[c, h] / Sum(cc, d[cc, h])
    a[c, s] = data["output", c, s] - data["input", c, s]

    p = Variable(
        m,
        "p",
        type=VariableType.POSITIVE,
        domain=c,
        description="commodity price",
    )
    y = Variable(
        m, "y", type=VariableType.POSITIVE, domain=s, description="production"
    )
    i = Variable(
        m, "i", type=VariableType.POSITIVE, domain=h, description="income"
    )

    mkt = Equation(m, "mkt", domain=c, description="commodity market")
    profit = Equation(m, "profit", domain=s, description="zero profit")
    income = Equation(m, "income", domain=h, description="income index")

    mkt[c] = Sum(s, a[c, s] * y[s]) + Sum(h, e[c, h]) >= Sum(
        h.where[esub[h] != 1],
        (i[h] / Sum(cc, alpha[cc, h] * p[cc] ** (1 - esub[h])))
        * alpha[c, h]
        * (1 / p[c]) ** esub[h],
    ) + Sum(h.where[esub[h] == 1], i[h] * alpha[c, h] / p[c])

    profit[s] = -Sum(c, a[c, s] * p[c]) >= 0
    income[h] = i[h] >= Sum(c, p[c] * e[c, h])

    hansen = Model(
        m,
        "hansen",
        problem=Problem.MCP,
        matches={mkt: p, profit: y, income: i},
    )

    p.l[c] = 1
    y.l[s] = 1
    i.l[h] = 1
    p.lo[c] = Number(0.00001).where[Smax(h, alpha[c, h]) > 0]

    p.fx[c].where[Ord(c) == 1] = 1

    hansen.solve()

    # check correctness
    correct_result = [
        5.1549387635430755,
        2.827534834524584,
        0.5875814316920335,
        8.5599675080206,
    ]

    for expected, found in zip(correct_result, i.records.level.to_list()):
        assert math.isclose(expected, found)


if __name__ == "__main__":
    main()
