from __future__ import annotations

import math
import os
from pathlib import Path

from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Number
from gamspy import Ord
from gamspy import Parameter
from gamspy import Problem
from gamspy import Smax
from gamspy import Sum
from gamspy import Variable
from gamspy import VariableType


def main():
    m = Container(
        system_directory=os.getenv("SYSTEM_DIRECTORY", None),
        delayed_execution=int(os.getenv("DELAYED_EXECUTION", False)),
        load_from=str(Path(__file__).parent.absolute()) + "/hansmcp.gdx",
    )

    c = m["c"]
    h = m["h"]
    s = m["s"]
    cc = m["cc"]

    e = m["e"]
    d = m["d"]
    esub = m["esub"]
    data = m["data"]

    alpha = Parameter(m, "alpha", domain=[c, h])
    a = Parameter(m, "a", domain=[c, s])

    alpha[c, h] = d[c, h] / Sum(cc, d[cc, h])
    a[c, s] = data["output", c, s] - data["input", c, s]

    p = Variable(m, "p", type=VariableType.POSITIVE, domain=c)
    y = Variable(m, "y", type=VariableType.POSITIVE, domain=s)
    i = Variable(m, "i", type=VariableType.POSITIVE, domain=h)

    mkt = Equation(m, "mkt", domain=c)
    profit = Equation(m, "profit", domain=s)
    income = Equation(m, "income", domain=h)

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
