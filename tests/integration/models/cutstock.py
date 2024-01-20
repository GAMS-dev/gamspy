"""
Cutting Stock - A Column Generation Approach (CUTSTOCK)

The task is to cut out some paper products of different sizes from a
large raw paper roll, in order to meet a customer's order. The objective
is to minimize the required number of paper rolls.


P. C. Gilmore and R. E. Gomory, A linear programming approach to the
cutting stock problem, Part I, Operations Research 9 (1961), 849-859.

P. C. Gilmore and R. E. Gomory, A linear programming approach to the
cutting stock problem, Part II, Operations Research 11 (1963), 863-888.

Keywords: mixed integer linear programming, cutting stock, column generation,
          paper industry
"""

from __future__ import annotations

import os

import gamspy.math as gams_math
from gamspy import Card
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Options
from gamspy import Ord
from gamspy import Parameter
from gamspy import Sense
from gamspy import Set
from gamspy import Sum
from gamspy import Variable


def main():
    m = Container(
        system_directory=os.getenv("SYSTEM_DIRECTORY", None),
        delayed_execution=int(os.getenv("DELAYED_EXECUTION", False)),
    )

    # Sets
    i = Set(
        m,
        "i",
        records=[f"w{idx}" for idx in range(1, 5)],
        description="widths",
    )
    p = Set(
        m,
        "p",
        records=[f"p{idx}" for idx in range(1, 1001)],
        description="possible patterns",
    )
    pp = Set(m, "pp", domain=p, description="dynamic subset of p")

    # Parameters
    r = Parameter(m, "r", records=100, description="raw width")
    w = Parameter(
        m,
        "w",
        domain=i,
        records=[["w1", 45], ["w2", 36], ["w3", 31], ["w4", 14]],
        description="width",
    )
    d = Parameter(
        m,
        "d",
        domain=i,
        records=[["w1", 97], ["w2", 610], ["w3", 395], ["w4", 211]],
        description="demand",
    )
    aip = Parameter(
        m,
        "aip",
        domain=[i, p],
        description="number of width i in pattern growing in p",
    )

    # Master model variables
    xp = Variable(
        m, "xp", domain=p, type="integer", description="patterns used"
    )
    z = Variable(m, "z", description="objective variable")
    xp.up[p] = Sum(i, d[i])

    # Master model equations
    numpat = Equation(
        m,
        "numpat",
        definition=z == Sum(pp, xp[pp]),
        description="number of patterns used",
    )
    demand = Equation(m, "demand", domain=i, description="meet demand")
    demand[i] = Sum(pp, aip[i, pp] * xp[pp]) >= d[i]

    master = Model(
        m,
        "master",
        equations=[numpat, demand],
        problem="rmip",
        sense=Sense.MIN,
        objective=z,
    )

    # Pricing model variables
    y = Variable(m, "y", domain=i, type="integer", description="new pattern")
    y.up[i] = gams_math.ceil(r / w[i])

    defobj = Equation(
        m, "defobj", definition=z == (1 - Sum(i, demand.m[i] * y[i]))
    )
    knapsack = Equation(
        m,
        "knapsack",
        description="knapsack constraint",
        definition=Sum(i, w[i] * y[i]) <= r,
    )

    pricing = Model(
        m,
        "pricing",
        equations=[defobj, knapsack],
        problem="mip",
        sense=Sense.MIN,
        objective=z,
    )

    pp[p] = Ord(p) <= Card(i)
    aip[i, pp[p]].where[Ord(i) == Ord(p)] = gams_math.floor(r / w[i])

    pi = Set(m, "pi", domain=p, description="set of the last pattern")
    pi[p] = Ord(p) == Card(pp) + 1

    while len(pp) < len(p):
        master.solve(options=Options(relative_optimality_gap=0))
        pricing.solve(options=Options(relative_optimality_gap=0))

        if z.records["level"].values[0] >= -0.001:
            break

        aip[i, pi] = gams_math.Round(y.l[i])
        pp[pi] = True
        pi[p] = pi[p.lag(1)]

    master.problem = "mip"
    master.solve(options=Options(relative_optimality_gap=0))

    import math

    assert math.isclose(master.objective_value, 453.0000, rel_tol=0.001)

    patrep = Parameter(
        m, "patrep", domain=["*", "*"], description="solution pattern report"
    )
    demrep = Parameter(
        m,
        "demrep",
        domain=["*", "*"],
        description="solution demand supply report",
    )

    patrep["# produced", p] = gams_math.Round(xp.l[p])
    patrep[i, p].where[patrep["# produced", p]] = aip[i, p]
    patrep[i, "total"] = Sum(p, patrep[i, p])
    patrep["# produced", "total"] = Sum(p, patrep["# produced", p])

    demrep[i, "produced"] = Sum(p, patrep[i, p] * patrep["# produced", p])
    demrep[i, "demand"] = d[i]
    demrep[i, "over"] = demrep[i, "produced"] - demrep[i, "demand"]

    print(patrep.records)
    print(demrep.records)


if __name__ == "__main__":
    main()
