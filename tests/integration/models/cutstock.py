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

from gamspy import Set, Parameter, Variable, Equation, Model, Container
from gamspy import Ord, Card, Sum, Number, Sense
import gamspy.math as math


def main():
    m = Container()

    # Sets
    i = Set(m, "i", records=[f"w{idx}" for idx in range(1, 5)])
    p = Set(m, "p", records=[f"p{idx}" for idx in range(1, 1001)])
    pp = Set(m, "pp", domain=[p])

    # Parameters
    r = Parameter(m, "r", records=100)
    w = Parameter(
        m,
        "w",
        domain=[i],
        records=[["w1", 45], ["w2", 36], ["w3", 31], ["w4", 14]],
    )
    d = Parameter(
        m,
        "d",
        domain=[i],
        records=[["w1", 97], ["w2", 610], ["w3", 395], ["w4", 211]],
    )
    aip = Parameter(m, "aip", domain=[i, p])

    # Master model variables
    xp = Variable(m, "xp", domain=[p], type="integer")
    z = Variable(m, "z")
    xp.up[p] = Sum(i, d[i])

    # Master model equations
    numpat = Equation(m, "numpat", expr=z == Sum(pp, xp[pp]))
    demand = Equation(m, "demand", domain=[i])
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
    y = Variable(m, "y", domain=[i], type="integer")
    y.up[i] = math.ceil(r / w[i])

    defobj = Equation(
        m,
        "defobj",
        expr=z == (1 - Sum(i, demand.m[i] * y[i])),
    )
    knapsack = Equation(m, "knapsack", expr=Sum(i, w[i] * y[i]) <= r)

    pricing = Model(
        m,
        "pricing",
        equations=[defobj, knapsack],
        problem="mip",
        sense=Sense.MIN,
        objective=z,
    )

    pp[p] = Ord(p) <= Card(i)
    aip[i, pp[p]].where[Ord(i) == Ord(p)] = math.floor(r / w[i])

    pi = Set(m, "pi", domain=[p])
    pi[p] = Ord(p) == Card(pp) + 1

    m.addOptions({"optCr": 0, "limRow": 0, "limCol": 0, "solPrint": "off"})

    while len(pp) < len(p):
        master.solve()
        pricing.solve()

        if z.records["level"].values[0] >= -0.001:
            break

        aip[i, pi] = math.Round(y.l[i])
        pp[pi] = Number(1)
        pi[p] = pi[p.lag(1)]

    master.problem = "mip"
    master.solve()

    patrep = Parameter(m, "patrep", domain=["*", "*"])
    demrep = Parameter(m, "demrep", domain=["*", "*"])

    patrep["# produced", p] = math.Round(xp.l[p])
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
