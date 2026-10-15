"""
## LICENSETYPE: Demo
## MODELTYPE: MIP
## KEYWORDS: mixed integer linear programming, minimum, transportation problem, economies of scale, formulations


A Transportation Problem with Chartered Trucks (TRNSCHARTER)

This problem finds a least cost shipping schedule that meets requirements
at markets and supplies at factories. It extends the trnsport model in the
GAMS Model Library: every route can be served either by a parcel carrier
that charges per case or by a chartered truck that charges a flat fee and
carries any quantity. The shipper picks the cheaper option, so the cost of
a route is

  cost(i,j) = min(c(i,j) * x(i,j), charter(i,j))

The cost is concave, i.e. it rewards consolidating shipments on fewer
routes, which makes the problem non-convex.

gamspy.formulations.minimum makes the cost equal to the minimum by
selecting one operand with a binary variable, so the shipper pays for every
route it uses.

Dantzig, G B, Chapter 3.3. In Linear Programming and Extensions.
Princeton University Press, Princeton, New Jersey, 1963.
"""

from __future__ import annotations

import numpy as np

import gamspy.formulations as formulations
from gamspy import (
    Container,
    Equation,
    Model,
    Parameter,
    Sense,
    Set,
    Sum,
    Variable,
)


def main():
    m = Container()

    # Sets
    i = Set(
        m,
        name="i",
        records=["seattle", "san-diego"],
        description="canning plants",
    )
    j = Set(
        m,
        name="j",
        records=["new-york", "chicago", "topeka"],
        description="markets",
    )

    # Parameters
    a = Parameter(
        m,
        name="a",
        domain=i,
        records=np.array([350, 600]),
        description="capacity of plant i in cases",
    )
    b = Parameter(
        m,
        name="b",
        domain=j,
        records=np.array([325, 300, 275]),
        description="demand at market j in cases",
    )
    d = Parameter(
        m,
        name="d",
        domain=[i, j],
        records=np.array([[2.5, 1.7, 1.8], [2.5, 1.8, 1.4]]),
        description="distance in thousands of miles",
    )
    f = Parameter(
        m,
        name="f",
        records=90,
        description="freight in dollars per case per thousand miles",
    )
    c = Parameter(
        m,
        name="c",
        domain=[i, j],
        description="parcel cost in thousands of dollars per case",
    )
    c[i, j] = f * d[i, j] / 1000
    charter = Parameter(
        m,
        name="charter",
        domain=[i, j],
        description="flat fee of a chartered truck in thousands of dollars",
    )
    charter[i, j] = 20 + 15 * d[i, j]

    # Variables
    x = Variable(
        m,
        name="x",
        type="positive",
        domain=[i, j],
        description="shipment quantities in cases",
    )

    # Equations
    supply = Equation(
        m, name="supply", domain=i, description="observe supply limit at plant i"
    )
    supply[i] = Sum(j, x[i, j]) <= a[i]
    demand = Equation(
        m, name="demand", domain=j, description="satisfy demand at market j"
    )
    demand[j] = Sum(i, x[i, j]) >= b[j]

    # cost[i, j] == min(c[i, j] * x[i, j], charter[i, j])
    cost, eqs = formulations.minimum(c[i, j] * x[i, j], charter[i, j])

    transport = Model(
        m,
        name="transport",
        equations=[supply, demand, *eqs],
        problem="mip",
        sense=Sense.MIN,
        objective=Sum([i, j], cost[i, j]),
    )
    transport.solve()

    print("Objective Function Value: ", round(transport.objective_value, 3))
    print(x.pivot(), cost.pivot().round(3), sep="\n")

    # The cost equals the minimum on every route
    expected = np.minimum(c.toDense() * x.toDense(), charter.toDense())
    if np.abs(cost.toDense() - expected).max() > 1e-6:
        raise Exception("cost does not equal the minimum")


if __name__ == "__main__":
    main()
