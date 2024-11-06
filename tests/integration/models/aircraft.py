"""
## GAMSSOURCE: https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_aircraft.html
## LICENSETYPE: Demo
## MODELTYPE: LP
## KEYWORDS: linear programming, aircraft managing, allocation problem


Aircraft Allocation under uncertain Demand (AIRCRAF)

The objective of this model is to allocate aircrafts to routes to maximize
the expected profit when traffic demand is uncertain. Two different
formulations are used, the delta and the lambda formulation.


Dantzig, G B, Chapter 28. In Linear Programming and Extensions.
Princeton University Press, Princeton, New Jersey, 1963.
"""

from __future__ import annotations

import numpy as np

from gamspy import (
    Alias,
    Container,
    Equation,
    Model,
    Ord,
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
        records=["a", "b", "c", "d"],
        description="aircraft types and unassigned passengers",
    )
    j = Set(
        m,
        name="j",
        records=[f"route-{i}" for i in range(1, 6)],
        description="assigned and unassigned routes",
    )
    h = Set(
        m, name="h", records=list(range(1, 6)), description="demand states"
    )

    # Alias
    hp = Alias(m, name="hp", alias_with=h)

    # Parameters
    dd = Parameter(
        m,
        name="dd",
        domain=[j, h],
        records=np.array(
            [
                [200, 220, 250, 270, 300],
                [50, 150, 0, 0, 0],
                [140, 160, 180, 200, 220],
                [10, 50, 80, 100, 340],
                [580, 600, 620, 0, 0],
            ]
        ),
        description="demand distribution on route j",
    )

    lamda = Parameter(
        m,
        name="lamda",
        domain=[j, h],
        records=np.array(
            [
                [0.2, 0.05, 0.35, 0.2, 0.2],
                [0.3, 0.7, 0, 0, 0],
                [0.1, 0.2, 0.4, 0.2, 0.1],
                [0.2, 0.2, 0.3, 0.2, 0.1],
                [0.1, 0.8, 0.1, 0, 0],
            ]
        ),
        description="probability of demand state h on route j",
    )

    c = Parameter(
        m,
        name="c",
        domain=[i, j],
        records=np.array(
            [
                [18, 21, 18, 16, 10],
                [0, 15, 16, 14, 9],
                [0, 10, 0, 9, 6],
                [17, 16, 17, 15, 10],
            ]
        ),
        description="costs per aircraft (1000s)",
    )

    p = Parameter(
        m,
        name="p",
        domain=[i, j],
        records=np.array(
            [
                [16, 15, 28, 23, 81],
                [0, 10, 14, 15, 57],
                [0, 5, 0, 7, 29],
                [9, 11, 22, 17, 55],
            ]
        ),
        description="passenger capacity of aircraft i on route j",
    )

    aa = Parameter(
        m,
        name="aa",
        domain=i,
        records=np.array([10, 19, 25, 15]),
        description="aircraft availability",
    )
    k = Parameter(
        m,
        name="k",
        domain=j,
        records=np.array([13, 13, 7, 7, 1]),
        description="revenue lost (1000 per 100 bumped)",
    )
    ed = Parameter(m, name="ed", domain=j, description="expected demand")
    gamma = Parameter(
        m,
        name="gamma",
        domain=[j, h],
        description="probability of exceeding demand increment h on route j",
    )
    deltb = Parameter(
        m,
        name="deltb",
        domain=[j, h],
        description="incremental passenger load in demand states",
    )

    ed[j][...] = Sum(h, lamda[j, h] * dd[j, h])
    gamma[j, h] = Sum(hp.where[(Ord(hp) >= Ord(h))], lamda[j, hp])
    deltb[j, h] = dd[j, h] - dd[j, h - 1].where[dd[j, h]]

    # Variables
    x = Variable(
        m,
        name="x",
        domain=[i, j],
        type="positive",
        description="number of aircraft type i assigned to route j",
    )
    y = Variable(
        m,
        name="y",
        domain=[j, h],
        type="positive",
        description="passengers actually carried",
    )
    b = Variable(
        m,
        name="b",
        domain=[j, h],
        type="positive",
        description="passengers bumped",
    )
    oc = Variable(m, name="oc", type="positive", description="operating cost")
    bc = Variable(m, name="bc", type="positive", description="bumping cost")

    # Equations
    ab = Equation(m, name="ab", domain=i, description="aircraft balance")
    db = Equation(m, name="db", domain=j, description="demand balance")
    yd = Equation(
        m,
        name="yd",
        domain=[j, h],
        description="definition of boarded passengers",
    )
    bd = Equation(
        m,
        name="bd",
        domain=[j, h],
        description="definition of bumped passengers",
    )
    ocd = Equation(m, name="ocd", description="operating cost definition")
    bcd1 = Equation(
        m, name="bcd1", description="bumping cost definition: version 1"
    )
    bcd2 = Equation(
        m, name="bcd2", description="bumping cost definition: version 2"
    )

    phi = oc + bc
    ab[i] = Sum(j, x[i, j]) <= aa[i]
    db[j] = Sum(i, p[i, j] * x[i, j]) >= Sum(h.where[deltb[j, h]], y[j, h])
    yd[j, h] = y[j, h] <= Sum(i, p[i, j] * x[i, j])
    bd[j, h] = b[j, h] == dd[j, h] - y[j, h]
    ocd[...] = oc == Sum([i, j], c[i, j] * x[i, j])
    bcd1[...] = bc == Sum(j, k[j] * (ed[j] - Sum(h, gamma[j, h] * y[j, h])))
    bcd2[...] = bc == Sum([j, h], k[j] * lamda[j, h] * b[j, h])

    alloc1 = Model(
        m,
        name="alloc1",
        equations=[ab, db, ocd, bcd1],
        problem="LP",
        sense=Sense.MIN,
        objective=phi,
    )
    alloc2 = Model(
        m,
        name="alloc2",
        equations=[ab, yd, bd, ocd, bcd2],
        problem="LP",
        sense=Sense.MIN,
        objective=phi,
    )

    y.up[j, h] = deltb[j, h]
    alloc1.solve()

    print("Number of passengers carried with limit:")
    print(y.pivot())

    y.up[j, h] = np.inf

    alloc2.solve()
    print()
    print("Number of passengers carried without limit:")
    print(y.pivot())

    import math

    assert math.isclose(alloc2.objective_value, 1566.042189, rel_tol=0.001)


if __name__ == "__main__":
    main()
