"""
Aircraft Allocation under uncertain Demand (AIRCRAF)

The objective of this model is to allocate aircrafts to routes to maximize
the expected profit when traffic demand is uncertain. Two different
formulations are used, the delta and the lambda formulation.


Dantzig, G B, Chapter 28. In Linear Programming and Extensions.
Princeton University Press, Princeton, New Jersey, 1963.

Keywords: linear programming, aircraft managing, allocation problem
"""

from gamspy import (
    Set,
    Alias,
    Parameter,
    Variable,
    Equation,
    Container,
    Sum,
    Ord,
    Model,
    Sense,
)
import numpy as np


def main():
    m = Container()

    # Sets
    i = Set(m, name="i", records=["a", "b", "c", "d"])
    j = Set(m, name="j", records=[f"route-{i}" for i in range(1, 6)])
    h = Set(m, name="h", records=list(range(1, 6)))

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
    )

    aa = Parameter(
        m, name="aa", domain=[i], records=np.array([10, 19, 25, 15])
    )
    k = Parameter(m, name="k", domain=[j], records=np.array([13, 13, 7, 7, 1]))
    ed = Parameter(m, name="ed", domain=[j])
    gamma = Parameter(m, name="gamma", domain=[j, h])
    deltb = Parameter(m, name="deltb", domain=[j, h])

    ed[j].assign = Sum(h, lamda[j, h] * dd[j, h])
    gamma[j, h] = Sum(hp.where[(Ord(hp) >= Ord(h))], lamda[j, hp])
    deltb[j, h] = dd[j, h] - dd[j, h.lag(1, "linear")].where[dd[j, h]]

    # Variables
    x = Variable(m, name="x", domain=[i, j], type="positive")
    y = Variable(m, name="y", domain=[j, h], type="positive")
    b = Variable(m, name="b", domain=[j, h], type="positive")
    oc = Variable(m, name="oc", type="positive")
    bc = Variable(m, name="bc", type="positive")
    phi = Variable(m, name="phi")

    # Equations
    ab = Equation(m, name="ab", domain=[i])
    db = Equation(m, name="db", domain=[j])
    yd = Equation(m, name="yd", domain=[j, h])
    bd = Equation(m, name="bd", domain=[j, h])
    ocd = Equation(m, name="ocd")
    bcd1 = Equation(m, name="bcd1")
    bcd2 = Equation(m, name="bcd2")
    obj = Equation(m, name="obj")

    ab[i] = Sum(j, x[i, j]) <= aa[i]
    db[j] = Sum(i, p[i, j] * x[i, j]) >= Sum(h.where[deltb[j, h]], y[j, h])
    yd[j, h] = y[j, h] <= Sum(i, p[i, j] * x[i, j])
    bd[j, h] = b[j, h] == dd[j, h] - y[j, h]
    ocd.expr = oc == Sum([i, j], c[i, j] * x[i, j])
    bcd1.expr = bc == Sum(j, k[j] * (ed[j] - Sum(h, gamma[j, h] * y[j, h])))
    bcd2.expr = bc == Sum([j, h], k[j] * lamda[j, h] * b[j, h])
    obj.expr = phi == oc + bc

    alloc1 = Model(
        m,
        name="alloc1",
        equations=[ab, db, ocd, bcd1, obj],
        problem="LP",
        sense=Sense.MIN,
        objective=phi,
    )
    alloc2 = Model(
        m,
        name="alloc2",
        equations=[ab, yd, bd, ocd, bcd2, obj],
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


if __name__ == "__main__":
    main()
