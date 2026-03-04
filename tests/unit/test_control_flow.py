import numpy as np
import pytest

import gamspy as gp
from gamspy.exceptions import ValidationError

pytestmark = pytest.mark.unit


def test_loop():
    m = gp.Container()

    t = gp.Set(m, records=["1985", "1986", "1987", "1988", "1989", "1990"])
    pop = gp.Parameter(m, domain=t, records=[("1985", "3456")])
    growth = gp.Parameter(
        m, domain=t, records=np.array([25.3, 27.3, 26.2, 27.1, 26.6, 26.6])
    )

    with gp.Loop(t):
        pop[t + 1] = pop[t] + growth[t]

    assert pop.toList() == [
        ("1985", 3456.0),
        ("1986", 3481.3),
        ("1987", 3508.6000000000004),
        ("1988", 3534.8),
        ("1989", 3561.9),
        ("1990", 3588.5),
    ]

    m = gp.Container()
    i = gp.Set(m, records=[f"i{idx}" for idx in range(1, 4)])
    j = gp.Set(m, records=[f"j{idx}" for idx in range(1, 6)])
    k = gp.Set(
        m,
        domain=[i, j],
        records=[("i1", "j1"), ("i1", "j3"), ("i3", "j3"), ("i3", "j5")],
    )

    c = gp.Parameter(m, domain=i, records=[("i1", 3), ("i2", 1)])
    q = gp.Parameter(
        m, domain=[i, j], records=[("i1", "j1", 1), ("i1", "j2", 3), ("i1", "j4", 2)]
    )
    x = gp.Parameter(m, records=1)
    y = gp.Parameter(m, records=3)
    z = gp.Parameter(m, records=1)

    with gp.Loop(gp.Domain(i, j).where[q[i, j] > 0]):
        x[...] = x[...] + q[i, j]

    assert x.toValue() == 7.0

    with gp.Loop(i.where[c[i] + c[i] ** 2]):
        z[...] = z[...] + 1

    assert z.toValue() == 3.0

    with gp.Loop(i.where[gp.Sum(j, gp.math.abs(q[i, j]))]):
        z[...] = z[...] + 1

    assert z.toValue() == 4.0

    with gp.Loop(j.where[(gp.Ord(j) > 1) & (gp.Ord(j) < gp.Card(j))]):
        z[...] = z[...] + 1

    assert z.toValue() == 7.0

    with gp.Loop(gp.Domain(i, j).where[k[i, j]]):
        y[...] = y[...] + gp.Ord(i) + 2 * gp.Ord(j)

    assert y.toValue() == 35.0


def test_loop_with_solve():
    m = gp.Container()

    # Prepare data
    distances = [
        ["seattle", "new-york", 2.5],
        ["seattle", "chicago", 1.7],
        ["seattle", "topeka", 1.8],
        ["san-diego", "new-york", 2.5],
        ["san-diego", "chicago", 1.8],
        ["san-diego", "topeka", 1.4],
    ]

    capacities = [["seattle", 350], ["san-diego", 600]]
    demands = [["new-york", 325], ["chicago", 300], ["topeka", 275]]

    # Set
    i = gp.Set(
        m,
        name="i",
        records=["seattle", "san-diego"],
        description="canning plants",
    )
    j = gp.Set(
        m,
        name="j",
        records=["new-york", "chicago", "topeka"],
        description="markets",
    )

    # Data
    a = gp.Parameter(
        m,
        name="a",
        domain=i,
        records=capacities,
        description="capacity of plant i in cases",
    )
    b = gp.Parameter(
        m,
        name="b",
        domain=j,
        records=demands,
        description="demand at market j in cases",
    )
    d = gp.Parameter(
        m,
        name="d",
        domain=[i, j],
        records=distances,
        description="distance in thousands of miles",
    )
    c = gp.Parameter(
        m,
        name="c",
        domain=[i, j],
        description="transport cost in thousands of dollars per case",
    )
    c[i, j] = 90 * d[i, j] / 1000

    # Variable
    x = gp.Variable(
        m,
        name="x",
        domain=[i, j],
        type="Positive",
        description="shipment quantities in cases",
    )

    # Equation
    supply = gp.Equation(
        m,
        name="supply",
        domain=i,
        description="observe supply limit at plant i",
    )
    demand = gp.Equation(
        m, name="demand", domain=j, description="satisfy demand at market j"
    )

    supply[i] = gp.Sum(j, x[i, j]) <= a[i]
    demand[j] = gp.Sum(i, x[i, j]) >= b[j]

    transport = gp.Model(
        m,
        name="transport",
        equations=m.getEquations(),
        problem="LP",
        sense=gp.Sense.MIN,
        objective=gp.Sum((i, j), c[i, j] * x[i, j]),
    )

    dd = gp.Parameter(m, domain=i)
    mode = gp.Parameter(m, records=1)
    cnt = gp.Parameter(m, records=0)
    k = gp.Set(m, domain=[i, j])
    kval = gp.Set(m, domain=[i, j])
    k[i, j] = True

    if mode.toValue() == 1:
        with gp.Loop(k):
            kval[k] = True
            transport.solve()
            cnt[...] = cnt[...] + 1
            c[i, j] = c[i, j] * 1.1

        dd[i] = 10

    assert kval.toList() == [
        ("seattle", "new-york"),
        ("seattle", "chicago"),
        ("seattle", "topeka"),
        ("san-diego", "new-york"),
        ("san-diego", "chicago"),
        ("san-diego", "topeka"),
    ]


def test_loop_domain_tree():
    m = gp.Container()
    i = gp.Set(m, records=[f"i{idx}" for idx in range(1, 11)])
    ii = gp.Set(m, domain=i)
    j = gp.Set(m, domain=i, records=[f"i{idx}" for idx in range(1, 10)])
    jj = gp.Set(m, domain=j, records=[f"i{idx}" for idx in range(1, 9)])
    jjj = gp.Set(m, domain=jj, records=[f"i{idx}" for idx in range(1, 8)])

    with gp.Loop(i[jjj]):
        ii[i] = True

    assert ii.toList() == ["i1", "i2", "i3", "i4", "i5", "i6", "i7"]


def test_nested_loops():
    m = gp.Container()
    i = gp.Set(m, records=range(3))
    j = gp.Set(m, records=range(4))
    a = gp.Parameter(m, domain=[i, j])
    a.generateRecords()
    b = gp.Parameter(m, records=0)

    with gp.Loop(i):  # noqa: SIM117
        with gp.Loop(j):
            b[...] += a[i, j]

    assert np.isclose(a.records["value"].sum(), b.toValue())


def test_invalid_indices():
    m = gp.Container()
    i = gp.Set(m, records=range(3))
    a = gp.Parameter(m, domain=i)
    b = gp.Parameter(m)

    with pytest.raises(ValidationError):  # noqa: SIM117
        with gp.Loop(5):
            b[...] = a[i]

    with pytest.raises(ValidationError):  # noqa: SIM117
        with gp.Loop("5"):
            b[...] = a[i]
