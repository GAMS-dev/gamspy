from __future__ import annotations

import pandas as pd
import pytest

from gamspy import (
    Alias,
    Card,
    Container,
    Domain,
    Equation,
    Ord,
    Parameter,
    Product,
    Sand,
    Set,
    Smax,
    Smin,
    Sor,
    Sum,
    Variable,
)
from gamspy.exceptions import ValidationError

pytestmark = pytest.mark.unit


@pytest.fixture
def data():
    m = Container()
    canning_plants = ["seattle", "san-diego"]
    markets = ["new-york", "chicago", "topeka"]
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

    yield m, canning_plants, markets, capacities, demands, distances
    m.close()


def test_operations(data):
    m, canning_plants, markets, capacities, _, distances = data
    i = Set(
        m,
        name="i",
        records=canning_plants,
        description="Canning Plants",
    )
    j = Set(m, name="j", records=markets, description="Markets")

    # Params
    a = Parameter(m, name="a", domain=[i], records=capacities)
    d = Parameter(m, name="d", domain=[i, j], records=distances)
    c = Parameter(m, name="c", domain=[i, j])
    c[i, j] = 90 * d[i, j] / 1000

    x = Variable(m, name="x", domain=[i, j], type="Positive")
    z = Variable(m, name="z")

    # SUM
    # Operation with one index
    sum_op = Sum(j, x[i, j]) <= a[i]
    assert sum_op.gamsRepr() == "sum(j,x(i,j)) =l= a(i)"

    expression = Sum(i, True)
    assert expression.gamsRepr() == "sum(i,yes)"

    # Operation with two indices
    sum_op = Sum((i, j), c[i, j] * x[i, j]) == z
    assert sum_op.gamsRepr() == "sum((i,j),(c(i,j) * x(i,j))) =e= z"

    # PROD
    # Operation with one index
    sum_op = Product(j, x[i, j]) <= a[i]
    assert sum_op.gamsRepr() == "prod(j,x(i,j)) =l= a(i)"

    # Operation with two indices
    sum_op = Product((i, j), c[i, j] * x[i, j]) == z
    assert sum_op.gamsRepr() == "prod((i,j),(c(i,j) * x(i,j))) =e= z"

    # Smin
    # Operation with one index
    sum_op = Smin(j, x[i, j]) <= a[i]
    assert sum_op.gamsRepr() == "smin(j,x(i,j)) =l= a(i)"

    # Operation with two indices
    sum_op = Smin((i, j), c[i, j] * x[i, j]) == z
    assert sum_op.gamsRepr() == "smin((i,j),(c(i,j) * x(i,j))) =e= z"

    # Smax
    # Operation with one index
    sum_op = Smax(j, x[i, j]) <= a[i]
    assert sum_op.gamsRepr() == "smax(j,x(i,j)) =l= a(i)"

    # Operation with two indices
    smax_op = Smax((i, j), c[i, j] * x[i, j]) == z
    assert smax_op.gamsRepr() == "smax((i,j),(c(i,j) * x(i,j))) =e= z"

    # Sand
    # Operation with one index
    sand_op = Sand(j, x[i, j]) <= a[i]
    assert sand_op.gamsRepr() == "sand(j,x(i,j)) =l= a(i)"

    # Operation with two indices
    sand_op = Sand((i, j), c[i, j] * x[i, j]) == z
    assert sand_op.gamsRepr() == "sand((i,j),(c(i,j) * x(i,j))) =e= z"

    # Sor
    # Operation with one index
    sor_op = Sor(j, x[i, j]) <= a[i]
    assert sor_op.gamsRepr() == "sor(j,x(i,j)) =l= a(i)"

    # Operation with two indices
    sor_op = Sor((i, j), c[i, j] * x[i, j]) == z
    assert sor_op.gamsRepr() == "sor((i,j),(c(i,j) * x(i,j))) =e= z"

    # Ord, Card
    with pytest.raises(ValidationError):
        _ = Ord("bla")

    with pytest.raises(ValidationError):
        _ = Card("bla")

    expression = Ord(i) == Ord(j)
    assert expression.gamsRepr() == "(ord(i) eq ord(j))"
    expression = Ord(i) != Ord(j)
    assert expression.gamsRepr() == "(ord(i) ne ord(j))"
    expression = Card(i) == 5
    assert expression.gamsRepr() == "(card(i) eq 5)"
    expression = Card(i) != 5
    assert expression.gamsRepr() == "(card(i) ne 5)"
    expression = Card(i) <= 5
    assert expression.gamsRepr() == "(card(i) <= 5)"
    expression = Card(i) >= 5
    assert expression.gamsRepr() == "(card(i) >= 5)"

    sum_op = Sum((i, j), c[i, j] * x[i, j])
    expression = sum_op != sum_op
    assert (
        expression.gamsRepr()
        == "(sum((i,j),(c(i,j) * x(i,j))) ne sum((i,j),(c(i,j) * x(i,j))))"
    )


def test_operation_indices(data):
    m, *_ = data
    # Test operation index
    m = Container()
    mt = 2016
    mg = 17
    maxdt = 40
    t = Set(
        m,
        name="t",
        records=[f"t{i}" for i in range(1, mt + 1)],
        description="hours",
    )
    g = Set(
        m,
        name="g",
        records=[f"g{i}" for i in range(1, mg + 1)],
        description="generators",
    )
    t1 = Alias(m, name="t1", alias_with=t)
    tt = Set(
        m,
        name="tt",
        domain=[t],
        records=[f"t{i}" for i in range(1, maxdt + 1)],
        description="max downtime hours",
    )
    pMinDown = Parameter(
        m, name="pMinDown", domain=[g, t], description="minimum downtime"
    )
    vStart = Variable(m, name="vStart", type="binary", domain=[g, t])
    eStartFast = Equation(m, name="eStartFast", domain=[g, t])
    eStartFast[g, t1] = (
        Sum(
            tt[t].where[Ord(t) <= pMinDown[g, t1]],
            vStart[g, t.lead(Ord(t1) - pMinDown[g, t1])],
        )
        <= 1
    )
    assert (
        eStartFast.getDefinition()
        == "eStartFast(g,t1) .. sum(tt(t) $ (ord(t) <="
        " pMinDown(g,t1)),vStart(g,t + (ord(t1) - pMinDown(g,t1))))"
        " =l= 1;"
    )


def test_operation_overloads(data):
    m, *_ = data
    m = Container()
    c = Set(m, "c")
    s = Set(m, "s")
    a = Parameter(m, "a", domain=[c, s])
    p = Variable(m, "p", type="Positive", domain=c)

    # test neq
    profit = Equation(m, "profit", domain=s)
    profit[s] = -Sum(c, a[c, s] * p[c]) >= 0
    assert (
        profit.getDefinition()
        == "profit(s) .. ( - sum(c,(a(c,s) * p(c)))) =g= 0;"
    )

    # test ne
    bla = Parameter(m, "bla", domain=s)
    bla2 = Parameter(m, "bla2", domain=s)
    bla[...] = bla2[...] != 0

    assert bla.getAssignment() == "bla(s) = (bla2(s) ne 0);"


def test_truth_value(data):
    m, *_ = data
    i_list = [f"i{i}" for i in range(10)]
    i = Set(m, "i", records=i_list)
    j = Alias(m, "j", alias_with=i)
    x = Variable(m, "x", domain=[i, j])
    eq = Equation(m, "eq", domain=[i, j])

    with pytest.raises(ValidationError):
        eq[i, j].where[
            (Ord(i) < Card(i)) and (Ord(j) > 1) and (Ord(j) < Card(j))
        ] = x[i, j] >= 1

    with pytest.raises(ValidationError):
        if Card(i):
            ...

    with pytest.raises(ValidationError):
        if i:
            ...

    with pytest.raises(ValidationError):
        if j:
            ...

    a = Parameter(m, "a")
    with pytest.raises(ValidationError):
        if a:
            ...

    v = Variable(m, "v")
    with pytest.raises(ValidationError):
        if v:
            ...

    e = Equation(m, "e")
    with pytest.raises(ValidationError):
        if e:
            ...


def test_condition(data):
    m, *_ = data
    m = Container()
    jj = Set(m, "jj", records=[f"n{idx}" for idx in range(1, 11)])
    depot = Set(m, "depot", domain=jj, records=["n10"])

    card_j = Parameter(m, "card_j", domain=jj)
    card_j[jj] = Card(jj).where[depot[jj]]
    assert card_j.getAssignment() == "card_j(jj) = (card(jj) $ depot(jj));"

    ord_j = Parameter(m, "ord_j", domain=jj)
    ord_j[jj] = Ord(jj).where[depot[jj]]
    assert ord_j.getAssignment() == "ord_j(jj) = (ord(jj) $ depot(jj));"


def test_control_domain(data):
    m, *_ = data
    i = Set(m, "i", records=[f"i{idx}" for idx in range(1, 4)])
    j = Set(m, "j", records=[f"j{idx}" for idx in range(1, 4)])

    a = Parameter(m, "a")
    b = Parameter(m, "b", domain=[i, j])

    # Assignment
    with pytest.raises(ValidationError):
        b[i, j] = Sum(i, 1)

    # Operation inside operation
    with pytest.raises(ValidationError):
        a[...] = Sum(i, Sum(i, 1)) + Sum((i, j), 1)

    # Condition
    x = Variable(m, "k", domain=i)
    silly = Equation(m, "silly")
    silly[...] = Sum(i, x[i]) >= Card(i)

    c = Parameter(m, name="c", domain=[i])

    with pytest.raises(ValidationError):
        c[i] = Sum(Domain(i, j).where[i.sameAs(j)], 1)

    # Condition
    tm_data = pd.DataFrame(
        [
            ["1", 30],
            ["6", 100],
            ["10", 40],
            ["14", 50],
            ["15", 70],
            ["16", 35],
            ["20", 10],
        ]
    )

    # Sets
    w = Set(
        m,
        name="w",
        records=["icbm", "mrbm-1", "lr-bomber", "f-bomber", "mrbm-2"],
        description="weapons",
    )
    t = Set(
        m,
        name="t",
        records=[str(i) for i in range(1, 21)],
        description="targets",
    )

    # Parameters
    tm = Parameter(
        m,
        name="tm",
        domain=t,
        records=tm_data,
        description="minimum number of weapons per target",
    )

    # Variables
    x = Variable(
        m,
        name="x",
        domain=[w, t],
        type="Positive",
        description="weapons assignment",
    )

    # Equations
    minw = Equation(
        m,
        name="minw",
        domain=t,
        description="minimum number of weapons required per target",
    )

    with pytest.raises(ValidationError):
        minw[t].where[tm[t]] = Sum(t, tm[t]) >= tm[t]
