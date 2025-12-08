import math

import numpy as np
import pytest

import gamspy as gp

pytestmark = pytest.mark.unit


@pytest.fixture
def data():
    m = gp.Container()
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


def test_expression_evaluation():
    m = gp.Container()
    i = gp.Set(m, "i", records=["i1", "i2"])
    j = gp.Set(m, "j", records=["j1", "j2"])
    a = gp.Parameter(
        m,
        "a",
        domain=[i, j],
        records=[
            ["i1", "j1", 5],
            ["i1", "j2", 10],
            ["i2", "j1", 15],
            ["i2", "j2", 20],
        ],
    )
    b = gp.Parameter(
        m,
        "b",
        domain=[i, j],
        records=[
            ["i1", "j1", 2],
            ["i1", "j2", 4],
            ["i2", "j1", 6],
            ["i2", "j2", 8],
        ],
    )

    c = gp.Parameter(m, "c", domain=[i, j])

    # expression.records

    # full domain
    c[...] = a + b
    assert c.records.equals((a + b).records)
    assert (a + b).toList() == [
        ["i1", "j1", 7.0],
        ["i1", "j2", 14.0],
        ["i2", "j1", 21.0],
        ["i2", "j2", 28.0],
    ]

    # Indices are set + literal
    c[...] = a[i, "j1"] + b["i1", j]
    assert c.records.equals((a[i, "j1"] + b["i1", j]).records)

    # Indices are all literals
    assert (a["i1", "j1"] + b["i1", "j1"]).records.values.tolist()[0][0] == 7.0

    # Calculation with dimension = 0
    d = gp.Parameter(m, "d", records=5)
    e = gp.Parameter(m, "e", records=6)
    assert (d + e).records.values.tolist()[0][0] == 11
    assert (d + e).toValue() == 11
    assert ((d + e) * 5).toValue() == 55

    # operation.records
    c[...] = a + b
    e[...] = gp.Sum((i, j), c[i, j])
    assert e.records.equals(gp.Sum((i, j), c[i, j]).records)
    assert gp.Sum(j, c[i, j]).toList() == [["i1", 21.0], ["i2", 49.0]]
    e[...] = gp.Sum(i, gp.Sum(j, c[i, j]))
    assert e.records.equals(gp.Sum(i, gp.Sum(j, c[i, j])).records)

    # mathop.records
    f = gp.Parameter(m, "f", records=5)
    assert math.isclose(gp.math.exp(f).records.values.item(), 148.4131591025766)
    assert math.isclose(gp.math.sin(90).records.values.item(), 0.8939966636005579)
    assert math.isclose(gp.math.sin(90).toValue(), 0.8939966636005579)

    k = gp.Set(m, name="k", records=["k1", "k2", "k3"])
    g = gp.Parameter(m, "g", domain=k, records=[("k1", 4), ("k2", 10), ("k3", 0.5)])
    assert gp.math.lse_max(g[k], 5).records.values.tolist() == [
        ["k1", 5.313261687518223],
        ["k2", 10.006715348489118],
        ["k3", 5.011047744848594],
    ]
    assert gp.math.lse_max(g[k], 5).toList() == [
        ["k1", 5.313261687518223],
        ["k2", 10.006715348489118],
        ["k3", 5.011047744848594],
    ]

    # condition.records
    c[...] = a.where[a > 2]
    assert c.records.equals(a.where[a > 2].records)

    seed = 123
    m = gp.Container()
    A = gp.Set(m, records=range(2))
    S = gp.Set(m, records=range(2))
    AS = gp.Set(m, domain=[A, S])
    AS.generateRecords(seed=seed)
    risk_weight = gp.Parameter(m, domain=[A, S])
    risk_weight.generateRecords(seed=seed)
    exposure = gp.Parameter(m, domain=[A, S])
    segment_vars = gp.Parameter(m, domain=[A, S])
    risk_weight.generateRecords(seed=seed)
    exposure.generateRecords(seed=seed)
    segment_vars.generateRecords(seed=seed)
    assert np.isclose(
        gp.Sum(AS[A, S], risk_weight[AS] * exposure[AS] * segment_vars[AS])
        .records["value"]
        .to_numpy(),
        np.array([3.17705801e-01, 1.55903456e-04, 1.07003390e-02, 6.26734443e-03]),
    ).all()

    m.close()


def test_assume_variable_suffix(data):
    m, canning_plants, markets, capacities, demands, distances = data

    i = gp.Set(m, name="i", records=canning_plants)
    j = gp.Set(m, name="j", records=markets)

    a = gp.Parameter(m, name="a", domain=i, records=capacities)
    b = gp.Parameter(m, name="b", domain=j, records=demands)
    d = gp.Parameter(m, name="d", domain=[i, j], records=distances)
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
    obj = gp.Sum((i, j), c[i, j] * x[i, j])

    transport = gp.Model(
        m,
        name="transport",
        equations=m.getEquations(),
        problem="LP",
        sense=gp.Sense.MIN,
        objective=obj,
    )
    transport.solve()

    assert obj.toValue() == 153.675
    assert obj.records.values.tolist()[0][0] == 153.675

    gp.set_options({"ASSUME_VARIABLE_SUFFIX": 2})
    assert obj.records.values.tolist()[0][0] == 1.053
    gp.set_options({"ASSUME_VARIABLE_SUFFIX": 1})


def test_empty_indices():
    m = gp.Container()

    i = gp.Set(m, "i", records=["x", "y", "z"])
    a = gp.Parameter(m, "a", domain=i, records=[("x", 1), ("y", 2), ("z", 3)])
    a.where[a > 2] = a * 3
    assert a.records["value"].values.tolist() == [1, 2, 9]
    assert a.getAssignment() == "a(i) $ (a(i) > 2) = a(i) * 3;"

    v = gp.Variable(m, "v", domain=i)
    v.l = 5
    assert v.records["level"].values.tolist() == [5, 5, 5]

    e = gp.Equation(m, "e", domain=i)
    e.m = 5
    assert e.records["marginal"].values.tolist() == [5, 5, 5]

    e[i].where[gp.Ord(i) > 1] = v <= 5
    assert e.getDefinition() == "e(i) $ (ord(i) > 1) .. v(i) =l= 5;"
