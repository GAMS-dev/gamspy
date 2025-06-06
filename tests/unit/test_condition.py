from __future__ import annotations

import pandas as pd
import pytest

import gamspy.math as gamspy_math
from gamspy import (
    Container,
    Equation,
    Model,
    Number,
    Ord,
    Parameter,
    Sense,
    Set,
    Sum,
    Variable,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def data():
    m = Container()
    steel_plants = [
        "ahmsa",
        "fundidora",
        "sicartsa",
        "hylsa",
        "hylsap",
    ]
    markets = ["mexico-df", "monterrey", "guadalaja"]
    rail_distances = pd.DataFrame(
        [
            ["ahmsa", "mexico-df", 1204],
            ["ahmsa", "monterrey", 218],
            ["ahmsa", "guadalaja", 1125],
            ["ahmsa", "export", 739],
            ["fundidora", "mexico-df", 1017],
            ["fundidora", "guadalaja", 1030],
            ["fundidora", "export", 521],
            ["sicartsa", "mexico-df", 819],
            ["sicartsa", "monterrey", 1305],
            ["sicartsa", "guadalaja", 704],
            ["hylsa", "mexico-df", 1017],
            ["hylsa", "guadalaja", 1030],
            ["hylsa", "export", 521],
            ["hylsap", "mexico-df", 185],
            ["hylsap", "monterrey", 1085],
            ["hylsap", "guadalaja", 760],
            ["hylsap", "export", 315],
            ["import", "mexico-df", 428],
            ["import", "monterrey", 521],
            ["import", "guadalaja", 300],
        ]
    )

    yield m, steel_plants, markets, rail_distances
    m.close()


def test_condition_on_expression(data):
    m, steel_plants, markets, rail_distances = data
    i = Set(
        m,
        name="i",
        records=steel_plants,
        description="steel plants",
    )
    j = Set(
        m,
        name="j",
        records=markets,
        description="markets",
    )

    rd = Parameter(
        m,
        name="rd",
        domain=["*", "*"],
        records=rail_distances,
        description="rail distances from plants to markets",
    )
    muf = Parameter(
        m,
        name="muf",
        domain=[i, j],
        description="transport rate: final products",
    )

    # Condition
    muf[i, j] = (2.48 + 0.0084 * rd[i, j]).where[rd[i, j]]

    assert (
        muf.getAssignment()
        == "muf(i,j) = ((2.48 + (0.0084 * rd(i,j))) $ rd(i,j));"
    )


def test_condition_on_number(data):
    m, steel_plants, markets, _ = data
    i = Set(
        m,
        name="i",
        records=steel_plants,
        description="steel plants",
    )
    j = Set(
        m,
        name="j",
        records=markets,
        description="markets",
    )
    k = Set(
        m,
        name="k",
        records=steel_plants,
        description="steel plants",
    )

    p = Set(m, name="p", records=[f"pos{elem}" for elem in range(1, 11)])
    o = Set(m, name="o", records=[f"opt{elem}" for elem in range(1, 6)])

    sumc = Parameter(m, name="sumc", domain=[o, p])
    sumc[o, p] = gamspy_math.uniform(0, 1)

    op = Variable(m, name="op", type="free", domain=[o, p])

    defopLS = Equation(m, name="defopLS", domain=[o, p])
    defopLS[o, p].where[sumc[o, p] <= 0.5] = op[o, p] == 1

    assert (
        defopLS.getDefinition()
        == "defopLS(o,p) $ (sumc(o,p) <= 0.5) .. op(o,p) =e= 1;"
    )

    muf = Parameter(
        m,
        name="muf",
        domain=[i, j],
        description="transport rate: final products",
    )

    expression = Sum(i, muf[i, j]).where[muf[i, j] > 0]
    assert expression.getDeclaration() == "(sum(i,muf(i,j)) $ (muf(i,j) > 0))"

    k["ahmsa"] = True
    assert k.getAssignment() == 'k("ahmsa") = yes;'

    k["ahmsa"] = False
    assert k.getAssignment() == 'k("ahmsa") = no;'

    t = Set(
        m,
        name="t",
        records=[str(i) for i in range(1, 10)],
        description="no. of Monte-Carlo draws",
    )

    Util_gap = Parameter(
        m,
        name="Util_gap",
        domain=[t],
        description="gap between these two util",
    )

    Util_lic = Parameter(
        m,
        name="Util_lic",
        domain=[t],
        description="util solved w/o MN",
    )
    Util_lic2 = Parameter(
        m,
        name="Util_lic2",
        domain=[t],
        description="util solved w/ MN",
    )

    Util_gap[t] = Number(1).where[
        gamspy_math.Round(Util_lic[t], 10)
        != gamspy_math.Round(Util_lic2[t], 10)
    ]

    assert (
        Util_gap.getAssignment()
        == "Util_gap(t) = (1 $ (round(Util_lic(t),10) ne round("
        "Util_lic2(t),10)));"
    )


def test_condition_on_equation(data):
    m, *_ = data
    td_data = pd.DataFrame(
        [
            ["icbm", "2", 0.05],
            ["icbm", "6", 0.15],
            ["icbm", "7", 0.10],
            ["icbm", "8", 0.15],
            ["icbm", "9", 0.20],
            ["icbm", "18", 0.05],
            ["mrbm-1", "1", 0.16],
            ["mrbm-1", "2", 0.17],
            ["mrbm-1", "3", 0.15],
            ["mrbm-1", "4", 0.16],
            ["mrbm-1", "5", 0.15],
            ["mrbm-1", "6", 0.19],
            ["mrbm-1", "7", 0.19],
            ["mrbm-1", "8", 0.18],
            ["mrbm-1", "9", 0.20],
            ["mrbm-1", "10", 0.14],
            ["mrbm-1", "12", 0.02],
            ["mrbm-1", "14", 0.12],
            ["mrbm-1", "15", 0.13],
            ["mrbm-1", "16", 0.12],
            ["mrbm-1", "17", 0.15],
            ["mrbm-1", "18", 0.16],
            ["mrbm-1", "19", 0.15],
            ["mrbm-1", "20", 0.15],
            ["lr-bomber", "1", 0.04],
            ["lr-bomber", "2", 0.05],
            ["lr-bomber", "3", 0.04],
            ["lr-bomber", "4", 0.04],
            ["lr-bomber", "5", 0.04],
            ["lr-bomber", "6", 0.10],
            ["lr-bomber", "7", 0.08],
            ["lr-bomber", "8", 0.09],
            ["lr-bomber", "9", 0.08],
            ["lr-bomber", "10", 0.05],
            ["lr-bomber", "11", 0.01],
            ["lr-bomber", "12", 0.02],
            ["lr-bomber", "13", 0.01],
            ["lr-bomber", "14", 0.02],
            ["lr-bomber", "15", 0.03],
            ["lr-bomber", "16", 0.02],
            ["lr-bomber", "17", 0.05],
            ["lr-bomber", "18", 0.08],
            ["lr-bomber", "19", 0.07],
            ["lr-bomber", "20", 0.08],
            ["f-bomber", "10", 0.04],
            ["f-bomber", "11", 0.09],
            ["f-bomber", "12", 0.08],
            ["f-bomber", "13", 0.09],
            ["f-bomber", "14", 0.08],
            ["f-bomber", "15", 0.02],
            ["f-bomber", "16", 0.07],
            ["mrbm-2", "1", 0.08],
            ["mrbm-2", "2", 0.06],
            ["mrbm-2", "3", 0.08],
            ["mrbm-2", "4", 0.05],
            ["mrbm-2", "5", 0.05],
            ["mrbm-2", "6", 0.02],
            ["mrbm-2", "7", 0.02],
            ["mrbm-2", "10", 0.10],
            ["mrbm-2", "11", 0.05],
            ["mrbm-2", "12", 0.04],
            ["mrbm-2", "13", 0.09],
            ["mrbm-2", "14", 0.02],
            ["mrbm-2", "15", 0.01],
            ["mrbm-2", "16", 0.01],
        ]
    )

    wa_data = pd.DataFrame(
        [
            ["icbm", 200],
            ["mrbm-1", 100],
            ["lr-bomber", 300],
            ["f-bomber", 150],
            ["mrbm-2", 250],
        ]
    )

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

    w = Set(
        m,
        name="w",
        records=["icbm", "mrbm-1", "lr-bomber", "f-bomber", "mrbm-2"],
    )
    t = Set(m, name="t", records=[str(i) for i in range(1, 21)])

    td = Parameter(m, name="td", domain=[w, t], records=td_data)
    wa = Parameter(m, name="wa", domain=[w], records=wa_data)
    tm = Parameter(m, name="tm", domain=[t], records=tm_data)

    x = Variable(m, name="x", domain=[w, t], type="Positive")

    maxw = Equation(m, name="maxw", domain=[w])
    minw = Equation(m, name="minw", domain=[t])

    maxw[w] = Sum(t.where[td[w, t]], x[w, t]) <= wa[w]
    minw[t].where[tm[t]] = Sum(w.where[td[w, t]], x[w, t]) >= tm[t]

    assert (
        minw.getDefinition()
        == "minw(t) $ tm(t) .. sum(w $ td(w,t),x(w,t)) =g= tm(t);"
    )

    m = Container()

    p = Set(m, name="p", records=[f"pos{i}" for i in range(1, 11)])
    o = Set(m, name="o", records=[f"opt{i}" for i in range(1, 6)])

    sumc = Variable(m, name="sumc", type="free", domain=[o, p])
    op = Variable(m, name="op", type="free", domain=[o, p])

    defopLS = Equation(m, name="defopLS", domain=[o, p])
    defopLS[o, p] = op[o, p] == Number(1).where[sumc[o, p] >= 0.5]
    assert (
        defopLS.getDefinition()
        == "defopLS(o,p) .. op(o,p) =e= (1 $ (sumc(o,p) >= 0.5));"
    )

    k = Set(m, "k", domain=[p])
    k[p].where[k[p]] = True

    assert k.getAssignment() == "k(p) $ k(p) = yes;"

    m = Container()
    p = Set(m, name="p", records=[f"pos{i}" for i in range(1, 11)])
    k = Set(m, "k", domain=[p])
    k[p].where[k[p]] = True


def test_variable_discovery_in_implicit_equation(data):
    m, *_ = data
    # Instance Data
    products = ["Product_A", "Product_B", "Product_C"]
    time_periods = [1, 2, 3, 4]

    # Example data for parameters
    demand_data = pd.DataFrame(
        {
            "Product_A": {1: 100, 2: 150, 3: 120, 4: 180},
            "Product_B": {1: 80, 2: 100, 3: 90, 4: 120},
            "Product_C": {1: 50, 2: 60, 3: 70, 4: 80},
        }
    ).unstack()

    setup_cost_data = pd.DataFrame(
        [("Product_A", 500), ("Product_B", 400), ("Product_C", 300)]
    )
    holding_cost_data = pd.DataFrame(
        [("Product_A", 10), ("Product_B", 8), ("Product_C", 6)]
    )

    i = Set(m, name="i", description="products", records=products)
    t = Set(m, name="t", description="time periods", records=time_periods)

    d = Parameter(
        m,
        name="d",
        domain=[i, t],
        description="demand of product i in period t",
        records=demand_data,
    )
    s = Parameter(
        m,
        name="s",
        domain=i,
        description="fixed setup cost for product i",
        records=setup_cost_data,
    )
    h = Parameter(
        m,
        name="h",
        domain=i,
        description="holding cost for product i",
        records=holding_cost_data,
    )

    X = Variable(
        m,
        name="X",
        domain=[i, t],
        type="positive",
        description="lot size of product i in period t",
    )
    Y = Variable(
        m,
        name="Y",
        domain=[i, t],
        type="binary",
        description="indicates if product i is manufactures in period t",
    )
    Z = Variable(
        m,
        name="Z",
        domain=[i, t],
        type="positive",
        description="stock of product i in period t",
    )

    objective = Sum((i, t), s[i] * Y[i, t] + h[i] * Z[i, t])

    stock = Equation(
        m,
        name="stock",
        domain=[i, t],
        description="Stock balance equation",
    )
    stock[i, t].where[Ord(t) > 1] = (
        Z[i, t.lag(1)] + X[i, t] - Z[i, t] == d[i, t]
    )

    clsp = Model(
        m,
        name="CLSP",
        problem="MIP",
        equations=m.getEquations(),
        sense=Sense.MIN,
        objective=objective,
    )

    clsp.solve()

    assert X.records is not None


def test_operator_comparison_in_condition(data):
    m, *_ = data
    s = Set(m, name="s", records=[str(i) for i in range(1, 4)])
    c = Parameter(m, name="c", domain=[s])
    c[s].where[Ord(s) <= Ord(s)] = 1

    assert c.getAssignment() == "c(s) $ (ord(s) <= ord(s)) = 1;"


def test_condition_on_condition():
    m = Container()
    i = Set(m, "i")
    a = Parameter(m, "a", domain=i)

    assert a[i].where[a[i]].where[a[i]].gamsRepr() == "((a(i) $ a(i)) $ a(i))"
    assert (
        a[i].where[a[i]].where[a[i]].where[a[i]].gamsRepr()
        == "(((a(i) $ a(i)) $ a(i)) $ a(i))"
    )
