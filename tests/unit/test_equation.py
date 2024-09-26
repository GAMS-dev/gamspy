from __future__ import annotations

import gamspy._symbols.implicits as implicits
import numpy as np
import pandas as pd
import pytest
from gamspy import (
    Alias,
    Card,
    Container,
    Equation,
    EquationType,
    Model,
    Options,
    Ord,
    Parameter,
    Problem,
    Product,
    Sense,
    Set,
    Sum,
    Variable,
)
from gamspy.exceptions import ValidationError
from gamspy.math import sqr

pytestmark = pytest.mark.unit


@pytest.fixture
def data():
    m = Container()
    canning_plants = ["seattle", "san-diego"]
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

    yield m, canning_plants, capacities, demands, distances
    m.close()


def test_equation_creation(data):
    m, *_ = data
    # no name is fine now
    e1 = Equation(m)
    m.addEquation()
    with pytest.raises(ValidationError):
        _ = e1.getDefinition()

    # non-str type name
    pytest.raises(TypeError, Equation, m, 5)

    # no container
    pytest.raises(TypeError, Equation)

    # non-container type container
    pytest.raises(TypeError, Equation, 5, "j")

    # try to create a symbol with same name but different type
    _ = Set(m, "i")
    pytest.raises(TypeError, Equation, m, "i")

    # get already created symbol
    j1 = Equation(m, "j")
    j2 = Equation(m, "j")
    assert id(j1) == id(j2)

    # Equation and domain containers are different
    m2 = Container()
    set1 = Set(m, "set1")
    with pytest.raises(ValidationError):
        _ = Equation(m2, "eq1", domain=[set1])


def test_equation_types(data):
    m, *_ = data
    # Prepare data
    canning_plants = ["seattle", "san-diego"]

    x = Variable(
        m,
        name="x",
        domain=[],
        records={"lower": 1.0, "level": 1.5, "upper": 3.75},
    )

    i = Set(m, name="i", records=canning_plants, description="Canning Plants")

    c = Parameter(m, name="c", domain=[i], records=np.array([0.5, 0.6]))

    d = Parameter(m, name="d", records=0.5)
    eq1 = Equation(m, "eq1", type="nonbinding")
    eq1[...] = (x - d) == 0
    assert eq1.getDefinition() == "eq1 .. (x - d) =n= 0;"
    assert eq1.type == "nonbinding"

    y = Variable(m, "y", domain=[i])
    eq2 = Equation(m, "eq2", domain=[i], type="nonbinding")
    eq2[i] = (y[i] - c[i]) == 0
    assert eq2.getDefinition() == "eq2(i) .. (y(i) - c(i)) =n= 0;"

    eq2[i] = (y[i] - c[i]) == 0
    assert eq2.getDefinition() == "eq2(i) .. (y(i) - c(i)) =n= 0;"

    # eq
    eq3 = Equation(m, "eq3", domain=[i])
    eq3[i] = y[i] == c[i]
    assert eq3.type == "eq"

    # geq
    eq4 = Equation(m, "eq4", domain=[i])
    eq4[i] = y[i] >= c[i]
    assert eq4.type == "eq"

    # leq
    eq5 = Equation(m, "eq5", domain=[i])
    eq5[i] = y[i] <= c[i]
    assert eq5.type == "eq"

    assert str(EquationType.REGULAR) == "REGULAR"
    eq6 = Equation(m, "eq6", type=EquationType.REGULAR, domain=[i])
    assert eq6.type == "eq"

    assert EquationType.values() == [
        "REGULAR",
        "NONBINDING",
        "EXTERNAL",
        "BOOLEAN",
    ]

    eq6 = Equation(m, "eq6", domain=[i])
    with pytest.raises(ValidationError):
        eq6[i] = y[i] - c[i]


def test_nonbinding(data):
    m, *_ = data
    x = Variable(m, "x")
    e = Equation(m, "e", definition=x == 0, type="NONBINDING")
    assert e.getDefinition() == "e .. x =n= 0;"

    x1 = Variable(m, "x1")
    e1 = Equation(m, "e1", definition=x1 >= 0, type="NONBINDING")
    assert e1.getDefinition() == "e1 .. x1 =n= 0;"

    x2 = Variable(m, "x2")
    e2 = Equation(m, "e2", definition=x2 <= 0, type="NONBINDING")
    assert e2.getDefinition() == "e2 .. x2 =n= 0;"


def test_equation_declaration(data):
    m, *_ = data
    # Check if the name is reserved
    pytest.raises(ValidationError, Equation, m, "set")

    # Prepare data
    canning_plants = ["seattle", "san-diego"]
    markets = ["new-york", "chicago", "topeka"]

    i = Set(m, name="i", records=canning_plants, description="Canning Plants")
    j = Set(m, name="j", records=markets, description="Markets")

    # Equation declaration without an index
    cost = Equation(m, name="cost", description="define objective function")
    assert cost.gamsRepr() == "cost"
    assert (
        cost.getDeclaration() == 'Equation cost "define objective function";'
    )

    # Equation declaration with an index
    supply = Equation(
        m,
        name="supply",
        domain=[i],
        description="observe supply limit at plant i",
    )
    assert supply.gamsRepr() == "supply"
    assert (
        supply.getDeclaration()
        == 'Equation supply(i) "observe supply limit at plant i";'
    )

    # Equation declaration with more than one index
    bla = Equation(m, name="bla", domain=[i, j], description="some text")
    assert bla.gamsRepr() == "bla"
    assert bla.getDeclaration() == 'Equation bla(i,j) "some text";'

    u = Set(m, "u")
    v = Alias(m, "v", alias_with=u)
    e = Set(m, "e", domain=[u, v])
    eq = Equation(m, "eq", domain=[u, v])
    assert eq[e[u, v]].gamsRepr() == "eq(e(u,v))"


def test_equation_definition(data):
    m, canning_plants, capacities, _, distances = data
    i = Set(m, name="i", records=canning_plants, description="Canning Plants")
    j = Set(
        m,
        name="j",
        records=["new-york", "chicago", "topeka"],
        description="Markets",
    )

    # Params
    a = Parameter(m, name="a", domain=[i], records=capacities)
    d = Parameter(m, name="d", domain=[i, j], records=distances)
    c = Parameter(m, name="c", domain=[i, j])
    c[i, j] = 90 * d[i, j] / 1000

    x = Variable(m, name="x", domain=[i, j], type="Positive")
    z = Variable(m, name="z")

    # Equation definition without an index
    cost = Equation(m, name="cost", description="define objective function")
    cost[...] = Sum((i, j), c[i, j] * x[i, j]) == z
    with pytest.raises(TypeError):
        cost.records = 5

    assert cost[...]
    assert (
        cost.getDefinition() == "cost .. sum((i,j),(c(i,j) * x(i,j))) =e= z;"
    )

    # Equation definition with an index
    supply = Equation(
        m,
        name="supply",
        domain=[i],
        description="observe supply limit at plant i",
        definition=Sum(j, x[i, j]) <= a[i],
    )
    assert supply.getDefinition() == "supply(i) .. sum(j,x(i,j)) =l= a(i);"

    # Equation definition with more than one index
    bla = Equation(
        m,
        name="bla",
        domain=[i, j],
        description="observe supply limit at plant i",
    )
    bla[i, j] = x[i, j] <= a[i]
    assert bla.getDefinition() == "bla(i,j) .. x(i,j) =l= a(i);"

    bla[i, "topeka"] = x[i, "topeka"] <= a[i]
    assert bla.getDefinition() == 'bla(i,"topeka") .. x(i,"topeka") =l= a(i);'

    # Equation definition in constructor
    cost2 = Equation(
        m,
        name="cost2",
        description="define objective function",
        definition=Sum((i, j), c[i, j] * x[i, j]) == z,
    )
    assert (
        cost2.getDefinition() == "cost2 .. sum((i,j),(c(i,j) * x(i,j))) =e= z;"
    )

    # Equation definition in addEquation
    cost3 = m.addEquation(
        name="cost3",
        description="define objective function",
        definition=Sum((i, j), c[i, j] * x[i, j]) == z,
    )
    assert (
        cost3.getDefinition() == "cost3 .. sum((i,j),(c(i,j) * x(i,j))) =e= z;"
    )

    # eq[bla][...] test
    bla2 = Equation(
        m,
        name="bla2",
        domain=[i, j],
        description="observe supply limit at plant i",
    )
    bla2[i, j] = x[i, j] <= a[i]
    assert bla2.getDefinition() == "bla2(i,j) .. x(i,j) =l= a(i);"

    # eq[bla] with different domain
    with pytest.raises(ValidationError):
        _ = Equation(
            m,
            name="bla3",
            domain=[i, j],
            description="observe supply limit at plant i",
            definition=x[i, j] <= a[i],
            definition_domain=[i, "bla"],
        )

    m = Container()
    g = Set(m, name="g", records=[str(i) for i in range(1, 4)])
    t1 = Set(m, name="t1", records=[str(i) for i in range(1, 4)])
    t2 = Set(m, name="t2", records=[str(i) for i in range(1, 4)])

    eStartNaive = Equation(m, name="eStartNaive", domain=[g, t1])
    pMinDown = Parameter(m, name="pMinDown", domain=[g, t1])
    vStart = Parameter(m, name="vStart", domain=[g, t2])

    eStartNaive[g, t1] = (
        Sum(
            t2.where[
                (Ord(t1) >= Ord(t2)) & (Ord(t2) > Ord(t1) - pMinDown[g, t1])
            ],
            vStart[g, t2],
        )
        <= 1
    )
    assert (
        eStartNaive.getDefinition()
        == "eStartNaive(g,t1) .. sum(t2 $ ((ord(t1) >= ord(t2)) and"
        " (ord(t2) > (ord(t1) - pMinDown(g,t1)))),vStart(g,t2)) =l= 1;"
    )

    m = Container()
    i = Set(m, "i")
    j = Set(m, "j")

    a = Parameter(m, name="a", domain=[i, j])
    b = Parameter(m, name="b", domain=[i, j])
    c = Variable(m, name="c", domain=[i, j])
    assign_1 = Equation(m, name="assign_1", domain=[i, j])
    assign_1[...] = a[i, j] == b[i, j] + c[i, j]
    assert (
        assign_1.getDefinition()
        == "assign_1(i,j) .. a(i,j) =e= (b(i,j) + c(i,j));"
    )

    m = Container()
    k = Set(m, "k")

    a = Parameter(m, name="a")
    b = Variable(m, name="b", domain=[k])
    c = Parameter(m, name="c", domain=[k])
    assign_1 = Equation(m, name="assign_1")

    assign_1[...] = a == Sum(k, b[k] * c[k])
    assert (
        assign_1.getDefinition() == "assign_1 .. a =e= sum(k,(b(k) * c(k)));"
    )

    i = Set(m, "i", records=["OJ", "1M"])
    x = Variable(m, "x", domain=i)
    drinking_eqn = m.addEquation(
        "drinking_eqn",
        description="Don't drink too much",
        definition=x["OJ"] + x["1M"] <= 3,
    )
    assert (
        drinking_eqn.getDefinition()
        == 'drinking_eqn .. (x("OJ") + x("1M")) =l= 3;'
    )

    s = m.addVariable(
        "s", "free", description="Surplus variable for drinking_eqn"
    )
    drinking_eqn = m.addEquation(
        "drinking_eqn",
        description="Don't drink too much",
        definition=x["OJ"] + x["1M"] <= 3 + s,
    )
    assert (
        drinking_eqn.getDefinition()
        == 'drinking_eqn .. (x("OJ") + x("1M")) =l= (3 + s);'
    )


def test_equation_attributes(data):
    m, *_ = data
    pi = Equation(m, "pi")

    assert hasattr(pi, "l") and isinstance(pi.l, implicits.ImplicitParameter)
    assert pi.l.gamsRepr() == "pi.l"

    assert hasattr(pi, "m") and isinstance(pi.m, implicits.ImplicitParameter)
    assert pi.m.gamsRepr() == "pi.m"

    assert hasattr(pi, "lo") and isinstance(pi.lo, implicits.ImplicitParameter)
    assert pi.lo.gamsRepr() == "pi.lo"

    assert hasattr(pi, "up") and isinstance(pi.up, implicits.ImplicitParameter)
    assert pi.up.gamsRepr() == "pi.up"

    assert hasattr(pi, "scale") and isinstance(
        pi.scale, implicits.ImplicitParameter
    )
    assert pi.scale.gamsRepr() == "pi.scale"

    assert hasattr(pi, "stage") and isinstance(
        pi.stage, implicits.ImplicitParameter
    )
    assert pi.stage.gamsRepr() == "pi.stage"

    assert hasattr(pi, "range") and isinstance(
        pi.range, implicits.ImplicitParameter
    )
    assert pi.range.gamsRepr() == "pi.range"

    assert hasattr(pi, "slacklo") and isinstance(
        pi.slacklo, implicits.ImplicitParameter
    )
    assert pi.slacklo.gamsRepr() == "pi.slacklo"

    assert hasattr(pi, "slackup") and isinstance(
        pi.slackup, implicits.ImplicitParameter
    )
    assert pi.slackup.gamsRepr() == "pi.slackup"

    assert hasattr(pi, "slack") and isinstance(
        pi.slack, implicits.ImplicitParameter
    )
    assert pi.slack.gamsRepr() == "pi.slack"

    assert hasattr(pi, "infeas") and isinstance(
        pi.infeas, implicits.ImplicitParameter
    )
    assert pi.infeas.gamsRepr() == "pi.infeas"


def test_scalar_attr_assignment(data):
    m, *_ = data
    a = Equation(m, "a")
    a.l = 5
    assert a._assignment.getDeclaration() == "a.l = 5;"

    a.m = 5
    assert a._assignment.getDeclaration() == "a.m = 5;"

    a.lo = 5
    assert a._assignment.getDeclaration() == "a.lo = 5;"

    a.up = 5
    assert a._assignment.getDeclaration() == "a.up = 5;"

    a.scale = 5
    assert a._assignment.getDeclaration() == "a.scale = 5;"

    a.stage = 5
    assert a._assignment.getDeclaration() == "a.stage = 5;"


def test_mcp_equation(data):
    m, *_ = data
    c = Parameter(m, name="c", domain=[], records=0.5)
    x = Variable(
        m,
        name="x",
        domain=[],
        records={"lower": 1.0, "level": 1.5, "upper": 3.75},
    )
    f = Equation(m, name="f", type="nonbinding")
    f[...] = (x - c) == 0

    assert f.getDefinition() == "f .. (x - c) =n= 0;"

    f2 = Equation(m, name="f2", type="nonbinding", definition=(x - c) == 0)
    assert f2.getDefinition() == "f2 .. (x - c) =n= 0;"

    f3 = Equation(m, name="f3", type="nonbinding", definition=(x - c) == 0)
    assert f3.getDefinition() == "f3 .. (x - c) =n= 0;"

    f4 = Equation(m, name="f4", definition=(x - c) == 0)
    assert f4.getDefinition() == "f4 .. (x - c) =e= 0;"

    model = Model(m, "mcp_model", "MCP", matches={f: x})
    model.solve()


def test_changed_domain(data):
    m, *_ = data
    cont = Container()

    s = Set(cont, "s")
    m = Set(cont, "m")
    A = Equation(cont, "A", domain=[s, m])

    A.domain = ["s", "m"]
    assert A.getDeclaration() == "Equation A(*,*);"


def test_equation_assignment(data):
    m, *_ = data
    m = Container()

    i = Set(m, "i")
    j = Set(m, "j")
    a = Equation(m, "a", domain=[i])

    with pytest.raises(ValidationError):
        a[j] = 5

    m = Container()
    N = Parameter(m, "N", records=20)
    L = Parameter(m, "L", records=int(N.toValue()) / 2)
    v = Set(m, "v", records=range(0, 1001))
    i = Set(m, "i", domain=[v])
    x = Variable(m, "x", "free", [v])
    y = Variable(m, "y", "free", [v])
    e = Equation(m, "e")
    e[...] = Sum(i.where[(i.val == L - 1)], sqr(x[i]) + sqr(y[i])) == 1
    assert (
        e.getDefinition()
        == "e .. sum(i $ ((L - 1) eq i.val),(( sqr(x(i)) ) + ("
        " sqr(y(i)) ))) =e= 1;"
    )


def test_assignment_dimensionality(data):
    m, *_ = data
    j1 = Set(m, "j1")
    j2 = Set(m, "j2")
    j3 = Equation(m, "j3", domain=[j1, j2])
    with pytest.raises(ValidationError):
        j3["bla"] = 5

    j4 = Set(m, "j4")

    with pytest.raises(ValidationError):
        j3[j1, j2, j4] = 5

    i = Set(m, name="i")
    ii = Set(m, name="ii", domain=[i])
    j = Set(m, name="j")
    jj = Set(m, name="jj", domain=[j])
    k = Set(m, name="k")
    kk = Set(m, name="kk", domain=[k])
    TSAM = Variable(m, name="TSAM", domain=[i, j])
    A = Variable(m, name="A", domain=[i, j])
    Y = Variable(m, name="Y", domain=[i, j])
    NONZERO = Set(m, name="NONZERO")
    SAMCOEF = Equation(
        m, name="SAMCOEF", domain=[i, j], description="define SAM coefficients"
    )

    with pytest.raises(ValidationError):
        SAMCOEF[ii, jj, kk].where[NONZERO[ii, jj]] = (
            TSAM[ii, jj] == A[ii, jj] * Y[jj]
        )


def test_type(data):
    m, *_ = data
    eq1 = Equation(m, "eq1")
    eq1.type = EquationType.REGULAR
    assert eq1.type == "eq"

    eq2 = Equation(m, "eq2")
    eq2.type = EquationType.BOOLEAN
    assert eq2.type == "boolean"

    eq3 = Equation(m, "eq3")
    eq3.type = EquationType.NONBINDING
    assert eq3.type == "nonbinding"


def test_uels_on_axes(data):
    m, *_ = data
    s = pd.Series(index=["a", "b", "c"], data=[i + 1 for i in range(3)])
    e = Equation(m, "e", "eq", domain=["*"], records=s, uels_on_axes=True)
    assert e.records.level.tolist() == [1, 2, 3]


def test_expert_sync(data):
    m, *_ = data
    m = Container()
    i = Set(m, "i", records=["i1", "i2"])
    e = Equation(m, "e", domain=i)
    e.l = 5
    e.synchronize = False
    e.l = e.l * 5
    assert e.records.level.tolist() == [5.0, 5.0]
    e.synchronize = True
    assert e.records.level.tolist() == [25.0, 25.0]


def test_equation_listing(data):
    m, *_ = data
    m = Container()

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

    mv_data = pd.DataFrame(
        [
            ["1", 60],
            ["2", 50],
            ["3", 50],
            ["4", 75],
            ["5", 40],
            ["6", 60],
            ["7", 35],
            ["8", 30],
            ["9", 25],
            ["10", 150],
            ["11", 30],
            ["12", 45],
            ["13", 125],
            ["14", 200],
            ["15", 200],
            ["16", 130],
            ["17", 100],
            ["18", 100],
            ["19", 100],
            ["20", 150],
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
    td = Parameter(
        m, name="td", domain=[w, t], records=td_data, description="target data"
    )
    wa = Parameter(
        m,
        name="wa",
        domain=w,
        records=wa_data,
        description="weapons availability",
    )
    tm = Parameter(
        m,
        name="tm",
        domain=t,
        records=tm_data,
        description="minimum number of weapons per target",
    )
    mv = Parameter(
        m,
        name="mv",
        domain=t,
        records=mv_data,
        description="military value of target",
    )

    # Variables
    x = Variable(
        m,
        name="x",
        domain=[w, t],
        type="Positive",
        description="weapons assignment",
    )
    prob = Variable(
        m, name="prob", domain=t, description="probability for each target"
    )

    # Equations
    maxw = Equation(m, name="maxw", domain=w, description="weapons balance")
    minw = Equation(
        m,
        name="minw",
        domain=t,
        description="minimum number of weapons required per target",
    )
    probe = Equation(
        m, name="probe", domain=t, description="probability definition"
    )

    maxw[w] = Sum(t.where[td[w, t]], x[w, t]) <= wa[w]
    minw[t].where[tm[t]] = Sum(w.where[td[w, t]], x[w, t]) >= tm[t]
    probe[t] = prob[t] == 1 - Product(
        w.where[td[w, t]], (1 - td[w, t]) ** x[w, t]
    )

    _ = Sum(t, mv[t] * prob[t])
    etd = Sum(
        t,
        mv[t] * (1 - Product(w.where[td[w, t]], (1 - td[w, t]) ** x[w, t])),
    )

    war = Model(
        m,
        name="war",
        equations=[maxw, minw],
        problem=Problem.NLP,
        sense=Sense.MAX,
        objective=etd,
    )

    x.l[w, t].where[td[w, t]] = wa[w] / Card(t)

    war.solve()
    with pytest.raises(ValidationError):
        _ = maxw.getEquationListing()

    war.solve(options=Options(equation_listing_limit=10))

    assert len(maxw.getEquationListing().split("\n")) == 5

    with pytest.raises(ValidationError):
        maxw.getEquationListing(filters=[["f-bomber"], ["bla"]])

    assert (
        len(maxw.getEquationListing(filters=[["f-bomber"]]).split("\n")) == 1
    )

    assert len(maxw.getEquationListing(n=2).split("\n")) == 2


def test_equation_listing2(data):
    m, *_ = data
    cont = Container()

    # Prepare data
    steel_plants = ["ahmsa", "fundidora", "sicartsa", "hylsa", "hylsap"]
    markets = ["mexico-df", "monterrey", "guadalaja"]
    commodities = [
        "pellets",
        "coke",
        "nat-gas",
        "electric",
        "scrap",
        "pig-iron",
        "sponge",
        "steel",
    ]
    final_products = ["steel"]
    intermediate_products = ["sponge", "pig-iron"]
    raw_materials = ["pellets", "coke", "nat-gas", "electric", "scrap"]
    processes = ["pig-iron", "sponge", "steel-oh", "steel-el", "steel-bof"]
    productive_units = [
        "blast-furn",
        "openhearth",
        "bof",
        "direct-red",
        "elec-arc",
    ]

    io_coefficients = pd.DataFrame(
        [
            ["pellets", "pig-iron", -1.58],
            ["pellets", "sponge", -1.38],
            ["coke", "pig-iron", -0.63],
            ["nat-gas", "sponge", -0.57],
            ["electric", "steel-el", -0.58],
            ["scrap", "steel-oh", -0.33],
            ["scrap", "steel-bof", -0.12],
            ["pig-iron", "pig-iron", 1.00],
            ["pig-iron", "steel-oh", -0.77],
            ["pig-iron", "steel-bof", -0.95],
            ["sponge", "sponge", 1.00],
            ["sponge", "steel-el", -1.09],
            ["steel", "steel-oh", 1.00],
            ["steel", "steel-el", 1.00],
            ["steel", "steel-bof", 1.00],
        ]
    )

    capacity_utilization = pd.DataFrame(
        [
            ["blast-furn", "pig-iron", 1.0],
            ["openhearth", "steel-oh", 1.0],
            ["bof", "steel-bof", 1.0],
            ["direct-red", "sponge", 1.0],
            ["elec-arc", "steel-el", 1.0],
        ]
    )

    capacities_of_units = pd.DataFrame(
        [
            ["blast-furn", "ahmsa", 3.25],
            ["blast-furn", "fundidora", 1.40],
            ["blast-furn", "sicartsa", 1.10],
            ["openhearth", "ahmsa", 1.50],
            ["openhearth", "fundidora", 0.85],
            ["bof", "ahmsa", 2.07],
            ["bof", "fundidora", 1.50],
            ["bof", "sicartsa", 1.30],
            ["direct-red", "hylsa", 0.98],
            ["direct-red", "hylsap", 1.00],
            ["elec-arc", "hylsa", 1.13],
            ["elec-arc", "hylsap", 0.56],
        ]
    )

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

    product_prices = pd.DataFrame(
        [
            ["pellets", "domestic", 18.7],
            ["coke", "domestic", 52.17],
            ["nat-gas", "domestic", 14.0],
            ["electric", "domestic", 24.0],
            ["scrap", "domestic", 105.0],
            ["steel", "import", 150],
            ["steel", "export", 140],
        ]
    )

    demand_distribution = pd.DataFrame(
        [["mexico-df", 55], ["monterrey", 30], ["guadalaja", 15]]
    )

    dt = 5.209  # total demand for final goods in 1979
    rse = 40  # raw steel equivalence
    eb = 1.0  # export bound

    # Set
    i = Set(
        cont,
        name="i",
        records=pd.DataFrame(steel_plants),
        description="steel plants",
    )
    j = Set(
        cont, name="j", records=pd.DataFrame(markets), description="markets"
    )
    c = Set(
        cont,
        name="c",
        records=pd.DataFrame(commodities),
        description="commidities",
    )
    cf = Set(
        cont,
        name="cf",
        records=pd.DataFrame(final_products),
        domain=c,
        description="final products",
    )
    ci = Set(
        cont,
        name="ci",
        records=pd.DataFrame(intermediate_products),
        domain=c,
        description="intermediate products",
    )
    cr = Set(
        cont,
        name="cr",
        records=pd.DataFrame(raw_materials),
        domain=c,
        description="raw materials",
    )
    p = Set(
        cont,
        name="p",
        records=pd.DataFrame(processes),
        description="processes",
    )
    m = Set(
        cont,
        name="m",
        records=pd.DataFrame(productive_units),
        description="productive units",
    )

    # Data
    a = Parameter(
        cont,
        name="a",
        domain=[c, p],
        records=io_coefficients,
        description="input-output coefficients",
    )
    b = Parameter(
        cont,
        name="b",
        domain=[m, p],
        records=capacity_utilization,
        description="capacity utilization",
    )
    k = Parameter(
        cont,
        name="k",
        domain=[m, i],
        records=capacities_of_units,
        description="capacities of productive units",
    )
    dd = Parameter(
        cont,
        name="dd",
        domain=j,
        records=demand_distribution,
        description="distribution of demand",
    )
    d = Parameter(
        cont, name="d", domain=[c, j], description="demand for steel in 1979"
    )

    d["steel", j] = dt * (1 + rse / 100) * dd[j] / 100

    rd = Parameter(
        cont,
        name="rd",
        domain=["*", "*"],
        records=rail_distances,
        description="rail distances from plants to markets",
    )

    muf = Parameter(
        cont,
        name="muf",
        domain=[i, j],
        description="transport rate: final products",
    )
    muv = Parameter(
        cont, name="muv", domain=j, description="transport rate: imports"
    )
    mue = Parameter(
        cont, name="mue", domain=i, description="transport rate: exports"
    )

    muf[i, j] = (2.48 + 0.0084 * rd[i, j]).where[rd[i, j]]
    muv[j] = (2.48 + 0.0084 * rd["import", j]).where[rd["import", j]]
    mue[i] = (2.48 + 0.0084 * rd[i, "export"]).where[rd[i, "export"]]

    prices = Parameter(
        cont,
        name="prices",
        domain=[c, "*"],
        records=product_prices,
        description="product prices (us$ per unit)",
    )

    pdp = Parameter(cont, name="pd", domain=c, description="domestic prices")
    pv = Parameter(cont, name="pv", domain=c, description="import prices")
    pe = Parameter(cont, name="pe", domain=c, description="export prices")

    pdp[c] = prices[c, "domestic"]
    pv[c] = prices[c, "import"]
    pe[c] = prices[c, "export"]

    # Variable
    z = Variable(
        cont,
        name="z",
        domain=[p, i],
        type="Positive",
        description="process level",
    )
    x = Variable(
        cont,
        name="x",
        domain=[c, i, j],
        type="Positive",
        description="shipment of final products",
    )
    u = Variable(
        cont,
        name="u",
        domain=[c, i],
        type="Positive",
        description="purchase of domestic materials",
    )
    v = Variable(
        cont, name="v", domain=[c, j], type="Positive", description="imports"
    )
    e = Variable(
        cont, name="e", domain=[c, i], type="Positive", description="exports"
    )
    phipsi = Variable(cont, name="phipsi", description="raw material cost")
    philam = Variable(cont, name="philam", description="transport cost")
    phipi = Variable(cont, name="phipi", description="import cost")
    phieps = Variable(cont, name="phieps", description="export revenue")

    # Equation declaration
    mbf = Equation(
        cont,
        name="mbf",
        domain=[c, i],
        description="material balances: final products",
    )
    with pytest.raises(ValidationError):
        mbf.latexRepr()

    mbi = Equation(
        cont,
        name="mbi",
        domain=[c, i],
        description="material balances: intermediates",
    )
    mbr = Equation(
        cont,
        name="mbr",
        domain=[c, i],
        description="material balances: raw materials",
    )
    cc = Equation(
        cont, name="cc", domain=[m, i], description="capacity constraint"
    )
    mr = Equation(
        cont, name="mr", domain=[c, j], description="market requirements"
    )
    me = Equation(cont, name="me", domain=c, description="maximum export")
    apsi = Equation(
        cont, name="apsi", description="accounting: raw material cost"
    )
    alam = Equation(
        cont, name="alam", description="accounting: transport cost"
    )
    api = Equation(cont, name="api", description="accounting: import cost")
    aeps = Equation(cont, name="aeps", description="accounting: export cost")

    # Equation definition
    obj = phipsi + philam + phipi - phieps  # Total Cost

    mbf[cf, i] = Sum(p, a[cf, p] * z[p, i]) >= Sum(j, x[cf, i, j]) + e[cf, i]
    mbi[ci, i] = Sum(p, a[ci, p] * z[p, i]) >= 0
    mbr[cr, i] = Sum(p, a[cr, p] * z[p, i]) + u[cr, i] >= 0
    cc[m, i] = Sum(p, b[m, p] * z[p, i]) <= k[m, i]
    mr[cf, j] = Sum(i, x[cf, i, j]) + v[cf, j] >= d[cf, j]
    me[cf] = Sum(i, e[cf, i]) <= eb
    apsi[...] = phipsi == Sum((cr, i), pdp[cr] * u[cr, i])
    alam[...] = philam == Sum((cf, i, j), muf[i, j] * x[cf, i, j]) + Sum(
        (cf, j), muv[j] * v[cf, j]
    ) + Sum((cf, i), mue[i] * e[cf, i])
    api[...] = phipi == Sum((cf, j), pv[cf] * v[cf, j])
    aeps[...] = phieps == Sum((cf, i), pe[cf] * e[cf, i])

    mexss = Model(
        cont,
        name="mexss",
        equations=cont.getEquations(),
        problem="LP",
        sense=Sense.MIN,
        objective=obj,
    )

    mexss.solve(options=Options(equation_listing_limit=100))
    assert (
        len(
            mr.getEquationListing(filters=[["steel"], ["monterrey"]]).split(
                "\n"
            )
        )
        == 1
    )

    assert len(mr.getEquationListing(filters=[["steel"], []]).split("\n")) == 3


def test_alternative_syntax():
    m = Container()
    i = Set(m, "i", records=["i1", "i2"])
    e = Equation(m, "e", domain=i)

    assert e[i].l.gamsRepr() == "e.l(i)"
    assert e[i].m.gamsRepr() == "e.m(i)"
    assert e[i].lo.gamsRepr() == "e.lo(i)"
    assert e[i].up.gamsRepr() == "e.up(i)"
    assert e[i].scale.gamsRepr() == "e.scale(i)"
    assert e[i].stage.gamsRepr() == "e.stage(i)"
    assert e[i].range.gamsRepr() == "e.range(i)"
    assert e[i].slacklo.gamsRepr() == "e.slacklo(i)"
    assert e[i].slackup.gamsRepr() == "e.slackup(i)"
    assert e[i].slack.gamsRepr() == "e.slack(i)"

    e[i].l = 5
    assert e.getAssignment() == "e.l(i) = 5;"
    e[i].m = 5
    assert e.getAssignment() == "e.m(i) = 5;"
    e[i].lo = 5
    assert e.getAssignment() == "e.lo(i) = 5;"
    e[i].up = 5
    assert e.getAssignment() == "e.up(i) = 5;"
    e[i].scale = 5
    assert e.getAssignment() == "e.scale(i) = 5;"
    e[i].stage = 5
    assert e.getAssignment() == "e.stage(i) = 5;"
