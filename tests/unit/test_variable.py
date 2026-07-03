from __future__ import annotations

import platform
import sys

import numpy as np
import pandas as pd
import pytest

import gamspy as gp
import gamspy._symbols.implicits as implicits
from gamspy import (
    Container,
    Equation,
    Model,
    Options,
    Parameter,
    Product,
    Sand,
    Sense,
    Set,
    Smax,
    Smin,
    Sor,
    SpecialValues,
    Sum,
    Variable,
    VariableType,
)
from gamspy.exceptions import ValidationError

pytestmark = pytest.mark.unit


@pytest.fixture
def data():
    m = Container()
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

    yield m, capacities, demands, distances
    m.close()


def test_variable_creation(data):
    m, *_ = data
    # no name is fine now
    v = Variable(m)
    m.addVariable()
    with pytest.raises(ValidationError):
        _ = v.getAssignment()

    # non-str type name
    with pytest.raises(TypeError):
        Variable(m, 5)

    # no container
    with pytest.raises(ValidationError):
        Variable()

    # non-container type container
    with pytest.raises(TypeError):
        Variable(5, "j")

    # try to create a symbol with same name but different type
    _ = Set(m, "i")
    with pytest.raises(TypeError):
        Variable(m, "i")

    # get already created symbol
    j1 = Variable(m, "j")
    j2 = Variable(m, "j")
    assert id(j1) == id(j2)

    # Variable and domain containers are different
    m2 = Container()

    # Same name, different domain forwarding
    _ = Variable(m, "k")
    with pytest.raises(ValueError):
        Variable(m, "k", "free", None, None, True)

    set1 = Set(m, "set1")
    with pytest.raises(ValidationError):
        _ = Variable(m2, "var1", domain=[set1])

    # Incorrect type
    with pytest.raises(ValueError):
        _ = Variable(m2, type="Blabla")


def test_variable_string(data):
    m, *_ = data
    # Check if the name is reserved
    with pytest.raises(ValidationError):
        Variable(m, "set")

    i = Set(m, name="i", records=["bla", "damn"])
    j = Set(m, name="j", records=["test", "test2"])

    # Variable without data
    v4 = Variable(m, "v4")
    assert v4.gamsRepr() == "v4"
    assert v4.getDeclaration() == "free Variable v4 / /;"

    with pytest.raises(TypeError):
        v4.records = 5

    # Variable without domain
    v0 = Variable(m, name="v0", description="some text")
    assert v0.gamsRepr() == "v0"
    assert v0.getDeclaration() == 'free Variable v0 "some text" / /;'

    expression = -v0
    assert expression.getDeclaration() == "(-v0)"

    # Variable one domain
    v1 = Variable(m, name="v1", domain=[i])
    assert v1.gamsRepr() == "v1(i)"
    assert v1.getDeclaration() == "free Variable v1(i) / /;"

    assert (v1[i] == v1[i]).gamsRepr() == "v1(i) =e= v1(i)"

    # Variable two domain
    v2 = Variable(m, name="v2", domain=[i, j])
    assert v2.gamsRepr() == "v2(i,j)"
    assert v2.getDeclaration() == "free Variable v2(i,j) / /;"

    # Scalar variable with records
    pi = Variable(
        m,
        "pi",
        records=pd.DataFrame(data=[3.14159], columns=["level"]),
    )
    assert pi.getDeclaration() == "free Variable pi;"
    assert (-pi).gamsRepr() == "(-pi)"
    assert (pi != 3).gamsRepr() == "pi ne 3"

    # 1D variable with records
    v = Variable(
        m,
        "v",
        "free",
        domain=["*"],
        records=pd.DataFrame(
            data=[("i" + str(i), i) for i in range(5)],
            columns=["domain", "marginal"],
        ),
    )
    assert v.getDeclaration() == "free Variable v(*);"

    v3 = Variable(
        m,
        "v3",
        "positive",
        ["*", "*"],
        records=pd.DataFrame([("seattle", "san-diego"), ("chicago", "madison")]),
    )
    assert v3.getDeclaration() == "positive Variable v3(*,*);"


def test_variable_types(data):
    m, *_ = data
    i = Set(m, "i", records=["1", "2"])

    v = Variable(m, name="v", type="Positive")
    assert v.getDeclaration() == "positive Variable v / /;"

    v1 = Variable(m, name="v1", type="Negative")
    assert v1.getDeclaration() == "negative Variable v1 / /;"

    v2 = Variable(m, name="v2", type="Binary")
    assert v2.getDeclaration() == "binary Variable v2 / /;"

    v3 = Variable(m, name="v3", domain=[i], type="Integer")
    assert v3.getDeclaration() == "integer Variable v3(i) / /;"

    assert str(VariableType.FREE) == "free"
    assert VariableType.values() == [
        "binary",
        "integer",
        "positive",
        "negative",
        "free",
        "sos1",
        "sos2",
        "semicont",
        "semiint",
    ]

    v4 = Variable(m, name="v4", domain=[i], type=VariableType.POSITIVE)
    assert v4.type == "positive"


def test_variable_attributes(data):
    m, *_ = data
    m = Container()
    pi = Variable(
        m,
        "pi",
        records=pd.DataFrame(data=[3.14159], columns=["level"]),
    )

    assert hasattr(pi, "l") and isinstance(pi.l, implicits.ImplicitParameter)
    assert pi.l.gamsRepr() == "pi.l"

    assert hasattr(pi, "m") and isinstance(pi.m, implicits.ImplicitParameter)
    assert pi.m.gamsRepr() == "pi.m"

    assert hasattr(pi, "lo") and isinstance(pi.lo, implicits.ImplicitParameter)
    assert pi.lo.gamsRepr() == "pi.lo"

    assert hasattr(pi, "up") and isinstance(pi.up, implicits.ImplicitParameter)
    assert pi.up.gamsRepr() == "pi.up"

    assert hasattr(pi, "scale") and isinstance(pi.scale, implicits.ImplicitParameter)
    assert pi.scale.gamsRepr() == "pi.scale"

    assert hasattr(pi, "fx") and isinstance(pi.fx, implicits.ImplicitParameter)
    assert pi.fx.gamsRepr() == "pi.fx"

    assert hasattr(pi, "prior") and isinstance(pi.prior, implicits.ImplicitParameter)
    assert pi.prior.gamsRepr() == "pi.prior"

    assert hasattr(pi, "stage") and isinstance(pi.stage, implicits.ImplicitParameter)
    assert pi.stage.gamsRepr() == "pi.stage"

    i = Set(m, name="i", records=["bla", "damn"])
    test = Variable(m, "test", domain=[i])
    assert hasattr(test, "l") and isinstance(test.l, implicits.ImplicitParameter)
    assert hasattr(test, "m") and isinstance(test.m, implicits.ImplicitParameter)
    assert hasattr(test, "lo") and isinstance(test.lo, implicits.ImplicitParameter)
    assert hasattr(test, "up") and isinstance(test.up, implicits.ImplicitParameter)
    assert hasattr(test, "scale") and isinstance(
        test.scale, implicits.ImplicitParameter
    )
    assert hasattr(test, "fx") and isinstance(test.fx, implicits.ImplicitParameter)
    assert hasattr(test, "prior") and isinstance(
        test.prior, implicits.ImplicitParameter
    )
    assert hasattr(test, "stage") and isinstance(
        test.stage, implicits.ImplicitParameter
    )

    k = Set(m, "k")
    x = Variable(m, "x", domain=[k])
    x.l[k] = 5

    assert x.getAssignment() == "x.l(k) = 5;"


def test_scalar_attr_assignment(data):
    m, *_ = data
    a = Variable(m, "a")
    b = Variable(m, "b", "binary")
    a.l = 5
    assert a.getAssignment() == "a.l = 5;"

    a.m = 5
    assert a.getAssignment() == "a.m = 5;"

    a.lo = 5
    assert a.getAssignment() == "a.lo = 5;"

    a.up = 5
    assert a.getAssignment() == "a.up = 5;"

    with pytest.raises(ValidationError):
        b.scale = 5

    a.scale = 5
    assert a.getAssignment() == "a.scale = 5;"

    a.fx = 5
    assert a.getAssignment() == "a.fx = 5;"

    with pytest.raises(ValidationError):
        a.prior = 5

    b.prior = 5
    assert b.getAssignment() == "b.prior = 5;"

    a.stage = 5
    assert a.getAssignment() == "a.stage = 5;"


def test_implicit_variable(data):
    m, *_ = data
    i = Set(m, "i", records=[f"i{i}" for i in range(10)])
    a = Variable(m, "a", "free", [i])
    a.generateRecords()

    expression = -a[i] * 5
    assert expression.gamsRepr() == "(-a(i)) * 5"

    a.l[...] = 5

    assert a.records.level.to_list() == [
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
    ]


def test_assignment_dimensionality(data):
    m, *_ = data
    j1 = Set(m, "j1")
    j2 = Set(m, "j2")
    j3 = Variable(m, "j3", domain=[j1, j2])
    j4 = Set(m, "j4")

    e1 = Equation(m, "e1", domain=[j1, j2])

    with pytest.raises(ValidationError):
        e1[j1, j2] = j3[j1, j2, j4] * 5 <= 5


def test_type(data):
    m, *_ = data
    gamma = Variable(m, "gamma")
    gamma.type = VariableType.BINARY
    assert gamma.type == "binary"

    var1 = Variable(m, "var1")
    var1.type = VariableType.FREE
    assert var1.type == "free"

    var2 = Variable(m, "var2")
    var2.type = VariableType.POSITIVE
    assert var2.type == "positive"

    var3 = Variable(m, "var3")
    var3.type = VariableType.NEGATIVE
    assert var3.type == "negative"

    var4 = Variable(m, "var4")
    var4.type = VariableType.NEGATIVE
    assert var4.type == "negative"

    var5 = Variable(m, "var5")
    var5.type = VariableType.SEMICONT
    assert var5.type == "semicont"


def test_uels_on_axes(data):
    m, *_ = data
    s = pd.Series(index=["a", "b", "c"], data=[i + 1 for i in range(3)])
    v = Variable(m, "v", domain=["*"], records=s, uels_on_axes=True)
    assert v.records.level.tolist() == [1, 2, 3]


def test_variable_listing(data):
    m, capacities, demands, distances = data
    m = Container()

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

    a = Parameter(
        m,
        name="a",
        domain=i,
        records=capacities,
        description="capacity of plant i in cases",
    )
    b = Parameter(
        m,
        name="b",
        domain=j,
        records=demands,
        description="demand at market j in cases",
    )
    d = Parameter(
        m,
        name="d",
        domain=[i, j],
        records=distances,
        description="distance in thousands of miles",
    )
    c = Parameter(
        m,
        name="c",
        domain=[i, j],
        description="transport cost in thousands of dollars per case",
    )
    c[i, j] = 90 * d[i, j] / 1000

    # Variable
    x = Variable(
        m,
        name="x",
        domain=[i, j],
        type="Positive",
        description="shipment quantities in cases",
    )
    with pytest.raises(ValidationError):
        _ = x.getVariableListing()

    # Equation
    supply = Equation(
        m,
        name="supply",
        domain=i,
        description="observe supply limit at plant i",
    )
    demand = Equation(
        m, name="demand", domain=j, description="satisfy demand at market j"
    )

    supply[i] = Sum(j, x[i, j]) <= a[i]
    demand[j] = Sum(i, x[i, j]) >= b[j]

    transport = Model(
        m,
        name="transport",
        equations=m.getEquations(),
        problem="LP",
        sense=Sense.MIN,
        objective=Sum((i, j), c[i, j] * x[i, j]),
    )

    transport.solve(
        options=Options(equation_listing_limit=100, variable_listing_limit=100)
    )
    assert len(transport.getVariableListing().split("\n\n")) == 7
    assert transport.getVariableListing() == "\n".join(
        [
            "x(seattle,new-york)\n                (.LO, .L, .UP, .M = 0, 0, +INF, 0)\n        1       supply(seattle)\n        1       demand(new-york)\n        0.225   transport_objective\n",
            "x(seattle,chicago)\n                (.LO, .L, .UP, .M = 0, 0, +INF, 0)\n        1       supply(seattle)\n        1       demand(chicago)\n        0.153   transport_objective\n",
            "x(seattle,topeka)\n                (.LO, .L, .UP, .M = 0, 0, +INF, 0)\n        1       supply(seattle)\n        1       demand(topeka)\n        0.162   transport_objective\n",
            "x(san-diego,new-york)\n                (.LO, .L, .UP, .M = 0, 0, +INF, 0)\n        1       supply(san-diego)\n        1       demand(new-york)\n        0.225   transport_objective\n",
            "x(san-diego,chicago)\n                (.LO, .L, .UP, .M = 0, 0, +INF, 0)\n        1       supply(san-diego)\n        1       demand(chicago)\n        0.162   transport_objective\n",
            "x(san-diego,topeka)\n                (.LO, .L, .UP, .M = 0, 0, +INF, 0)\n        1       supply(san-diego)\n        1       demand(topeka)\n        0.126   transport_objective\n",
            "transport_objective_variable\n                (.LO, .L, .UP, .M = -INF, 0, +INF, 0)\n       -1       transport_objective\n",
        ]
    )
    assert len(x.getVariableListing(filters=[["seattle"], []]).split("\n\n")) == 3
    assert (
        len(x.getVariableListing(filters=[["seattle"], ["topeka"]]).split("\n\n")) == 1
    )
    assert len(x.getVariableListing(filters=[["seattle"], []], n=2).split("\n\n")) == 2

    transport2 = Model(
        m,
        name="transport2",
        equations=m.getEquations(),
        problem="LP",
        sense=Sense.MIN,
        objective=Sum((i, j), c[i, j] * x[i, j]),
    )
    transport2.solve()
    with pytest.raises(ValidationError):
        _ = transport2.getVariableListing()


def test_alternative_syntax():
    m = Container()
    i = Set(m, "i", records=["i1", "i2"])
    v = Variable(m, "v", domain=i)
    v2 = Variable(m, "v2", domain=i, type="integer")

    assert v[i].l.gamsRepr() == "v.l(i)"
    assert v[i].m.gamsRepr() == "v.m(i)"
    assert v[i].lo.gamsRepr() == "v.lo(i)"
    assert v[i].up.gamsRepr() == "v.up(i)"
    assert v[i].scale.gamsRepr() == "v.scale(i)"
    assert v[i].fx.gamsRepr() == "v.fx(i)"
    assert v[i].prior.gamsRepr() == "v.prior(i)"
    assert v[i].stage.gamsRepr() == "v.stage(i)"

    v[i].l = 5
    assert v.getAssignment() == "v.l(i) = 5;"
    v[i].m = 5
    assert v.getAssignment() == "v.m(i) = 5;"
    v[i].lo = 5
    assert v.getAssignment() == "v.lo(i) = 5;"
    v[i].up = 5
    assert v.getAssignment() == "v.up(i) = 5;"

    with pytest.raises(ValidationError):
        v2[i].scale = 5

    v[i].scale = 5
    assert v.getAssignment() == "v.scale(i) = 5;"
    v[i].fx = 5
    assert v.getAssignment() == "v.fx(i) = 5;"

    with pytest.raises(ValidationError):
        v[i].prior = 5

    v2[i].prior = 5
    assert v2.getAssignment() == "v2.prior(i) = 5;"
    v[i].stage = 5
    assert v.getAssignment() == "v.stage(i) = 5;"


def test_alternative_operation_syntax():
    m = Container()

    i = Set(m)
    j = Set(m)
    x = Variable(m, domain=[i, j])
    y = Variable(m)

    # Test sum
    with pytest.raises(ValidationError):
        y.sum()

    expr = x.sum()
    expr2 = Sum((i, j), x[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    expr = x.sum(i)
    expr2 = Sum(i, x[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    expr = x.sum(i, j)
    expr2 = Sum((i, j), x[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    # Test product
    with pytest.raises(ValidationError):
        y.product()

    expr = x.product()
    expr2 = Product((i, j), x[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    expr = x.product(i)
    expr2 = Product(i, x[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    expr = x.product(i, j)
    expr2 = Product((i, j), x[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    # Test smin
    with pytest.raises(ValidationError):
        y.smin()

    expr = x.smin()
    expr2 = Smin((i, j), x[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    expr = x.smin(i)
    expr2 = Smin(i, x[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    expr = x.smin(i, j)
    expr2 = Smin((i, j), x[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    # Test smax
    with pytest.raises(ValidationError):
        y.smax()

    expr = x.smax()
    expr2 = Smax((i, j), x[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    expr = x.smax(i)
    expr2 = Smax(i, x[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    expr = x.smax(i, j)
    expr2 = Smax((i, j), x[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    # Test sand
    with pytest.raises(ValidationError):
        y.sand()

    expr = x.sand()
    expr2 = Sand((i, j), x[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    expr = x.sand(i)
    expr2 = Sand(i, x[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    expr = x.sand(i, j)
    expr2 = Sand((i, j), x[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    # Test sor
    with pytest.raises(ValidationError):
        y.sor()

    expr = x.sor()
    expr2 = Sor((i, j), x[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    expr = x.sor(i)
    expr2 = Sor(i, x[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    expr = x.sor(i, j)
    expr2 = Sor((i, j), x[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    ### ImplicitVariable
    # Test sum
    expr = x[i, j].sum()
    expr2 = Sum((i, j), x[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    expr = x[i, j].sum(i)
    expr2 = Sum(i, x[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    expr = x[i, j].sum(i, j)
    expr2 = Sum((i, j), x[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    # Test product
    expr = x[i, j].product()
    expr2 = Product((i, j), x[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    expr = x[i, j].product(i)
    expr2 = Product(i, x[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    expr = x[i, j].product(i, j)
    expr2 = Product((i, j), x[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    # Test smin
    expr = x[i, j].smin()
    expr2 = Smin((i, j), x[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    expr = x[i, j].smin(i)
    expr2 = Smin(i, x[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    expr = x[i, j].smin(i, j)
    expr2 = Smin((i, j), x[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    # Test smax
    expr = x[i, j].smax()
    expr2 = Smax((i, j), x[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    expr = x[i, j].smax(i)
    expr2 = Smax(i, x[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    expr = x[i, j].smax(i, j)
    expr2 = Smax((i, j), x[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    # Test sand
    expr = x[i, j].sand()
    expr2 = Sand((i, j), x[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    expr = x[i, j].sand(i)
    expr2 = Sand(i, x[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    expr = x[i, j].sand(i, j)
    expr2 = Sand((i, j), x[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    # Test sor
    expr = x[i, j].sor()
    expr2 = Sor((i, j), x[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    expr = x[i, j].sor(i)
    expr2 = Sor(i, x[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    expr = x[i, j].sor(i, j)
    expr2 = Sor((i, j), x[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    ### Sum over attribute
    # Test sum
    expr = x.l[i, j].sum()
    expr2 = Sum((i, j), x.l[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    expr = x.l[i, j].sum(i)
    expr2 = Sum(i, x.l[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    expr = x.l[i, j].sum(i, j)
    expr2 = Sum((i, j), x.l[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    # Test product
    expr = x.l[i, j].product()
    expr2 = Product((i, j), x.l[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    expr = x.l[i, j].product(i)
    expr2 = Product(i, x.l[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    expr = x.l[i, j].product(i, j)
    expr2 = Product((i, j), x.l[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    # Test smin
    expr = x.l[i, j].smin()
    expr2 = Smin((i, j), x.l[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    expr = x.l[i, j].smin(i)
    expr2 = Smin(i, x.l[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    expr = x.l[i, j].smin(i, j)
    expr2 = Smin((i, j), x.l[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    # Test smax
    expr = x.l[i, j].smax()
    expr2 = Smax((i, j), x.l[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    expr = x.l[i, j].smax(i)
    expr2 = Smax(i, x.l[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    expr = x.l[i, j].smax(i, j)
    expr2 = Smax((i, j), x.l[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    # Test sand
    expr = x.l[i, j].sand()
    expr2 = Sand((i, j), x.l[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    expr = x.l[i, j].sand(i)
    expr2 = Sand(i, x.l[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    expr = x.l[i, j].sand(i, j)
    expr2 = Sand((i, j), x.l[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    # Test sor
    expr = x.l[i, j].sor()
    expr2 = Sor((i, j), x.l[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    expr = x.l[i, j].sor(i)
    expr2 = Sor(i, x.l[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()

    expr = x.l[i, j].sor(i, j)
    expr2 = Sor((i, j), x.l[i, j])
    assert expr.gamsRepr() == expr2.gamsRepr()


def test_variable_tolist():
    m = gp.Container()
    i = gp.Set(m, "i", records=["seattle", "san-diego"])
    v = gp.Variable(m, "v", domain=[i])

    # Setting records using np.array updates the 'level' by default
    v.setRecords(np.array([1.0, 2.0]))

    # Default columns="level"
    assert v.toList() == [("seattle", 1.0), ("san-diego", 2.0)]

    # Multiple columns (level and marginal)
    # Default marginal values for variables are 0.0 unless specified otherwise
    assert v.toList(columns=["level", "marginal"]) == [
        ("seattle", 1.0, 0.0),
        ("san-diego", 2.0, 0.0),
    ]

    # Empty Variable
    v_empty = gp.Variable(m, "v_empty", domain=[i])
    assert v_empty.toList() == []


def test_variable_equation_tovalue():
    m = gp.Container()

    # Valid scalar Variable
    v = gp.Variable(m, "v")
    v.l[...] = 100.0
    v.m[...] = 5.0
    assert v.toValue() == 100.0  # Defaults to "level"
    assert v.toValue(column="level") == 100.0
    assert v.toValue(column="marginal") == 5.0

    # Valid scalar Equation
    eq = gp.Equation(m, "eq")
    eq.l[...] = -50.0
    assert eq.toValue() == -50.0

    # Invalid: Non-scalar Variable
    i = gp.Set(m, "i", records=["A", "B"])
    v_1d = gp.Variable(m, "v_1d", domain=[i])
    with pytest.raises(TypeError):
        v_1d.toValue()

    # Invalid: Bad column
    with pytest.raises(TypeError):
        v.toValue(column="invalid_column")


def test_tolist_invalid_columns():
    m = gp.Container()
    i = gp.Set(m, "i", records=["A"])
    v = gp.Variable(m, "v", domain=[i])
    v.setRecords(np.array([1.0]))

    with pytest.raises(TypeError):
        # Using an invalid attribute string
        v.toList(columns="invalid_attr")

    with pytest.raises(TypeError):
        # Passing incorrect type to columns
        v.toList(columns=123)


def test_variable_equation_todict():
    m = gp.Container()
    i = gp.Set(m, "i", records=["A", "B"])
    j = gp.Set(m, "j", records=["X", "Y"])

    # 1D Variable
    v = gp.Variable(m, "v", domain=[i])
    v.setRecords(np.array([1.5, 2.5]))

    # Natural Orient, Single Column
    assert v.toDict() == {"A": 1.5, "B": 2.5}
    assert v.toDict(columns="level") == {"A": 1.5, "B": 2.5}

    # Natural Orient, Multiple Columns
    dict_multi = v.toDict(columns=["level", "marginal"])
    assert dict_multi == {
        "A": {"level": 1.5, "marginal": 0.0},
        "B": {"level": 2.5, "marginal": 0.0},
    }

    # Columns Orient
    dict_cols = v.toDict(columns="level", orient="columns")
    assert "i" in dict_cols and "level" in dict_cols
    assert list(dict_cols["level"].values()) == [1.5, 2.5]

    # 2D Equation
    eq = gp.Equation(m, "eq", domain=[i, j])
    # Setting level value for specific records
    eq.l["A", "X"] = 10.0
    eq.l["B", "Y"] = 20.0

    eq_dict = eq.toDict()
    assert ("A", "X") in eq_dict and ("B", "Y") in eq_dict
    assert eq_dict[("A", "X")] == 10.0
    assert eq_dict[("B", "Y")] == 20.0

    # Invalid: Scalar
    v_scalar = gp.Variable(m, "v_scalar")
    with pytest.raises(TypeError):
        v_scalar.toDict()

    # Invalid: Bad column
    with pytest.raises(TypeError):
        v.toDict(columns="invalid_column")


def test_variable_setrecords_scalar():
    m = gp.Container()
    v = gp.Variable(m, "v")

    # Set scalar Variable using float/int
    v.setRecords(100.0)
    assert v.toValue() == 100.0

    v.setRecords(np.array(50.0))
    assert v.toValue() == 50.0


def test_variable_setrecords_array():
    m = gp.Container()
    i = gp.Set(m, "i", records=["A", "B"])
    v = gp.Variable(m, "v", domain=[i])

    # 1D array updates level
    v.setRecords(np.array([10.0, 20.0]))
    assert v.toList() == [("A", 10.0), ("B", 20.0)]


def test_variable_setrecords_dataframe():
    m = gp.Container()
    i = gp.Set(m, "i", records=["A", "B"])
    v = gp.Variable(m, "v", domain=[i])

    # DataFrame format for Variables requires explicit column names for attributes
    # so the ingestor knows which attribute (e.g., 'level') is being populated.
    df = pd.DataFrame({"i": ["A", "B"], "level": [5.0, 10.0]})

    v.setRecords(df)
    assert v.toList() == [("A", 5.0), ("B", 10.0)]


def test_variable_setrecords_clear():
    m = gp.Container()
    v = gp.Variable(m, "v")
    v.setRecords(10.0)
    assert v.records is not None

    # Clear
    v.setRecords(None)
    assert v.records is None


class UnconvertibleType:
    """A mock object designed to fail pandas DataFrame conversion."""

    @property
    def __dict__(self):
        raise ValueError("Cannot convert me")


def test_variable_setrecords_edge_cases():
    m = gp.Container()
    i = gp.Set(m, "i", records=["A", "B"])
    v_scalar = gp.Variable(m, "v_scalar")
    v_1d = gp.Variable(m, "v_1d", domain=[i])

    # Pass scalar to non-scalar variable
    with pytest.raises(
        ValueError, match="Attempting to set a scalar value, but symbol is not scalar"
    ):
        v_1d.setRecords(10.0)

    # Set scalar Variable with dataframe containing multiple rows
    df_multi = pd.DataFrame({"level": [1.0, 2.0]})
    with pytest.raises(
        ValueError, match="Attempting to set multiple records for a scalar symbol"
    ):
        v_scalar.setRecords(df_multi)

    # Dict with dimension mismatch
    with pytest.raises(
        ValueError, match="Dimensionality mismatch between arrays and symbol"
    ):
        v_1d.setRecords({"level": np.array([[1.0, 2.0]])})

    # Dict with shape mismatch between attributes
    with pytest.raises(ValueError, match="Arrays passed do not have the same shape"):
        v_1d.setRecords({"level": np.array([1.0, 2.0]), "marginal": np.array([1.0])})

    # Dict with shape mismatch against domains
    with pytest.raises(
        ValueError, match="Shape mismatch between numpy arrays and domains"
    ):
        v_1d.setRecords({"level": np.array([1.0, 2.0, 3.0])})

    # Series size > 1 for scalar variable
    with pytest.raises(
        ValueError,
        match=r"pandas.Series must have size exactly = 1 for a scalar symbol",
    ):
        v_scalar.setRecords(pd.Series([1.0, 2.0]))

    # Series for non-scalar with dimensionality mismatch
    with pytest.raises(
        ValueError,
        match="Dimensionality of table is inconsistent with domain specification",
    ):
        # Single index series going to a 2D symbol
        v_2d = gp.Variable(m, "v_2d", domain=[i, i])
        v_2d.setRecords(pd.Series([1.0, 2.0], index=["A", "B"]), uels_on_axes=True)

    # DataFrame dimensionality mismatch
    df_wrong = pd.DataFrame({"i": ["A", "B"], "j": ["X", "Y"], "level": [1.0, 2.0]})
    with pytest.raises(
        ValueError,
        match="Dimensionality of records is inconsistent with domain specification",
    ):
        v_1d.setRecords(df_wrong)

    # DataFrame table with MultiIndex containing attributes in multiple indexes
    mi = pd.MultiIndex.from_tuples([("level", "marginal"), ("lower", "upper")])
    df_bad_mi = pd.DataFrame(index=mi, columns=["A", "B"])
    with pytest.raises(ValueError, match="Attributes detected in more than one index"):
        v_1d.setRecords(df_bad_mi, uels_on_axes=True)

    # Unconvertible type
    with pytest.raises(TypeError, match="Could not convert to pandas DataFrame"):
        v_1d.setRecords(UnconvertibleType())


def test_variable_drop_methods():
    m = Container()
    i = Set(m, "i", records=["1", "2", "3", "4", "5"])
    v = Variable(m, "v", domain=[i])

    # A free variable defaults are: level=0.0, marginal=0.0, lower=-inf, upper=inf, scale=1.0
    def get_test_records():
        return pd.DataFrame(
            [
                ["1", SpecialValues.NA, 0.0, -float("inf"), float("inf"), 1.0],
                ["2", SpecialValues.UNDEF, 0.0, -float("inf"), float("inf"), 1.0],
                ["3", SpecialValues.EPS, 0.0, -float("inf"), float("inf"), 1.0],
                ["4", float("nan"), 0.0, -float("inf"), float("inf"), 1.0],
                ["5", 0.0, 0.0, -float("inf"), float("inf"), 1.0],  # Default values
            ],
            columns=["i", "level", "marginal", "lower", "upper", "scale"],
        )

    # Test dropNA
    v.setRecords(get_test_records())
    v.dropNA()
    assert "1" not in v.records["i"].values

    # Test dropUndef
    v.setRecords(get_test_records())
    v.dropUndef()
    assert "2" not in v.records["i"].values
    # Note: UNDEF also evaluates as NaN natively, so it drops "4" as well
    assert "4" not in v.records["i"].values

    # Test dropEps
    v.setRecords(get_test_records())
    v.dropEps()
    assert "3" not in v.records["i"].values

    # Test dropMissing (pandas isna catches NA, UNDEF, and NaN)
    v.setRecords(get_test_records())
    v.dropMissing()
    assert "4" not in v.records["i"].values

    # Test dropDefaults
    v.setRecords(get_test_records())
    v.dropDefaults()
    assert "5" not in v.records["i"].values

    # Ensure no errors occur when calling drop methods on a variable with None records
    v2 = Variable(m, "v2")
    v2.dropNA()
    v2.dropUndef()
    v2.dropEps()
    v2.dropMissing()
    v2.dropDefaults()
    assert v2.records is None


@pytest.mark.skipif(
    not (platform.system() == "Linux" and sys.version_info.minor == 14),
    reason="Test only for linux.",
)
def test_toSparseCoo():
    from scipy.sparse import coo_matrix

    m = gp.Container()
    i = gp.Set(m, "i", records=["A", "B"])
    j = gp.Set(m, "j", records=["X", "Y"])

    # 2D Variable
    v = gp.Variable(m, "v", domain=[i, j])

    arr_2d = np.array([[1.5, 0.0], [0.0, 2.5]])
    v.setRecords(arr_2d)

    # Test column="level" (default)
    mat_level = v.toSparseCoo(column="level")
    assert isinstance(mat_level, coo_matrix)
    assert mat_level.shape == (2, 2)
    assert np.array_equal(mat_level.toarray(), arr_2d)

    # Test column="marginal" (should be default 0.0)
    mat_marginal = v.toSparseCoo(column="marginal")
    assert np.array_equal(mat_marginal.toarray(), np.zeros((2, 2)))

    # 1D Equation
    eq = gp.Equation(m, "eq", domain=[i])
    eq.setRecords(np.array([10.0, -5.0]))
    mat_eq = eq.toSparseCoo()
    assert mat_eq.shape == (1, 2)
    assert np.array_equal(mat_eq.toarray(), np.array([[10.0, -5.0]]))

    # Scalar Equation
    eq_scalar = gp.Equation(m, "eq_scalar")
    eq_scalar.setRecords(-10.0)
    assert eq_scalar.toSparseCoo().toarray()[0, 0] == -10.0

    # Invalid: Type Check
    with pytest.raises(TypeError, match="Argument 'column' must be type str"):
        v.toSparseCoo(column=123)

    # Invalid: Bad column string
    with pytest.raises(TypeError, match="must be one of the following"):
        v.toSparseCoo(column="invalid_col")

    # Invalid: >2D Variable/Equation
    k = gp.Set(m, "k", records=["1"])
    v_3d = gp.Variable(m, "v_3d", domain=[i, j, k])
    v_3d.setRecords(np.zeros((2, 2, 1)))

    with pytest.raises(
        ValidationError, match="only available for data that has dimension <= 2"
    ):
        v_3d.toSparseCoo()


def test_toDense():
    m = gp.Container()
    i = gp.Set(m, "i", records=["A", "B"])
    j = gp.Set(m, "j", records=["X", "Y"])

    # 0D Variable (Scalar)
    v_scalar = gp.Variable(m, "v_scalar")
    v_scalar.setRecords(100.0)
    arr_v_scalar = v_scalar.toDense()
    assert isinstance(arr_v_scalar, (float, np.floating))
    assert arr_v_scalar.item() == 100.0

    # 2D Variable
    v = gp.Variable(m, "v", domain=[i, j])
    arr_2d = np.array([[1.5, 0.0], [0.0, 2.5]])
    v.setRecords(arr_2d)

    # Test column="level" (default)
    arr_level = v.toDense(column="level")
    assert isinstance(arr_level, np.ndarray)
    assert arr_level.shape == (2, 2)
    assert np.array_equal(arr_level, arr_2d)

    # Test column="marginal" (should be default 0.0)
    arr_marginal = v.toDense(column="marginal")
    assert np.array_equal(arr_marginal, np.zeros((2, 2)))

    # 1D Equation
    eq = gp.Equation(m, "eq", domain=[i])
    eq.setRecords(np.array([10.0, -5.0]))
    arr_eq = eq.toDense()
    assert arr_eq.shape == (2,)
    assert np.array_equal(arr_eq, np.array([10.0, -5.0]))

    # Empty Equation
    eq_empty = gp.Equation(m, "eq_empty", domain=[i])
    assert np.allclose(
        eq_empty.toDense(), np.full(eq_empty.shape, eq_empty._default_records["level"])
    )
    assert np.allclose(
        eq_empty.toDense(column="marginal"),
        np.full(eq_empty.shape, eq_empty._default_records["marginal"]),
    )
    assert np.allclose(
        eq_empty.toDense(column="lower"),
        np.full(eq_empty.shape, eq_empty._default_records["lower"]),
    )
    assert np.allclose(
        eq_empty.toDense(column="upper"),
        np.full(eq_empty.shape, eq_empty._default_records["upper"]),
    )
    assert np.allclose(
        eq_empty.toDense(column="scale"),
        np.full(eq_empty.shape, eq_empty._default_records["scale"]),
    )

    # Domain has no records
    m = gp.Container()
    i = gp.Set(m, "i", records=range(5))
    v = gp.Variable(m, "v", domain=i)
    v.generateRecords()
    i.records = None
    with pytest.raises(
        ValidationError, match=r"The domain element `i` of `v` has no records."
    ):
        v.toDense()

    # Invalid: Type Check
    with pytest.raises(TypeError, match="Argument 'column' must be type str"):
        v.toDense(column=123)

    # Invalid: Bad column string
    with pytest.raises(TypeError, match="must be one of the following"):
        v.toDense(column="invalid_col")


def test_implicit_variable_toDense():
    m = Container()
    i = Set(m, "i", records=["i1", "i2"])
    j = Set(m, "j", records=["j1", "j2", "j3"])
    k = Set(m, "k", records=["k1", "k2"])

    arr_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    p = Parameter(m, "p", domain=[i, j], records=arr_2d)
    arr_3d = np.arange(12).reshape((2, 3, 2)).astype(float)
    p3 = Parameter(m, "p3", domain=[i, j, k], records=arr_3d, uels_on_axes=True)

    v = Variable(m, "v", domain=[i, j])
    v.l[i, j] = p[i, j]
    v.m[i, j] = p[i, j] + 10
    v.up[i, j] = p[i, j] + 100
    v.lo[i, j] = p[i, j] - 100
    v3 = Variable(m, "v3", domain=[i, j, k])
    v3.l[i, j, k] = p3[i, j, k]

    # Full indexing returns the level by default and matches the parent.
    full = v[i, j].toDense()
    assert isinstance(full, np.ndarray)
    assert full.dtype == float
    assert np.array_equal(full, arr_2d)
    assert np.array_equal(v3[i, j, k].toDense(), arr_3d)

    # The column argument selects the attribute and agrees with the parent.
    for column in ("level", "marginal", "lower", "upper", "scale"):
        assert np.array_equal(v[i, j].toDense(column), v.toDense(column))
    assert np.array_equal(v[i, j].toDense("marginal"), arr_2d + 10)
    assert np.array_equal(v[i, j].toDense("upper"), arr_2d + 100)
    assert np.array_equal(v[i, j].toDense("lower"), arr_2d - 100)

    # Literal indices reduce the dimensionality.
    assert np.array_equal(v[i, "j2"].toDense(), arr_2d[:, 1])
    assert np.array_equal(v["i1", j].toDense(), arr_2d[0, :])
    assert np.array_equal(v[i, "j2"].toDense("marginal"), (arr_2d + 10)[:, 1])
    assert np.array_equal(v3[i, "j2", k].toDense(), arr_3d[:, 1, :])
    # All indices fixed -> 0-d scalar array.
    scalar = v["i2", "j3"].toDense()
    assert scalar.shape == ()
    assert scalar.item() == arr_2d[1, 2]

    # Transpose / permutation reorders the axes (for every attribute).
    assert np.array_equal(v.t().toDense(), arr_2d.T)
    assert np.array_equal(v.T.toDense(), arr_2d.T)
    assert np.array_equal(v.t().toDense("marginal"), (arr_2d + 10).T)
    assert np.array_equal(v3.t().toDense(), np.transpose(arr_3d, [0, 2, 1]))
    assert np.array_equal(
        gp.math.permute(v3, [1, 0, 2]).toDense(), np.transpose(arr_3d, [1, 0, 2])
    )

    # Parent without records: level/marginal are zeros, bounds use the
    # (free variable) defaults, consistent with the parent's toDense.
    w = Variable(m, "w", domain=[i, j])
    assert np.allclose(w[i, j].toDense(), np.zeros((2, 3)))
    assert np.allclose(w[i, "j1"].toDense(), np.zeros(2))
    assert np.allclose(w[i, j].toDense("marginal"), np.zeros((2, 3)))
    assert np.array_equal(w[i, j].toDense("lower"), w.toDense("lower"))

    # Invalid column.
    with pytest.raises(TypeError, match="Argument 'column' must be type str"):
        v[i, j].toDense(column=123)
    with pytest.raises(TypeError, match="must be one of the following"):
        v[i, j].toDense(column="invalid_col")

    # The temporary parameters used internally must not leak into the container.
    assert set(m.data) == {"i", "j", "k", "p", "p3", "v", "v3", "w"}

    m.close()
