from __future__ import annotations

import pandas as pd
import pytest

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
    assert a.isValid()

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


def test_expert_sync(data):
    m, *_ = data
    m = Container()
    i = Set(m, "i", records=["i1", "i2"])
    v = Variable(m, "v", domain=i)
    v.l = 5
    v.synchronize = False
    v.l = v.l * 5
    assert v.records.level.tolist() == [5.0, 5.0]
    v.synchronize = True
    assert v.records.level.tolist() == [25.0, 25.0]


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
