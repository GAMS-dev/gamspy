from __future__ import annotations

import pandas as pd
import pytest

import gamspy.math as gp_math
from gamspy import (
    Alias,
    Card,
    Container,
    Number,
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
from gamspy._algebra.expression import SetExpression
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

    yield m, canning_plants, markets, distances, capacities, demands
    m.close()


def test_set_creation(data):
    m, *_ = data
    # no name is fine now
    i = Set(m)
    m.addSet()
    assert len(i) == 0
    with pytest.raises(ValidationError):
        _ = i.getAssignment()

    # non-str type name
    pytest.raises(TypeError, Set, m, 5)

    # no container
    pytest.raises((ValidationError, TypeError), Set)

    # non-container type container
    pytest.raises(TypeError, Set, 5, "j")

    # try to create a symbol with same name but different type
    _ = Parameter(m, "i")
    pytest.raises(TypeError, Set, m, "i")

    # get already created symbol
    j1 = Set(m, "j")
    j2 = Set(m, "j")
    assert id(j1) == id(j2)

    # Set and domain containers are different
    m2 = Container()
    set1 = Set(m, "set1")
    with pytest.raises(ValidationError):
        _ = Set(m2, "set2", domain=[set1])

    # GAMSPy symbols are not iterable to avoid infinite loop on builtin Python sum.
    with pytest.raises(ValidationError):
        sum(i)


def test_set_string(data):
    m, canning_plants, *_ = data
    # Check if the name is reserved
    pytest.raises(ValidationError, Set, m, "set")

    # Without records
    b = Set(m, "b")
    assert b.gamsRepr() == "b"
    assert b.getDeclaration() == "Set b(*) / /;"

    # Without domain
    i = Set(m, "i", records=canning_plants, description="dummy set")
    assert i.gamsRepr() == "i"
    assert i.getDeclaration() == 'Set i(*) "dummy set";'

    # With one domain
    j = Set(m, "j", records=["seattle", "san-diego", "california"])
    k = Set(m, "k", domain=[j], records=canning_plants)
    assert k.gamsRepr() == "k"
    assert k.getDeclaration() == "Set k(j);"

    # With two domain
    r = Set(m, "m", records=[f"i{i}" for i in range(2)])
    n = Set(m, "n", records=[f"j{i}" for i in range(2)])
    a = Set(m, "a", [r, n])
    a.generateRecords(density=1)
    assert a.gamsRepr() == "a"
    assert a.getDeclaration() == "Set a(m,n);"


def test_records_assignment(data):
    m, *_ = data
    new_cont = Container()
    i = Set(m, "i")
    j = Set(m, "j", domain=[i])
    k = Set(new_cont, "k")

    s = Set(m, "s", domain=[i])
    with pytest.raises(TypeError):
        s.records = 5

    with pytest.raises(ValidationError):
        j[k] = 5


def test_set_operators(data):
    m, canning_plants, *_ = data
    i = Set(m, "i", records=canning_plants)
    card = Card(i)
    assert card.gamsRepr() == "card(i)"

    ord = Ord(i)
    assert ord.gamsRepr() == "ord(i)"


def test_implicit_sets(data):
    m, canning_plants, *_ = data
    m = Container()
    j = Set(m, "j", records=["seattle", "san-diego", "california"])
    k = Set(m, "k", domain=[j], records=canning_plants)

    expr = k[j] <= k[j]
    assert expr.gamsRepr() == "(k(j) <= k(j))"
    expr = k[j] >= k[j]
    assert expr.gamsRepr() == "(k(j) >= k(j))"

    k[j] = ~k[j]

    assert k.getAssignment() == "k(j) = ( not k(j));"


def test_set_operations(data):
    m, canning_plants, *_ = data
    i = Set(m, "i", records=canning_plants)
    k = Set(m, "k", records=canning_plants)
    union = i + k
    assert union.gamsRepr() == "(i + k)"

    intersection = i * k
    assert intersection.gamsRepr() == "(i * k)"

    complement = ~i
    assert complement.gamsRepr() == "( not i)"

    difference = i - k
    assert difference.gamsRepr() == "(i - k)"


def test_dynamic_sets(data):
    m, *_ = data
    m = Container()
    i = Set(m, name="i", records=[f"i{idx}" for idx in range(1, 4)])
    i["i1"] = False

    assert i.getAssignment() == 'i("i1") = no;'

    m = Container()
    k = Set(m, name="k", records=[f"k{idx}" for idx in range(1, 4)])
    k["k1"] = False


def test_lag_and_lead(data):
    m, *_ = data
    set = Set(m, name="S", records=["a", "b", "c"], description="Test text")
    alias = Alias(m, "A", alias_with=set)

    # Circular lag
    new_set = set.lag(n=5, type="circular")
    assert new_set.gamsRepr() == "S -- 5"
    new_set = alias.lag(n=5, type="circular")
    assert new_set.gamsRepr() == "A -- 5"

    # Circular lead
    new_set = set.lead(n=5, type="circular")
    assert new_set.gamsRepr() == "S ++ 5"
    new_set = alias.lead(n=5, type="circular")
    assert new_set.gamsRepr() == "A ++ 5"

    # Linear lag
    new_set = set.lag(n=5, type="linear")
    assert new_set.gamsRepr() == "S - 5"
    new_set = set - 5
    assert new_set.gamsRepr() == "S - 5"
    new_set = alias.lag(n=5, type="linear")
    assert new_set.gamsRepr() == "A - 5"
    new_set = alias - 5
    assert new_set.gamsRepr() == "A - 5"

    # Linear lead
    new_set = set.lead(n=5, type="linear")
    assert new_set.gamsRepr() == "S + 5"
    new_set = set + 5
    assert new_set.gamsRepr() == "S + 5"
    new_set = alias.lead(n=5, type="linear")
    assert new_set.gamsRepr() == "A + 5"
    new_set = alias + 5
    assert new_set.gamsRepr() == "A + 5"

    # Incorrect type
    pytest.raises(ValueError, set.lead, 5, "bla")
    pytest.raises(ValueError, alias.lead, 5, "bla")
    pytest.raises(ValueError, set.lag, 5, "bla")
    pytest.raises(ValueError, alias.lag, 5, "bla")

    m = Container()
    s = Set(m, name="s", records=[f"s{i}" for i in range(1, 4)])
    t = Set(m, name="t", records=[f"t{i}" for i in range(1, 6)])

    sMinDown = Set(m, name="sMinDown", domain=[s, t])
    sMinDown[s, t.lead(Ord(t) - Ord(s))] = 1

    assert sMinDown.getAssignment() == "sMinDown(s,t + (ord(t) - ord(s))) = 1;"

    i = Set(m, "i", records=range(3))
    t = Set(m, "t", records=range(3))
    a = Parameter(m, "a", domain=i)
    b = Parameter(m, "b", domain=i)
    a[i.lag(i.val)] = 5
    b[i - i.val] = 5
    assert a.records.equals(b.records)

    c = Parameter(m, "c", domain=i)
    d = Parameter(m, "d", domain=i)
    c[i.lead(i.val)] = 5
    d[i + i.val] = 5
    assert c.records.equals(d.records)


def test_set_attributes(data):
    m, *_ = data
    i = Set(m, "i")
    assert i.pos.gamsRepr() == "i.pos"
    assert i.ord.gamsRepr() == "i.ord"
    assert i.off.gamsRepr() == "i.off"
    assert i.rev.gamsRepr() == "i.rev"
    assert i.uel.gamsRepr() == "i.uel"
    assert i.len.gamsRepr() == "i.len"
    assert i.tlen.gamsRepr() == "i.tlen"
    assert i.val.gamsRepr() == "i.val"
    assert i.tval.gamsRepr() == "i.tval"
    assert i.first.gamsRepr() == "i.first"
    assert i.last.gamsRepr() == "i.last"

    i = Set(m, "i", records=range(5))
    j = Set(m, "j", records=range(3))
    a = Parameter(m, "a", domain=[i, j])
    a[i, j].where[i.ord == j.ord] = 5
    assert a.toList() == [("0", "0", 5.0), ("1", "1", 5.0), ("2", "2", 5.0)]

    b = Parameter(m, "b", domain=[i, j])
    b[i, j].where[i.ord != j.ord] = 5
    assert b.toList() == [
        ("0", "1", 5.0),
        ("0", "2", 5.0),
        ("1", "0", 5.0),
        ("1", "2", 5.0),
        ("2", "0", 5.0),
        ("2", "1", 5.0),
        ("3", "0", 5.0),
        ("3", "1", 5.0),
        ("3", "2", 5.0),
        ("4", "0", 5.0),
        ("4", "1", 5.0),
        ("4", "2", 5.0),
    ]

    m = Container()
    N = Parameter(m, "N", records=20)
    L = Parameter(m, "L", records=int(N.toValue()) / 2)
    v = Set(m, "v", records=range(1, int(N.toValue()) + 1))
    i = Set(m, "i", domain=v)
    i[v] = Number(1).where[v.val < L]
    j = Alias(m, "j", i)

    tight = Set(
        m,
        "tight",
        description="diameter constraints that are conjectured tight",
        domain=[v, v],
    )
    tight[i, j] = Number(1).where[
        (i.val >= (L - 1) / 2)
        & (j.val >= L - 1 - i.val)
        & (j.val <= i.val)
        & (j.val <= L - i.val)
    ]
    assert (
        tight._assignment.gamsRepr()
        == "tight(i,j) = (1 $ ((((i.val >= ((L - 1) / 2)) and (j.val >= ((L - 1) - i.val))) and (j.val <= i.val)) and (j.val <= (L - i.val))));"
    )

    m = Container()

    i = Set(m, "i", records=range(3))
    j = Alias(m, "j", alias_with=i)
    assert i.pos.records.values.tolist() == [
        ["0", "position", 1.0],
        ["1", "position", 2.0],
        ["2", "position", 3.0],
    ]
    assert j.pos.records.values.tolist() == [
        ["0", "position", 1.0],
        ["1", "position", 2.0],
        ["2", "position", 3.0],
    ]
    m.close()


def test_set_assignment():
    container = Container()

    i = Set(container, "i")
    m = Set(container, "m")
    g = Set(container, "g")
    k = Parameter(container, "k", domain=[m, i])
    f = Parameter(container, "f", domain=[m, g, i])
    mpos = Set(container, "mpos", domain=[m, i])
    hpos = Set(container, "hpos", domain=[m, i])

    # Set Union
    assert isinstance(
        mpos[m, i] + hpos[m, i], SetExpression
    )  # ImplicitSet + ImplicitSet
    assert isinstance(mpos[m, i] + 1, SetExpression)  # ImplicitSet + Number
    assert isinstance(1 + hpos[m, i], SetExpression)  # Number + ImplicitSet
    assert isinstance(
        (mpos[m, i] + hpos[m, i]) + 1, SetExpression
    )  # SetExpression + Number
    assert isinstance(
        1 + (mpos[m, i] + hpos[m, i]), SetExpression
    )  # Number + SetExpression
    assert isinstance(
        (mpos[m, i] + hpos[m, i]) + (mpos[m, i] + hpos[m, i]), SetExpression
    )  # SetExpression + SetExpression
    assert isinstance(
        (mpos[m, i] + hpos[m, i]) + mpos[m, i], SetExpression
    )  # SetExpression + ImplicitSet
    assert isinstance(
        mpos[m, i] + (mpos[m, i] + hpos[m, i]), SetExpression
    )  # ImplicitSet + SetExpression

    # Set Difference
    assert isinstance(
        mpos[m, i] - hpos[m, i], SetExpression
    )  # ImplicitSet - ImplicitSet
    assert isinstance(mpos[m, i] - 1, SetExpression)  # ImplicitSet - Number
    assert isinstance(1 - hpos[m, i], SetExpression)  # Number - ImplicitSet
    assert isinstance(
        (mpos[m, i] - hpos[m, i]) - 1, SetExpression
    )  # SetExpression - Number
    assert isinstance(
        1 - (mpos[m, i] - hpos[m, i]), SetExpression
    )  # Number - SetExpression
    assert isinstance(
        (mpos[m, i] - hpos[m, i]) - (mpos[m, i] - hpos[m, i]), SetExpression
    )  # SetExpression - SetExpression
    assert isinstance(
        (mpos[m, i] - hpos[m, i]) - mpos[m, i], SetExpression
    )  # SetExpression - ImplicitSet
    assert isinstance(
        mpos[m, i] - (mpos[m, i] - hpos[m, i]), SetExpression
    )  # ImplicitSet - SetExpression

    # Set Intersection
    assert isinstance(
        mpos[m, i] * hpos[m, i], SetExpression
    )  # ImplicitSet * ImplicitSet
    assert isinstance(mpos[m, i] * 1, SetExpression)  # ImplicitSet * Number
    assert isinstance(1 * hpos[m, i], SetExpression)  # Number * ImplicitSet
    assert isinstance(
        (mpos[m, i] * hpos[m, i]) * 1, SetExpression
    )  # SetExpression * Number
    assert isinstance(
        1 * (mpos[m, i] * hpos[m, i]), SetExpression
    )  # Number * SetExpression
    assert isinstance(
        (mpos[m, i] * hpos[m, i]) * (mpos[m, i] * hpos[m, i]), SetExpression
    )  # SetExpression * SetExpression
    assert isinstance(
        (mpos[m, i] * hpos[m, i]) * mpos[m, i], SetExpression
    )  # SetExpression * ImplicitSet
    assert isinstance(
        mpos[m, i] * (mpos[m, i] * hpos[m, i]), SetExpression
    )  # ImplicitSet * SetExpression

    # Set Complement
    assert isinstance(~mpos[m, i], SetExpression)  # not ImplicitSet
    assert isinstance(
        ~(mpos[m, i] + hpos[m, i]), SetExpression
    )  # not SetExpression

    mpos[m, i] = (
        (Number(1).where[(k[m, i] + Sum(g, gp_math.abs(f[m, g, i])))])
        - hpos[m, i]
    )
    assert isinstance(mpos._assignment.right, SetExpression)

    assert (
        mpos.getAssignment()
        == "mpos(m,i) = ((yes $ (k(m,i) + sum(g,abs(f(m,g,i))))) - hpos(m,i));"
    )

    with pytest.raises(ValidationError):
        mpos[m, i] = (
            (Number(2).where[(k[m, i] + Sum(g, gp_math.abs(f[m, g, i])))])
            - hpos[m, i]
        )

    with pytest.raises(ValidationError):
        mpos[m, i] = (
            hpos[m, i]
            + Number(2).where[(k[m, i] + Sum(g, gp_math.abs(f[m, g, i])))]
        )

    reb = Set(container, "reb")
    con = Set(container, "con")
    col = Set(container, "col")
    col[i] = 1 - (reb[i] - con[i])
    assert isinstance(col._assignment.right, SetExpression)
    assert col.getAssignment() == "col(i) = (yes - (reb(i) - con(i)));"

    m = Container()
    i = Set(m, "i", records=["i1", "i2"])
    j = Set(m, "j", domain=i, records=["i1"])
    k = Set(m, "k", domain=i)
    k[i] = j[i] + 1
    with pytest.raises(ValidationError):
        k[i] = j[i] + 5


def test_sameas(data):
    m, *_ = data
    i = Set(m, "i")
    j = Alias(m, "j", i)
    assert i.sameAs(j).gamsRepr() == "sameAs(i,j)"
    assert j.sameAs(i).gamsRepr() == "sameAs(j,i)"

    m = Container()
    i = Set(m, "i", records=["1", "2", "3"])
    p = Parameter(m, "p", [i])
    p[i] = i.sameAs("2")

    assert p.getAssignment() == 'p(i) = sameAs(i,"2");'


def test_assignment_dimensionality(data):
    m, *_ = data
    j1 = Set(m, "j1")
    j2 = Set(m, "j2")
    j3 = Set(m, "j3", domain=[j1, j2])
    with pytest.raises(ValidationError):
        j3["bla"] = 5

    j4 = Set(m, "j4")

    with pytest.raises(ValidationError):
        j3[j1, j2, j4] = 5

    j5 = Set(m, "j5", domain=[j1, j2])
    j6 = Set(m, "j6", domain=[j1, j2])

    with pytest.raises(ValidationError):
        j6[j1, j2] = j5[j1, j2, j3]


def test_domain_verification(data):
    m, *_ = data
    m = Container()
    i1 = Set(m, "i1", records=["i1", "i2"])
    i2 = Set(m, "i2", records=["i1"], domain=i1)

    with pytest.raises(ValidationError):
        i2["i3"] = True


def test_uels_on_axes(data):
    m, *_ = data
    s = pd.Series(index=["a", "b"])
    i = Set(m, "i", records=s, uels_on_axes=True)
    assert i.records["uni"].tolist() == ["a", "b"]


def test_expert_sync(data):
    m, *_ = data
    m = Container()
    i = Set(m, "i", records=["i1"])
    i.synchronize = False
    i["i2"] = True
    assert i.records.uni.tolist() == ["i1"]
    i.synchronize = True
    assert i.records.uni.tolist() == ["i1", "i2"]


def test_singleton():
    m = Container()
    s = Set(m, "s", is_singleton=True)
    s2 = Set(m, "s2", is_singleton=True)
    assert s.getDeclaration() == "Singleton Set s(*) / /;"

    with pytest.raises(ValidationError):
        _ = Set(m, "s3", is_singleton=True, records=["i1", "i2"])

    with pytest.raises(ValidationError):
        _ = Set(m, "s4", domain=[s, s2], is_singleton=True)

    node = Set(m, "node", records=range(1, 11))
    T1 = Set(m, "T1", domain=node, is_singleton=True, records=["1"])

    result = Parameter(m, "result", domain=["*"])
    b = Parameter(m, "b", domain=node)

    b[T1] = 1
    result["b-T1"] = b[T1]

    s5 = Set(m, "s5", is_singleton=True, records=["s1"])
    assert s5.toList() == ["s1"]
    s5[s5].where[False] = True
    assert s5.records is None


def test_indexing():
    m = Container()
    i = Set(m, "i", records=range(3))
    j = Set(m, "j", records=range(3))
    a = Parameter(m, "a", domain=[i, j])
    v = Variable(m, "v", domain=[i, j])
    e = Variable(m, "e", domain=[i, j])

    assert i[1].gamsRepr() == 'i("1")'
    assert i["1"].gamsRepr() == i[1].gamsRepr()
    assert a[i, 2].gamsRepr() == 'a(i,"2")'
    assert a[i, "2"].gamsRepr() == a[i, 2].gamsRepr()
    assert a[1, 2].gamsRepr() == 'a("1","2")'
    assert v[i, 2].gamsRepr() == 'v(i,"2")'
    assert v[i, "2"].gamsRepr() == v[i, 2].gamsRepr()
    assert v[1, 2].gamsRepr() == 'v("1","2")'
    assert v.l[i, 2].gamsRepr() == 'v.l(i,"2")'
    assert v.l[i, "2"].gamsRepr() == v.l[i, 2].gamsRepr()
    assert v.l[1, 2].gamsRepr() == 'v.l("1","2")'
    assert e[i, 2].gamsRepr() == 'e(i,"2")'
    assert e[i, "2"].gamsRepr() == e[i, 2].gamsRepr()
    assert e[1, 2].gamsRepr() == 'e("1","2")'
    assert e.l[i, 2].gamsRepr() == 'e.l(i,"2")'
    assert e.l[1, 2].gamsRepr() == 'e.l("1","2")'

    k = Set(m, "k", records=range(3))
    k[0] = False

    a.generateRecords()
    a[0, 1] = 5

    v.generateRecords()
    v.l[0, 1] = 5

    e.generateRecords()
    e.l[0, 1] = 5


def test_alternative_operation_syntax():
    m = Container()

    i = Set(m)
    j = Set(m)
    x = Set(m, domain=[i, j])
    y = Parameter(m)

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
