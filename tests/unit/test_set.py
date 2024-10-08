from __future__ import annotations

import pandas as pd
import pytest

from gamspy import Alias, Card, Container, Ord, Parameter, Set
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


def test_set_string(data):
    m, canning_plants, *_ = data
    # Check if the name is reserved
    pytest.raises(ValidationError, Set, m, "set")

    # Without records
    b = Set(m, "b")
    assert b.gamsRepr() == "b"
    assert b.getDeclaration() == "Set b(*);"

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
    new_set = alias.lag(n=5, type="linear")
    assert new_set.gamsRepr() == "A - 5"

    # Linear lead
    new_set = set.lead(n=5, type="linear")
    assert new_set.gamsRepr() == "S + 5"
    new_set = alias.lead(n=5, type="linear")
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


def test_sameas(data):
    m, *_ = data
    i = Set(m, "i")
    j = Alias(m, "j", i)
    assert i.sameAs(j).gamsRepr() == "( sameAs(i,j) )"
    assert j.sameAs(i).gamsRepr() == "( sameAs(j,i) )"

    m = Container()
    i = Set(m, "i", records=["1", "2", "3"])
    p = Parameter(m, "p", [i])
    p[i] = i.sameAs("2")

    assert p.getAssignment() == 'p(i) = ( sameAs(i,"2") );'


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
    assert s.getDeclaration() == "Singleton Set s(*);"

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
