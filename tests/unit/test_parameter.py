from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gamspy as gp
from gamspy import (
    Alias,
    Container,
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
from gamspy.exceptions import GamspyException, ValidationError

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


def test_parameter_creation(data):
    m, *_ = data
    # no name is fine
    a = Parameter(m)
    m.addParameter()
    with pytest.raises(ValidationError):
        _ = a.getAssignment()

    # non-str type name
    with pytest.raises(TypeError):
        Parameter(m, 5)

    # no container
    with pytest.raises((TypeError, ValidationError)):
        Parameter()

    # non-container type container
    with pytest.raises(TypeError):
        Parameter(5, "j")

    # try to create a symbol with same name but different type
    _ = Set(m, "i")
    with pytest.raises(TypeError):
        Parameter(m, "i")

    # get already created symbol
    j1 = Parameter(m, "j")
    j2 = Parameter(m, "j")
    assert id(j1) == id(j2)

    # Parameter and domain containers are different
    m2 = Container()
    set1 = Set(m, "set1")
    with pytest.raises(ValidationError):
        _ = Parameter(m2, "param1", domain=[set1])


def test_parameter_string(data):
    m, canning_plants, _, _, capacities, _ = data
    # Check if the name is reserved
    with pytest.raises(ValidationError):
        Parameter(m, "set")

    i = Set(m, name="i", records=canning_plants, description="Canning Plants")
    a = Parameter(m, name="a", domain=[i], records=capacities, description="capacities")

    assert a.getDeclaration() == 'Parameter a(i) "capacities";'

    b = Parameter(m, "b")
    assert b.getDeclaration() == "Parameter b / /;"
    assert (b == 5).gamsRepr() == "b eq 5"
    assert (-b).getDeclaration() == "(-b)"
    assert (b != 5).gamsRepr() == "b ne 5"


def test_implicit_parameter_string(data):
    m, canning_plants, _, _, capacities, _ = data
    m = Container()

    i = Set(m, name="i", records=canning_plants, description="Canning Plants")
    a = Parameter(
        m,
        name="a",
        domain=[i],
        records=capacities,
    )

    assert a[i].gamsRepr() == "a(i)"

    a[i] = -a[i] * 5

    assert a.getAssignment() == "a(i) = (-a(i)) * 5;"

    cont = Container()

    s = Set(cont, "s")
    m = Set(cont, "m")
    A = Parameter(cont, "A", domain=[s, m])

    A.domain = ["s", "m"]
    assert A.getDeclaration() == "Parameter A(*,*) / /;"


def test_parameter_assignment(data):
    m, *_ = data
    m = Container()

    i = Set(m, "i")
    j = Set(m, "j")
    a = Parameter(m, "a", domain=[i])

    with pytest.raises(ValidationError):
        a[j] = 5


def test_implicit_parameter_assignment(data):
    m, canning_plants, _, _, capacities, _ = data
    m = Container()
    i = Set(m, name="i", records=canning_plants, description="Canning Plants")
    a = Parameter(
        m,
        name="a",
        domain=[i],
        records=capacities,
    )

    b = Parameter(
        m,
        name="b",
        domain=[i],
        records=capacities,
    )

    a[i] = b[i]
    assert a.getAssignment() == "a(i) = b(i);"

    v = Variable(m, "v", domain=[i])
    v.l[i] = v.l[i] * 5

    assert v.getAssignment() == "v.l(i) = v.l(i) * 5;"


def test_equality(data):
    m, *_ = data
    m = Container()
    j = Set(m, "j")
    h = Set(m, "h")
    hp = Alias(m, "hp", h)
    lamb = Parameter(m, "lambda", domain=[j, h])
    gamma = Parameter(m, "gamma", domain=[j, h])
    gamma[j, h] = Sum(hp.where[Ord(hp) >= Ord(h)], lamb[j, hp])
    assert (
        gamma.getAssignment()
        == "gamma(j,h) = sum(hp $ (ord(hp) >= ord(h)),lambda(j,hp));"
    )


def test_override(data):
    m, *_ = data
    # Parameter record override
    s = Set(m, name="s", records=[str(i) for i in range(1, 4)])
    c = Parameter(m, name="c", domain=[s])
    c = m.addParameter(
        name="c",
        domain=[s],
        records=[("1", 1), ("2", 2), ("3", 3)],
        description="new description",
    )
    assert c.description == "new description"

    # Try to add the same parameter
    with pytest.raises(ValueError):
        m.addParameter("c", [s, s])


def test_undef():
    m = Container(debugging_level="keep")
    _ = Parameter(
        m, name="rho", records=[np.nan]
    )  # Instead of using numpy there might be a NA from the math package

    generated = m.generateGamsString()
    expected = "$onMultiR\n$onUNDF\n$onDotL\nParameter rho / Undf /;\n$offDotL\n$offUNDF\n$offMulti\n"
    assert generated == expected


def test_assignment_dimensionality(data):
    m, *_ = data
    j1 = Set(m, "j1")
    j2 = Set(m, "j2")
    j3 = Parameter(m, "j3", domain=[j1, j2])
    with pytest.raises(ValidationError):
        j3["bla"] = 5

    j4 = Set(m, "j4")

    with pytest.raises(ValidationError):
        j3[j1, j2, j4] = 5

    with pytest.raises(ValidationError):
        j3[j1, j2] = j3[j1, j2, j4] * 5


def test_domain_verification(data):
    m, *_ = data
    m = Container()
    i1 = Set(m, "i1", records=["i1", "i2"])
    a1 = Parameter(m, "a1", domain=i1, records=[("i1", 1), ("i2", 2)])
    a1["i1"] = 5

    with pytest.raises(ValidationError):
        a1["i3"] = 5

    with pytest.raises(ValidationError):
        a1["i3"] = a1["i3"] * 5


def test_uels_on_axes(data):
    m, *_ = data
    s = pd.Series(index=["a", "b", "c"], data=[i + 1 for i in range(3)])
    i = Parameter(m, "i", ["*"], records=s, uels_on_axes=True)
    assert i.records.value.tolist() == [1, 2, 3]


def test_domain_violation(data):
    m, *_ = data
    col = Set(m, "col", records=[("col" + str(i), i) for i in range(1, 10)])
    row = Set(m, "row", records=[("row" + str(i), i) for i in range(1, 10)])

    initial_state_data = pd.DataFrame(
        [
            [0, 0, 0, 0, 8, 6, 0, 0, 0],
            [0, 7, 0, 9, 0, 2, 0, 0, 0],
            [6, 9, 0, 0, 0, 0, 2, 0, 8],
            [8, 0, 0, 0, 9, 0, 7, 0, 2],
            [4, 0, 0, 0, 0, 0, 0, 0, 3],
            [2, 0, 9, 0, 1, 0, 0, 0, 4],
            [5, 0, 3, 0, 0, 0, 0, 7, 6],
            [0, 0, 0, 5, 0, 8, 0, 2, 0],
            [0, 0, 0, 3, 7, 0, 0, 0, 0],
        ],
        index=["roj" + str(i) for i in range(1, 10)],
        columns=["col" + str(i) for i in range(1, 10)],
    )

    with pytest.raises(GamspyException):
        _ = Parameter(
            m,
            "initial_state",
            domain=[row, col],
            records=initial_state_data,
            uels_on_axes=True,
        )


def test_expert_sync(data):
    m, *_ = data
    f = Parameter(m, "f")
    f.setRecords(3)  # Python: 3 GAMS: 3
    assert f.toValue() == 3
    f[...] = 2  # Python: 2 GAMS: 2
    assert f.toValue() == 2
    f.synchronize = False
    f.setRecords(3)  # Python: 3 GAMS: 2
    assert f.toValue() == 3
    f[...] = 1  # Python: 3 GAMS: 1
    assert f.toValue() == 3
    f.synchronize = True  # Python: 1 GAMS: 1 (GAMS wins because the user has assignment statement last)
    assert f.toValue() == 1


def test_expert_sync2(data):
    m, *_ = data
    f = Parameter(m, "f")
    f.setRecords(3)  # Python: 3 GAMS: 3
    assert f.toValue() == 3
    f[...] = 2  # Python: 2 GAMS: 2
    assert f.toValue() == 2
    f.synchronize = False
    f[...] = 1  # Python: 2 GAMS: 1
    assert f.toValue() == 2
    f.setRecords(3)  # Python: 3 GAMS: 1
    assert f.toValue() == 3
    f.synchronize = (
        True  # Python: 3 GAMS: 3 (Python wins because the user has setRecords last)
    )
    assert f.toValue() == 3


def test_expert_sync3():
    m = gp.Container()
    a = gp.Parameter(m, "a")
    b = gp.Parameter(m, "b")

    a.synchronize = False
    a[...] = 2
    b[...] = 1
    assert b.toValue() == 1
    assert a.records is None
    a.synchronize = True

    assert a.toValue() == 2
    assert b.toValue() == 1
    m.close()


def test_control_domain(data):
    m, *_ = data
    i = Set(m, "i", records=["i1", "i2"])
    j = Set(m, "j", records=["j1", "j2"])

    a = Parameter(m, "a", domain=i)
    b = Parameter(m, "b", domain=j, records=[("j1", 1), ("j2", 2)])

    with pytest.raises(ValidationError):
        a[i] = b[j]

    with pytest.raises(ValidationError):
        a[i.lead(1)] = b[j]

    with pytest.raises(ValidationError):
        a[i] = b[j.lead(1)]

    with pytest.raises(ValidationError):
        a[i.lead(1)] = b[j.lead(1)]


def test_alternative_operation_syntax():
    m = Container()

    i = Set(m)
    j = Set(m)
    x = Parameter(m, domain=[i, j])
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

    ### ImplicitParameter
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


def test_domain_with_no_records_validation():
    # Should not fail if the domain set does not have any records
    # See: #766
    m = gp.Container()
    i = gp.Set(m)
    a = gp.Parameter(m, domain=i)
    assert a["i2"].records is None
