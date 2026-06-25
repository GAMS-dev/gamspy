from __future__ import annotations

import math
import platform
import sys

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


def test_get_sparsity():
    m = gp.Container()

    # Scalar symbol (domain is "none") -> Should return NaN
    p_scalar = gp.Parameter(m, "p_scalar", records=42.0)
    assert math.isnan(p_scalar.getSparsity())

    # Relaxed domain -> Should return NaN
    p_relaxed = gp.Parameter(m, "p_relaxed", domain=["*"])
    p_relaxed.setRecords([("A", 10.0)])
    assert math.isnan(p_relaxed.getSparsity())

    # Create regular sets for domain-based sparsity checks
    i = gp.Set(m, "i", records=["i1", "i2"])  # 2 elements
    j = gp.Set(m, "j", records=["j1", "j2", "j3"])  # 3 elements

    # Partially sparse symbol
    # Maximum possible records = 2 * 3 = 6
    p_sparse = gp.Parameter(m, "p_sparse", domain=[i, j])
    p_sparse.setRecords([("i1", "j1", 10.0)])  # 1 record

    # Sparsity = 1 - (1 / 6) = 5/6 = 0.833333...
    assert math.isclose(p_sparse.getSparsity(), 5 / 6)

    # Fully dense symbol
    p_dense = gp.Parameter(m, "p_dense", domain=[i, j])
    p_dense.setRecords(
        [
            ("i1", "j1", 1),
            ("i1", "j2", 2),
            ("i1", "j3", 3),
            ("i2", "j1", 4),
            ("i2", "j2", 5),
            ("i2", "j3", 6),
        ]
    )
    # 6 records / 6 possible = Sparsity 0.0
    assert p_dense.getSparsity() == 0.0

    # Completely empty symbol
    p_empty = gp.Parameter(m, "p_empty", domain=[i, j])
    # 0 records / 6 possible = Sparsity 1.0
    assert p_empty.getSparsity() == 1.0

    # Domain with no records -> Should return NaN
    # Since set 'k' has no records, we can't compute a maximum density.
    k = gp.Set(m, "k")
    p_no_domain_recs = gp.Parameter(m, "p_no_domain_recs", domain=[i, k])
    assert math.isnan(p_no_domain_recs.getSparsity())


def test_parameter_tovalue():
    m = gp.Container()

    # Valid scalar
    p_scalar = gp.Parameter(m, "p_scalar", records=42.5)
    assert p_scalar.toValue() == 42.5

    # Empty scalar
    p_empty = gp.Parameter(m, "p_empty")
    with pytest.raises(ValidationError):
        p_empty.toValue()

    # Invalid: Non-scalar
    i = gp.Set(m, "i", records=["A", "B"])
    p_1d = gp.Parameter(m, "p_1d", domain=[i], records=np.array([1.0, 2.0]))
    with pytest.raises(TypeError):
        p_1d.toValue()


def test_parameter_tolist():
    m = gp.Container()

    # Scalar Parameter
    p_scalar = gp.Parameter(m, "p_scalar", records=42.0)
    assert p_scalar.toList() == [42.0]

    # 1D Parameter
    i = gp.Set(m, "i", records=["seattle", "san-diego"])
    p_1d = gp.Parameter(m, "p_1d", domain=[i], records=np.array([10.5, 20.5]))
    assert p_1d.toList() == [("seattle", 10.5), ("san-diego", 20.5)]

    # 2D Parameter
    j = gp.Set(m, "j", records=["A", "B"])
    p_2d = gp.Parameter(
        m,
        "p_2d",
        domain=[i, j],
        records=[("seattle", "A", 100), ("san-diego", "B", 200)],
    )
    assert p_2d.toList() == [("seattle", "A", 100.0), ("san-diego", "B", 200.0)]

    # Empty Parameter
    p_empty = gp.Parameter(m, "p_empty", domain=[i])
    assert p_empty.toList() == []


def test_parameter_todict():
    m = gp.Container()

    i = gp.Set(m, "i", records=["A", "B"])
    j = gp.Set(m, "j", records=["X", "Y"])

    # 1D Parameter
    p_1d = gp.Parameter(m, "p_1d", domain=[i], records=np.array([10.5, 20.5]))
    assert p_1d.toDict() == {"A": 10.5, "B": 20.5}

    dict_cols_1d = p_1d.toDict(orient="columns")
    assert "i" in dict_cols_1d and "value" in dict_cols_1d
    assert list(dict_cols_1d["value"].values()) == [10.5, 20.5]

    # 2D Parameter
    p_2d = gp.Parameter(
        m, "p_2d", domain=[i, j], records=[("A", "X", 100), ("B", "Y", 200)]
    )
    assert p_2d.toDict() == {("A", "X"): 100.0, ("B", "Y"): 200.0}

    # Invalid: Scalar
    p_scalar = gp.Parameter(m, "p_scalar", records=5.0)
    with pytest.raises(TypeError):
        p_scalar.toDict()


def test_parameter_setrecords_scalar():
    m = gp.Container()
    p = gp.Parameter(m, "p")

    # Set scalar using float
    p.setRecords(42.5)
    assert p.toValue() == 42.5

    # Set scalar using int
    p.setRecords(10)
    assert p.toValue() == 10.0


def test_parameter_setrecords_list_and_array():
    m = gp.Container()
    i = gp.Set(m, "i", records=["A", "B"])

    p1 = gp.Parameter(m, "p1", domain=[i])
    # 1D array
    p1.setRecords(np.array([10.5, 20.5]))
    assert p1.toList() == [("A", 10.5), ("B", 20.5)]

    with pytest.raises(TypeError, match="Attempted conversion to numpy array failed"):
        p1.setRecords(np.array(["invalid", 10]))

    p2 = gp.Parameter(m, "p2", domain=[i])
    # List of tuples
    p2.setRecords([("A", 100), ("B", 200)])
    assert p2.toList() == [("A", 100.0), ("B", 200.0)]


def test_parameter_setrecords_dataframe():
    m = gp.Container()
    i = gp.Set(m, "i", records=["A", "B"])
    p = gp.Parameter(m, "p", domain=[i])

    # Using DataFrame
    df = pd.DataFrame([["A", 3.14], ["B", 2.71]])
    p.setRecords(df)
    assert p.toList() == [("A", 3.14), ("B", 2.71)]


def test_parameter_setrecords_clear():
    m = gp.Container()
    p = gp.Parameter(m, "p", records=42.0)
    assert p.records is not None

    # Clear records
    p.setRecords(None)
    assert p.records is None


def test_remap_str_special_values():
    # Covers string special values remapping (EPS, NA, UNDEF)
    m = gp.Container()
    i = gp.Set(m, "i", records=["A", "B", "C"])
    p = gp.Parameter(m, "p", domain=[i])

    df = pd.DataFrame({"i": ["A", "B", "C"], "value": ["EPS", "NA", "UNDEF"]})
    p.setRecords(df)

    # Check if they were successfully mapped to numeric special values
    vals = p.records["value"].tolist()

    # EPS acts as 0
    assert vals[0] == 0.0
    assert pd.isna(vals[1])


def test_scalar_setRecords():
    m = gp.Container()
    p = gp.Parameter(m, "test")
    p.setRecords(5)
    assert p.toValue() == 5


class UnconvertibleType:
    """A mock object designed to fail pandas DataFrame conversion."""

    @property
    def __dict__(self):
        raise ValueError("Cannot convert me")


def test_setrecords_edge_cases():
    m = gp.Container()
    i = gp.Set(m, "i", records=["A", "B"])
    _ = gp.Set(m, "j", records=["X", "Y"])
    p_1d = gp.Parameter(m, "p_1d", domain=[i])
    p_scalar = gp.Parameter(m, "p_scalar")

    # Pass 2D array to 1D parameter (Dimension mismatch)
    with pytest.raises(
        ValueError, match="Dimension mismatch between numpy array and parameter domain"
    ):
        p_1d.setRecords(np.array([[1, 2], [3, 4]]))

    # Pass 1D array of wrong shape
    with pytest.raises(
        ValueError, match="Shape mismatch between numpy array and parameter domains"
    ):
        p_1d.setRecords(np.array([1, 2, 3]))  # Expected size 2

    # Array to relaxed domain parameter
    p_relaxed = gp.Parameter(m, "p_relaxed", domain=["*"])
    with pytest.raises(
        ValueError,
        match=r"Data conversion for array format requires a 'regular' domain type",
    ):
        p_relaxed.setRecords(np.array([1, 2]))

    # Pandas Series size > 1 for scalar parameter
    with pytest.raises(
        ValueError,
        match=r"pandas.Series must have size exactly = 1 for a scalar parameter",
    ):
        p_scalar.setRecords(pd.Series([1.0, 2.0]))

    # DataFrame dimensionality mismatch (Table)
    df_wrong = pd.DataFrame([["A", "X", 1], ["B", "Y", 2]])  # 3 columns for a 1D param
    with pytest.raises(
        ValueError,
        match="Dimensionality of records is inconsistent with domain specification",
    ):
        p_1d.setRecords(df_wrong)

    # Unconvertible type
    with pytest.raises(TypeError, match="Could not convert to pandas DataFrame"):
        p_1d.setRecords(UnconvertibleType())


@pytest.mark.skipif(
    not (platform.system() == "Linux" and sys.version_info.minor == 14),
    reason="Test only for linux.",
)
def test_toSparseCoo():
    from scipy.sparse import coo_matrix

    m = gp.Container()

    # 0D Parameter (Scalar)
    p_scalar = gp.Parameter(m, "p_scalar")
    p_scalar.setRecords(42.5)
    mat_scalar = p_scalar.toSparseCoo()
    assert isinstance(mat_scalar, coo_matrix)
    assert mat_scalar.shape == (1, 1)
    assert mat_scalar.toarray()[0, 0] == 42.5

    # 1D Parameter
    i = gp.Set(m, "i", records=["A", "B", "C"])
    p_1d = gp.Parameter(m, "p_1d", domain=[i])
    p_1d.setRecords(np.array([10.0, 0.0, 20.0]))
    mat_1d = p_1d.toSparseCoo()
    assert isinstance(mat_1d, coo_matrix)
    assert mat_1d.shape == (1, 3)
    assert np.array_equal(mat_1d.toarray(), np.array([[10.0, 0.0, 20.0]]))

    # 2D Parameter
    j = gp.Set(m, "j", records=["X", "Y"])
    p_2d = gp.Parameter(m, "p_2d", domain=[i, j])
    arr_2d = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
    p_2d.setRecords(arr_2d)
    mat_2d = p_2d.toSparseCoo()
    assert isinstance(mat_2d, coo_matrix)
    assert mat_2d.shape == (3, 2)
    assert np.array_equal(mat_2d.toarray(), arr_2d)

    # Empty Parameter
    p_empty = gp.Parameter(m, "p_empty", domain=[i])
    assert p_empty.toSparseCoo() is None

    # Invalid: >2D Parameter
    k = gp.Set(m, "k", records=["1"])
    p_3d = gp.Parameter(m, "p_3d", domain=[i, j, k])
    p_3d.setRecords(np.zeros((3, 2, 1)))
    with pytest.raises(
        ValueError, match="only available for data that has dimension <= 2"
    ):
        p_3d.toSparseCoo()


def test_toDense():
    m = gp.Container()

    # 0D Parameter (Scalar)
    p_scalar = gp.Parameter(m, "p_scalar")
    p_scalar.setRecords(42.5)
    arr_scalar = p_scalar.toDense()
    assert isinstance(arr_scalar, np.ndarray)
    assert arr_scalar.shape == ()
    assert arr_scalar.item() == 42.5

    # 1D Parameter
    i = gp.Set(m, "i", records=["A", "B", "C"])
    p_1d = gp.Parameter(m, "p_1d", domain=[i])
    arr_1d_input = np.array([10.0, 0.0, 20.0])
    p_1d.setRecords(arr_1d_input)

    arr_1d_output = p_1d.toDense()
    assert isinstance(arr_1d_output, np.ndarray)
    assert arr_1d_output.shape == (3,)
    assert np.array_equal(arr_1d_output, arr_1d_input)

    # 2D Parameter
    j = gp.Set(m, "j", records=["X", "Y"])
    p_2d = gp.Parameter(m, "p_2d", domain=[i, j])
    arr_2d_input = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    p_2d.setRecords(arr_2d_input)

    arr_2d_output = p_2d.toDense()
    assert isinstance(arr_2d_output, np.ndarray)
    assert arr_2d_output.shape == (3, 2)
    assert np.array_equal(arr_2d_output, arr_2d_input)

    # 3D Parameter (Dense handles >2D arrays)
    k = gp.Set(m, "k", records=["1", "2"])
    p_3d = gp.Parameter(m, "p_3d", domain=[i, j, k])
    arr_3d_input = np.arange(12).reshape((3, 2, 2))
    p_3d.setRecords(arr_3d_input)

    arr_3d_output = p_3d.toDense()
    assert arr_3d_output.shape == (3, 2, 2)
    assert np.array_equal(arr_3d_output, arr_3d_input)

    # Empty Parameter
    p_empty = gp.Parameter(m, "p_empty", domain=[i])
    assert np.allclose(p_empty.toDense(), np.zeros(p_empty.shape, dtype=float))

    # Domain has no records
    m = gp.Container()
    i = gp.Set(m, "i", records=range(5))
    p = gp.Parameter(m, "p", domain=i)
    p.generateRecords()
    i.records = None
    with pytest.raises(
        ValidationError, match=r"The domain element `i` of `p` has no records."
    ):
        p.toDense()
