from __future__ import annotations

import pandas as pd
import pytest

from gamspy import (
    Alias,
    Container,
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
    UniverseAlias,
)
from gamspy.exceptions import ValidationError

pytestmark = pytest.mark.unit


@pytest.fixture
def m():
    container = Container()
    yield container
    container.close()


def test_alias_creation(m):
    i = Set(m, "i")
    m.addAlias(alias_with=i)

    a = Alias(m, alias_with=i)
    assert len(a) == 0
    with pytest.raises(ValidationError):
        _ = a.getAssignment()

    # no alias
    with pytest.raises(TypeError):
        _ = Alias(m)

    # non-str type name
    with pytest.raises(TypeError):
        _ = Alias(m, 5, i)

    # no container
    with pytest.raises(ValidationError):
        _ = Alias()

    # non-container type container
    with pytest.raises(TypeError):
        _ = Alias(5, "j", i)

    # try to create a symbol with same name but different type
    with pytest.raises(TypeError):
        _ = Alias(m, "i", i)

    # get already created symbol
    j1 = Alias(m, "j", i)
    j2 = Alias(m, "j", i)
    assert id(j1) == id(j2)

    # len of Alias
    i2 = Set(m, records=["i1", "i2"])
    k2 = Alias(m, "k2", alias_with=i2)
    assert len(k2) == 2

    k2["i1"] = False
    assert k2.getAssignment() == 'k2("i1") = no;'

    # synch
    with pytest.raises(ValidationError):
        k2.synchronize = True

    with pytest.raises(ValidationError):
        _ = k2.synchronize


def test_alias_string(m):
    # Set and Alias without domain
    i = Set(m, name="i", records=["a", "b", "c"])
    j = Alias(m, name="j", alias_with=i)
    assert j.gamsRepr() == "j"
    assert j.getDeclaration() == "Alias(i,j);"

    # Set and Alias with domain
    k = Set(m, name="k", domain=[i], records=["a", "b"])
    l = Alias(m, name="l", alias_with=k)
    assert l.gamsRepr() == "l"
    assert l.getDeclaration() == "Alias(k,l);"

    # Check if the name is reserved
    with pytest.raises(ValidationError):
        _ = Alias(m, "set", i)


def test_override(m):
    # Try to add the same Alias with non-Set alias_with
    u = Set(m, "u")
    v = Alias(m, "v", alias_with=u)
    eq = Equation(m, "eq", domain=[u, v])
    with pytest.raises(TypeError):
        _ = m.addAlias("v", eq)

    # Try to add the same alias
    with pytest.raises(TypeError):
        _ = m.addAlias("u", u)


def test_alias_attributes(m):
    i = Set(m, "i")
    j = Alias(m, "j", alias_with=i)
    assert j.pos.gamsRepr() == "j.pos"
    assert j.ord.gamsRepr() == "j.ord"
    assert j.off.gamsRepr() == "j.off"
    assert j.rev.gamsRepr() == "j.rev"
    assert j.uel.gamsRepr() == "j.uel"
    assert j.len.gamsRepr() == "j.len"
    assert j.tlen.gamsRepr() == "j.tlen"
    assert j.val.gamsRepr() == "j.val"
    assert j.tval.gamsRepr() == "j.tval"
    assert j.first.gamsRepr() == "j.first"
    assert j.last.gamsRepr() == "j.last"


def test_universe_alias_creation(m):
    # non-str type name
    with pytest.raises(TypeError):
        _ = UniverseAlias(m, 5)

    # no container
    with pytest.raises(ValidationError):
        UniverseAlias()

    # non-container type container
    with pytest.raises(TypeError):
        UniverseAlias(5, "j")

    # try to create a symbol with same name but different type
    _ = Set(m, "i")
    with pytest.raises(TypeError):
        UniverseAlias(m, "i")

    # get already created symbol
    j1 = UniverseAlias(m, "j")
    j2 = UniverseAlias(m, "j")
    assert id(j1) == id(j2)

    u = UniverseAlias(m, name="u")
    p = Parameter(m, name="p", domain=u)
    p[u] = 2


def test_universe_alias(m, tmp_path):
    gdx_path = str(tmp_path / "test.gdx")

    h = UniverseAlias(m, "h")
    _ = Set(m, "i", records=["i1", "i2"])
    _ = Set(m, "j", records=["j1", "j2"])

    assert h.records.values.tolist() == [["i1"], ["i2"], ["j1"], ["j2"]]

    m.write(gdx_path)

    bla = Container()
    bla.read(gdx_path)
    assert bla.data["h"].records.values.tolist() == h.records.values.tolist()

    m = Container()

    r = UniverseAlias(m, name="new_universe")
    k = Set(m, name="k", domain=r, records="Chicago")
    assert k.getDeclaration() == "Set k(*);"

    u1 = m.addUniverseAlias(name="universe_name")
    assert u1.name == "universe_name"

    u2 = m.addUniverseAlias()
    assert u2.name == "u2"


def test_alias_state(m):
    i = Set(m, name="i", records=["a", "b", "c"])
    j = Alias(m, name="j", alias_with=i)
    i.modified = False
    j.setRecords(["a", "b"])
    assert not i.modified

    i.modified = False
    j.records = pd.DataFrame([["a", "b"]])
    assert i.modified


def test_alias_modified_list(m):
    nodes = m.addSet("nodes", description="nodes", records=["s"])
    i = m.addAlias("i", nodes)
    _ = m.addSet("s", domain=[i], description="sources", records=["s"])
    modified_names = m._get_modified_symbols()
    assert modified_names == []


def test_indexing(m):
    row = Set(m, "row", records=[("r-" + str(i), i) for i in range(1, 11)])
    col = Set(m, "col", records=[("c-" + str(i), i) for i in range(1, 11)])

    r = Parameter(
        m,
        "r",
        domain=row,
        records=[
            [record, 4] if record in row.records["uni"][:7].values else [record, 5]
            for record in row.records["uni"]
        ],
    )
    c = Parameter(
        m,
        "c",
        domain=col,
        records=[
            [record, 3] if record in col.records["uni"][:5].values else [record, 2]
            for record in col.records["uni"]
        ],
    )

    a = Parameter(m, "a", domain=[row, col])

    dyn_col = Set(m, name="dyn_col", domain=[col])
    dyn_col_alias = Alias(m, name="dyn_col_alias", alias_with=dyn_col)
    dyn_col[col] = Ord(col) < 5

    a[row, dyn_col_alias[col]] = 13.2 + r[row] * c[dyn_col_alias]
    assert (
        a.getAssignment()
        == "a(row,dyn_col_alias(col)) = 13.2 + r(row) * c(dyn_col_alias);"
    )

    dyn_col_alias["c-1"] = False
    assert dyn_col_alias.toList() == [f"c-{idx}" for idx in range(2, 5)]


def test_alternative_operation_syntax():
    m = Container()

    i = Set(m)
    j = Set(m)
    a = Set(m, domain=[i, j])
    x = Alias(m, alias_with=a)
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
