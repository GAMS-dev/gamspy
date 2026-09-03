from __future__ import annotations

import pandas as pd
import pytest

import gamspy as gp
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


def test_alias_creation(container):
    i = Set(container, "i")
    container.addAlias(alias_with=i)

    a = Alias(container, alias_with=i)
    assert len(a) == 0
    with pytest.raises(ValidationError):
        _ = a.getAssignment()

    # no alias
    with pytest.raises(ValueError):
        _ = Alias(container)

    # non-str type name
    with pytest.raises(TypeError):
        _ = Alias(container, 5, i)

    # no container
    with pytest.raises(ValidationError):
        _ = Alias()

    # non-container type container
    with pytest.raises(TypeError):
        _ = Alias(5, "j", i)

    # try to create a symbol with same name but different type
    with pytest.raises(TypeError):
        _ = Alias(container, "i", i)

    # get already created symbol
    j1 = Alias(container, "j", i)
    j2 = Alias(container, "j", i)
    assert id(j1) == id(j2)

    # len of Alias
    i2 = Set(container, records=["i1", "i2"])
    k2 = Alias(container, "k2", alias_with=i2)
    assert len(k2) == 2

    k2["i1"] = False
    assert k2.getAssignment() == 'k2("i1") = no;'

    with pytest.warns(DeprecationWarning):
        i2.synchronize = True

    with pytest.warns(DeprecationWarning):
        k2.synchronize = True


def test_alias_string(container):
    # Set and Alias without domain
    i = Set(container, name="i", records=["a", "b", "c"])
    j = Alias(container, name="j", alias_with=i)
    assert j.gamsRepr() == "j"
    assert j.getDeclaration() == "Alias(i,j);"

    # Set and Alias with domain
    k = Set(container, name="k", domain=[i], records=["a", "b"])
    l = Alias(container, name="l", alias_with=k)
    assert l.gamsRepr() == "l"
    assert l.getDeclaration() == "Alias(k,l);"

    # Check if the name is reserved
    with pytest.raises(ValidationError):
        _ = Alias(container, "set", i)


def test_override(container):
    # Try to add the same Alias with non-Set alias_with
    u = Set(container, "u")
    v = Alias(container, "v", alias_with=u)
    eq = Equation(container, "eq", domain=[u, v])
    with pytest.raises(ValueError):
        _ = container.addAlias("v", eq)

    # Try to add the same alias
    with pytest.raises(TypeError):
        _ = container.addAlias("u", u)


def test_alias_attributes(container):
    i = Set(container, "i")
    j = Alias(container, "j", alias_with=i)
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


def test_universe_alias_creation(container):
    # non-str type name
    with pytest.raises(TypeError):
        _ = UniverseAlias(container, 5)

    # no container
    with pytest.raises(ValidationError):
        UniverseAlias()

    # non-container type container
    with pytest.raises(TypeError):
        UniverseAlias(5, "j")

    # try to create a symbol with same name but different type
    _ = Set(container, "i")
    with pytest.raises(TypeError):
        UniverseAlias(container, "i")

    # get already created symbol
    j1 = UniverseAlias(container, "j")
    j2 = UniverseAlias(container, "j")
    assert id(j1) == id(j2)

    u = UniverseAlias(container, name="u")
    p = Parameter(container, name="p", domain=u)
    p[u] = 2


def test_universe_alias(container, tmp_path):
    gdx_path = str(tmp_path / "test.gdx")

    h = UniverseAlias(container, "h")
    _ = Set(container, "i", records=["i1", "i2"])
    _ = Set(container, "j", records=["j1", "j2"])

    assert h.toList() == ["i1", "i2", "j1", "j2"]

    container.write(gdx_path)

    bla = Container()
    bla.read(gdx_path)
    assert bla.data["h"].toList() == h.toList()

    container = Container()

    r = UniverseAlias(container, name="new_universe")
    k = Set(container, name="k", domain=r, records="Chicago")
    assert k.getDeclaration() == "Set k(*);"

    u1 = container.addUniverseAlias(name="universe_name")
    assert u1.name == "universe_name"

    u2 = container.addUniverseAlias()
    assert u2.name == "u2"


def test_alias_state(container):
    i = Set(container, name="i", records=["a", "b", "c"])
    j = Alias(container, name="j", alias_with=i)
    i._should_unload_to_gams = False
    j.setRecords(["a", "b"])
    assert not i._should_unload_to_gams

    i._should_unload_to_gams = False
    j.records = pd.DataFrame([["a", "b"]])
    assert i._should_unload_to_gams


def test_alias_modified_list(container):
    nodes = container.addSet("nodes", description="nodes", records=["s"])
    i = container.addAlias("i", nodes)
    _ = container.addSet("s", domain=[i], description="sources", records=["s"])
    symbol_names = container._symbols_to_unload()
    assert symbol_names == []


def test_indexing(container):
    row = Set(container, "row", records=[("r-" + str(i), i) for i in range(1, 11)])
    col = Set(container, "col", records=[("c-" + str(i), i) for i in range(1, 11)])

    r = Parameter(
        container,
        "r",
        domain=row,
        records=[
            [record, 4] if record in row.records["uni"][:7].values else [record, 5]
            for record in row.records["uni"]
        ],
    )
    c = Parameter(
        container,
        "c",
        domain=col,
        records=[
            [record, 3] if record in col.records["uni"][:5].values else [record, 2]
            for record in col.records["uni"]
        ],
    )

    a = Parameter(container, "a", domain=[row, col])

    dyn_col = Set(container, name="dyn_col", domain=[col])
    dyn_col_alias = Alias(container, name="dyn_col_alias", alias_with=dyn_col)
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


def test_universe_records():
    m = Container()
    i = Set(m, "i", domain="*")
    uni = UniverseAlias(m, "uni")
    i["2"].where[False] = False
    assert uni.toList() == ["2"]


def test_universe_alias_addElements():
    m = gp.Container()
    _ = gp.Set(m, "i", records=[f"i{idx}" for idx in range(1, 5)])
    uni = gp.UniverseAlias(m, "uni")
    assert uni.toList() == ["i1", "i2", "i3", "i4"]
    uni.addElements(["i5", "i6"])
    assert uni.toList() == ["i1", "i2", "i3", "i4", "i5", "i6"]
    assert list(m.data.keys()) == ["i", "uni"]


@pytest.mark.unit
def test_alias_redeclaration_does_not_duplicate_declaration(container):
    i = Set(container, "i", records=["i1"])
    j = Alias(container, "j", alias_with=i)

    assert Alias(container, "j", alias_with=i) is j
    assert [s for s in container._unsaved_statements if s is j] == [j]


@pytest.mark.unit
def test_universe_alias_redeclaration_does_not_duplicate_declaration(container):
    u = UniverseAlias(container, "u")

    assert UniverseAlias(container, "u") is u
    assert [s for s in container._unsaved_statements if s is u] == [u]


@pytest.mark.unit
def test_alias_redeclaration_preserves_state(container):
    i = Set(container, "i", records=["i1"])
    j = Alias(container, "j", alias_with=i)
    j._metadata["origin"] = "first declaration"

    _ = Alias(container, "j", alias_with=i)

    assert j._metadata == {"origin": "first declaration"}
    assert j.alias_with is i


@pytest.mark.unit
def test_alias_with_must_be_in_same_container(container):
    other = Container()
    i = Set(other, "i", records=["i1"])

    with pytest.raises(ValidationError):
        _ = Alias(container, "j", alias_with=i)

    with pytest.raises(ValidationError):
        _ = container.addAlias("j", alias_with=i)

    # an Alias in another container is rejected on the symbol that was passed,
    # not on the root set it resolves to
    j = Alias(other, "j", alias_with=i)
    with pytest.raises(ValidationError, match="`j`"):
        _ = Alias(container, "k", alias_with=j)

    other.close()


@pytest.mark.unit
def test_alias_chain_in_same_container_is_allowed(container):
    i = Set(container, "i", records=["i1"])
    a = Alias(container, "a", alias_with=i)
    b = Alias(container, "b", alias_with=a)

    assert a.alias_with is i
    assert b.alias_with is i  # chains resolve to the root Set


@pytest.mark.unit
def test_alias_of_alias_redeclaration_is_idempotent(container):
    i = Set(container, "i", records=["i1"])
    a = Alias(container, "a", alias_with=i)
    b = Alias(container, "b", alias_with=a)

    assert Alias(container, "b", alias_with=a) is b
    # the root Set produces the same `Alias(i,b);` statement, so it is accepted too
    assert Alias(container, "b", alias_with=i) is b
    assert [s for s in container._unsaved_statements if s is b] == [b]


@pytest.mark.unit
def test_alias_redeclaration_rejects_a_different_target(container):
    i = Set(container, "i", records=["i1"])
    k = Set(container, "k", records=["k1"])
    b = Alias(container, "b", alias_with=i)
    c = Alias(container, "c", alias_with=k)

    with pytest.raises(ValueError):
        _ = Alias(container, "b", alias_with=k)

    # an alias resolving to a different root is still a different target
    with pytest.raises(ValueError):
        _ = Alias(container, "b", alias_with=c)

    with pytest.raises(ValueError):
        _ = Alias(container, "b")

    assert b.alias_with is i


@pytest.mark.unit
def test_alias_cannot_alias_itself(container):
    i = Set(container, "i", records=["i1"])
    j = Alias(container, "j", alias_with=i)

    with pytest.raises(ValueError, match="itself"):
        _ = Alias(container, "j", alias_with=j)

    with pytest.raises(ValueError, match="itself"):
        _ = container.addAlias("j", j)
