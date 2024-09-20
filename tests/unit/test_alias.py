from __future__ import annotations

import os

import pandas as pd
import pytest
from gamspy import (
    Alias,
    Container,
    Equation,
    Ord,
    Parameter,
    Set,
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

    # no alias
    with pytest.raises(TypeError):
        _ = Alias(m)

    # non-str type name
    with pytest.raises(TypeError):
        _ = Alias(m, 5, i)

    # no container
    with pytest.raises(TypeError):
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
    k2 = Alias(m, alias_with=i2)
    assert len(k2) == 2

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
    pytest.raises(TypeError, UniverseAlias)

    # non-container type container
    pytest.raises(TypeError, UniverseAlias, 5, "j")

    # try to create a symbol with same name but different type
    _ = Set(m, "i")
    pytest.raises(TypeError, UniverseAlias, m, "i")

    # get already created symbol
    j1 = UniverseAlias(m, "j")
    j2 = UniverseAlias(m, "j")
    assert id(j1) == id(j2)

    u = UniverseAlias(m, name="u")
    p = Parameter(m, name="p", domain=u)
    p[u] = 2


def test_universe_alias(m):
    gdx_path = os.path.join("tmp", "test.gdx")

    h = UniverseAlias(m, "h")
    assert len(h) == 0
    _ = Set(m, "i", records=["i1", "i2"])
    _ = Set(m, "j", records=["j1", "j2"])

    assert h.records.values.tolist() == [["i1"], ["i2"], ["j1"], ["j2"]]

    assert len(h) == 4

    m.write(gdx_path)

    bla = Container()
    bla.read(gdx_path)
    assert bla.data["h"].records.values.tolist() == h.records.values.tolist()

    m = Container()

    r = UniverseAlias(m, name="new_universe")
    k = Set(m, name="k", domain=r, records="Chicago")
    assert k.getDeclaration() == "Set k(*);"


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
    modified_names = m._get_touched_symbol_names()
    assert modified_names == []


def test_indexing(m):
    row = Set(m, "row", records=[("r-" + str(i), i) for i in range(1, 11)])
    col = Set(m, "col", records=[("c-" + str(i), i) for i in range(1, 11)])

    r = Parameter(
        m,
        "r",
        domain=row,
        records=[
            [record, 4]
            if record in row.records["uni"][:7].values
            else [record, 5]
            for record in row.records["uni"]
        ],
    )
    c = Parameter(
        m,
        "c",
        domain=col,
        records=[
            [record, 3]
            if record in col.records["uni"][:5].values
            else [record, 2]
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
        == "a(row,dyn_col_alias(col)) = (13.2 + (r(row) * c(dyn_col_alias)));"
    )

    dyn_col_alias["c-1"] = False
    assert dyn_col_alias.toList() == [f"c-{idx}" for idx in range(2, 5)]
