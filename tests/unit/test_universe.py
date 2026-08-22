from __future__ import annotations

import pytest

import gamspy as gp

pytestmark = pytest.mark.unit


def test_universe_equality():
    assert gp.UNIVERSE == "*"
    assert "*" == gp.UNIVERSE  # noqa: SIM300
    assert gp.UNIVERSE != "i"
    assert hash(gp.UNIVERSE) == hash("*")
    assert str(gp.UNIVERSE) == "*"
    assert repr(gp.UNIVERSE) == "'*'"
    assert gp.UNIVERSE.gamsRepr() == "*"
    assert gp.UNIVERSE.latexRepr() == "*"


@pytest.mark.parametrize("universe", [gp.UNIVERSE, "*"])
def test_universe_in_declarations(universe):
    m = gp.Container()
    i = gp.Set(m, "i", records=["i1", "i2"])

    s = gp.Set(m, "s", domain=universe)
    p = gp.Parameter(m, "p", domain=[i, universe])
    v = gp.Variable(m, "v", domain=[universe, i])
    e = gp.Equation(m, "e", domain=universe)

    assert p.domain[1] is gp.UNIVERSE
    assert v.domain[0] is gp.UNIVERSE
    assert e.domain[0] is gp.UNIVERSE

    assert s.domain == [gp.UNIVERSE]
    assert s.domain_names == ["*"]
    assert s.getDeclaration() == "Set s(*) / /;"
    assert p.getDeclaration() == "Parameter p(i,*) / /;"
    assert v.getDeclaration() == "free Variable v(*,i) / /;"
    assert e.getDeclaration() == "Equation e(*) / /;"
    m.close()


def test_universe_is_default_set_domain():
    m = gp.Container()
    assert gp.Set(m, "i").domain == [gp.UNIVERSE]
    assert gp.Set(m, "j", domain=gp.UNIVERSE).domain == [gp.UNIVERSE]
    m.close()


def test_universe_survives_serialization(tmp_path):
    m = gp.Container()
    i = gp.Set(m, "i", records=["i1", "i2"])
    _ = gp.Parameter(m, "p", domain=[i, gp.UNIVERSE])

    path = str(tmp_path / "model.zip")
    gp.serialize(m, path)
    m2 = gp.deserialize(path)

    assert m2["p"].domain[1] is gp.UNIVERSE
    m.close()
    m2.close()


def test_universe_alias_as_sole_index_is_rendered():
    m = gp.Container()
    item = gp.UniverseAlias(m, "item")
    i = gp.Set(m, "i", records=["i1", "i2"])
    lim = gp.Set(m, "lim", records=["up", "lo"])

    # A symbol declared over the universe carries no index, but a UniverseAlias
    # used as an index is a named symbol and must be rendered.
    assert lim.gamsRepr() == "lim"
    assert lim[item].gamsRepr() == "lim(item)"
    assert lim[item].latexRepr() == "lim_{item}"
    assert gp.Sum(lim[item], 1).gamsRepr() == "sum(lim(item),1)"
    assert gp.Domain(lim[item], i).gamsRepr() == "(lim(item),i)"
    assert gp.Number(1).where[lim[item]].gamsRepr() == "1 $ (lim(item))"

    lim[item] = True
    assert lim._assignment.getDeclaration() == "lim(item) = yes;"
    assert lim.records["uni"].tolist() == ["i1", "i2", "up", "lo"]
    m.close()
