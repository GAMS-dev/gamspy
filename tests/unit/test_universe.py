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
