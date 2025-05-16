from __future__ import annotations

import pytest

import gamspy as gp
from gamspy import Container, Parameter, Set, Variable

pytestmark = pytest.mark.unit


@pytest.fixture
def data():
    m = Container()
    yield m
    m.close()


def test_parameter_special_values(data):
    m = data
    x = Parameter(m, "x", records=5)
    x[...] = x + gp.SpecialValues.EPS
    assert x.getAssignment() == "x = (x + EPS);"

    i = Set(m, "i", records=["i1", "i2"])

    # Test special values in parameter
    a = Parameter(m, "a", records=gp.SpecialValues.EPS)
    assert a.toValue() == -0.0

    b = Parameter(m, "b", domain=[i])
    b[...] = gp.SpecialValues.EPS

    assert b.getAssignment() == "b(i) = EPS;"

    b[...] = gp.SpecialValues.NA
    assert b.getAssignment() == "b(i) = NA;"

    b[...] = gp.SpecialValues.UNDEF
    assert b.getAssignment() == "b(i) = UNDF;"

    b[...] = gp.SpecialValues.POSINF
    assert b.getAssignment() == "b(i) = INF;"

    b[...] = gp.SpecialValues.NEGINF
    assert b.getAssignment() == "b(i) = -INF;"


def test_implicit_parameter_special_values(data):
    m = data
    i = Set(m, "i", records=["i1", "i2"])

    b = Variable(m, "b", domain=[i])
    b.l[...] = gp.SpecialValues.EPS

    assert b.getAssignment() == "b.l(i) = EPS;"

    b.l[...] = gp.SpecialValues.NA
    assert b.getAssignment() == "b.l(i) = NA;"

    b.l[...] = gp.SpecialValues.UNDEF
    assert b.getAssignment() == "b.l(i) = UNDF;"

    b.l[...] = gp.SpecialValues.POSINF
    assert b.getAssignment() == "b.l(i) = INF;"

    b.l[...] = gp.SpecialValues.NEGINF
    assert b.getAssignment() == "b.l(i) = -INF;"


def test_operation_special_values(data):
    m = data
    tax = Set(m, "tax", records=["i1", "i2"])
    bla = Set(m, "bla", records=["x"])
    results = Parameter(m, "results", domain=[tax, bla])

    x = Variable(m, "x", domain=tax)
    e = Variable(m, "e", domain=tax)
    results[tax, "x"] = gp.math.Max(x.l[tax] - e.l[tax], gp.SpecialValues.EPS)

    assert (
        results.getAssignment()
        == 'results(tax,"x") = max((x.l(tax) - e.l(tax)),EPS);'
    )

    dummy = Parameter(m, "dummy")
    i = Set(m, "i", records=["i1", "i2"])
    dummy[...] = gp.Sum(i, -0.0)
    assert dummy.getAssignment() == "dummy = sum(i,EPS);"


def test_eps(data):
    m = data
    i = Set(m, "i", records=[f"i{i}" for i in range(11)])
    f = Parameter(
        m,
        "f",
        domain=i,
        records=[("i0", gp.SpecialValues.EPS), ("i1", 1)],
    )

    assert f.toList() == [("i0", -0.0), ("i1", 1.0)]
