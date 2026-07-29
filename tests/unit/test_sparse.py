from __future__ import annotations

import os

import pytest

from gamspy import (
    Alias,
    Container,
    Equation,
    Parameter,
    Set,
    Sum,
    Variable,
    deserialize,
    serialize,
    sparse,
)
from gamspy._algebra.expression import split_assignment
from gamspy.exceptions import ValidationError

pytestmark = pytest.mark.unit


@pytest.fixture
def data():
    m = Container()
    i = Set(m, "i", records=["i1", "i2", "i3"])
    a = Parameter(m, "a", domain=i, records=[("i1", 1), ("i3", 3)])
    yield m, i, a
    m.close()


def test_sparse_assignment(data):
    m, i, a = data
    p = Parameter(m, "p", domain=i, records=[("i1", 99), ("i2", 99), ("i3", 99)])

    p[i] = sparse(a[i])

    assert p.getAssignment() == "p(i) $= a(i);"
    # Only the non-zero values of a are assigned, i2 is left untouched.
    assert p.toList() == [("i1", 1.0), ("i2", 99.0), ("i3", 3.0)]


def test_sparse_assignment_equals_condition_on_the_same_expression(data):
    m, i, a = data
    p = Parameter(m, "p", domain=i)
    p2 = Parameter(m, "p2", domain=i)

    p[i] = sparse(a[i] - 1)
    p2[i].where[a[i] - 1] = a[i] - 1

    assert p.toList() == p2.toList()


def test_sparse_assignment_with_condition(data):
    m, i, a = data
    j = Alias(m, "j", i)
    b = Parameter(m, "b", domain=[i, j], records=[("i1", "i2", 5), ("i3", "i1", 7)])
    p = Parameter(m, "p", domain=i)

    p[i].where[a[i] > 1] = sparse(Sum(j, b[i, j]))

    assert p.getAssignment() == "p(i) $ (a(i) > 1) $= sum(j,b(i,j));"
    assert p.toList() == [("i3", 7.0)]


def test_sparse_assignment_scalar(data):
    m, i, a = data
    s = Parameter(m, "s", records=42)

    s[...] = sparse(Sum(i, a[i]) - 4)

    assert s.getAssignment() == "s $= sum(i,a(i)) - 4;"
    # The right hand side is zero, hence the previous value remains.
    assert s.toValue() == 42


def test_sparse_assignment_set(data):
    m, i, a = data
    s = Set(m, "s", domain=i)

    s[i] = sparse(a[i])

    assert s.getAssignment() == "s(i) $= a(i);"
    assert s[i].toList() == ["i1", "i3"]


def test_sparse_assignment_alias(data):
    m, i, a = data
    s = Set(m, "s", domain=i)
    al = Alias(m, "al", s)

    al[i] = sparse(a[i])

    assert al.getAssignment() == "al(i) $= a(i);"
    assert s[i].toList() == ["i1", "i3"]


def test_sparse_assignment_variable_attribute(data):
    m, i, a = data
    x = Variable(m, "x", domain=i)

    x.l[i] = sparse(a[i])

    assert x.getAssignment() == "x.l(i) $= a(i);"
    assert x[i].l.toList() == [("i1", 1.0), ("i3", 3.0)]


def test_sparse_assignment_of_set_membership(data):
    m, i, _ = data
    s = Set(m, "s", domain=i)

    s[i] = sparse(True)

    assert s.getAssignment() == "s(i) $= yes;"
    assert s[i].toList() == ["i1", "i2", "i3"]


def test_sparse_assignment_in_equation_definition(data):
    m, i, a = data
    x = Variable(m, "x", domain=i)
    e = Equation(m, "e", domain=i)

    with pytest.raises(ValidationError):
        e[i] = sparse(x[i] >= a[i])

    with pytest.raises(ValidationError):
        e[i].where[a[i]] = sparse(x[i] >= a[i])

    e[i] = x[i] >= a[i]

    # Equation attributes can be assigned sparsely.
    e.m[i] = sparse(a[i])
    assert e.getAssignment() == "e.m(i) $= a(i);"


def test_sparse_assignment_is_not_operable(data):
    _, i, a = data

    with pytest.raises(TypeError):
        _ = sparse(a[i]) * 2


def test_sparse_assignment_latex_repr(data):
    m, i, a = data
    p = Parameter(m, "p", domain=i)

    p[i] = sparse(a[i])

    assert p._assignment.latexRepr() == "p_{i} \\stackrel{\\$}{=} a_{i}"


def test_sparse_assignment_serialization(data, tmp_path):
    m, i, a = data
    p = Parameter(m, "p", domain=i)
    s = Set(m, "s", domain=i)
    x = Variable(m, "x", domain=i)

    p[i].where[a[i] > 1] = sparse(a[i] * 2)
    s[i] = sparse(a[i])
    x.l[i] = sparse(a[i])

    serialization_path = os.path.join(tmp_path, "sparse.zip")
    serialize(m, serialization_path)
    m2 = deserialize(serialization_path)

    assert m2["p"].getAssignment() == "p(i) $ (a(i) > 1) $= a(i) * 2;"
    assert m2["s"].getAssignment() == "s(i) $= a(i);"
    assert m2["x"].getAssignment() == "x.l(i) $= a(i);"


def test_split_assignment():
    assert split_assignment("p(i) = a(i);") == ("p(i)", "=", "a(i);")
    assert split_assignment("p(i) $= a(i);") == ("p(i)", "$=", "a(i);")
    assert split_assignment("p(i) $ (a(i) > 1) $= a(i) * 2;") == (
        "p(i) $ (a(i) > 1)",
        "$=",
        "a(i) * 2;",
    )

    with pytest.raises(ValidationError):
        split_assignment("e(i) .. x(i) =g= a(i);")
