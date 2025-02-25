from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

import gamspy as gp
from gamspy import Alias, Container, Parameter, Set, Sum, Variable
from gamspy.exceptions import ValidationError
from gamspy.math import dim, permute, trace, vector_norm

pytestmark = pytest.mark.unit


@pytest.fixture
def data():
    m = Container()
    yield m
    m.close()


def test_matrix_mult_bad(data):
    m = data
    i = Set(m, name="i", records=["i1", "i2", "i3"])
    j = Set(m, name="j", records=["j1", "j2", "j3"])

    a_recs = np.random.randint(1, 11, size=(3))
    b_recs = np.random.randint(1, 11, size=(3))
    a = Parameter(m, name="a", domain=[i], records=a_recs, uels_on_axes=True)
    b = Parameter(m, name="b", domain=[j], records=b_recs, uels_on_axes=True)
    c = Parameter(m, name="c", domain=[], uels_on_axes=True)

    pytest.raises(ValidationError, lambda: c @ a)
    pytest.raises(ValidationError, lambda: a @ c)
    pytest.raises(ValidationError, lambda: a @ b)


def test_simple_matrix_matrix(data):
    m = data
    """Test simple case where domain calculation is trivial
    matrix x matrix"""
    i = Set(m, name="i", records=["i1", "i2", "i3"])
    j = Set(m, name="j", records=["j1", "j2", "j3"])
    k = Set(m, name="k", records=["k1", "k2", "k3"])

    a_recs = np.random.randint(1, 11, size=(3, 3))
    b_recs = np.random.randint(1, 11, size=(3, 3))
    a = Parameter(
        m, name="a", domain=[i, j], records=a_recs, uels_on_axes=True
    )
    b = Parameter(
        m, name="b", domain=[j, k], records=b_recs, uels_on_axes=True
    )
    c = a @ b
    assert c.domain == [i, k]

    c = Parameter(m, name="c", domain=[i, k])
    c[i, k] = a @ b
    c_recs = c.toDense()
    assert np.allclose(c_recs, a_recs @ b_recs)

    a2 = Parameter(
        m, name="a2", domain=[k, k], records=a_recs, uels_on_axes=True
    )
    # dims do not match
    pytest.raises(ValidationError, lambda: a2 @ b)


def test_simple_matrix_vector(data):
    m = data
    """Test simple case where domain calculation is trivial
    matrix x vector"""
    i = Set(m, name="i", records=["i1", "i2", "i3"])
    j = Set(m, name="j", records=["j1", "j2", "j3"])

    a_recs = np.random.randint(1, 11, size=(3, 3))
    b_recs = np.random.randint(1, 11, size=(3))
    a = Parameter(
        m, name="a", domain=[i, j], records=a_recs, uels_on_axes=True
    )
    b = Parameter(m, name="b", domain=[j], records=b_recs, uels_on_axes=True)
    c = a @ b
    assert c.domain == [i]
    c = Parameter(m, name="c", domain=[i])
    c[i] = a @ b
    c_recs = c.toDense()
    assert np.allclose(c_recs, a_recs @ b_recs)

    a2 = Parameter(
        m, name="a2", domain=[j, i], records=a_recs, uels_on_axes=True
    )
    # dims do not match
    pytest.raises(ValidationError, lambda: a2 @ b)


def test_simple_vector_vector(data):
    m = data
    """Test simple case where domain calculation is trivial
    vector x vector, aka inner product"""
    i = Set(m, name="i", records=["i1", "i2", "i3"])

    a_recs = np.random.randint(1, 11, size=(3))
    b_recs = np.random.randint(1, 11, size=(3))
    a = Parameter(m, name="a", domain=[i], records=a_recs, uels_on_axes=True)
    b = Parameter(m, name="b", domain=[i], records=b_recs, uels_on_axes=True)
    c = a @ b
    assert c.domain == []
    c = Parameter(m, name="c", domain=[])
    c[...] = a @ b
    c_recs = c.toDense()
    assert np.allclose(c_recs, a_recs @ b_recs)


def test_vector_vector_with_conflicting_sum_domain(data):
    m = data
    i = Set(m, name="i", records=["i1", "i2", "i3"])
    j = Alias(m, name="j", alias_with=i)

    x_recs = np.random.randint(1, 11, size=(3, 3))
    y_recs = np.random.randint(1, 11, size=(3))
    x = Variable(m, name="x", domain=[j, i], records=x_recs, uels_on_axes=True)
    y = Parameter(m, name="y", domain=[i], records=y_recs, uels_on_axes=True)

    right_side = x @ y  # has controlled_domain domain of i
    assert right_side.controlled_domain == [i]

    z = Parameter(m, name="z", domain=[i])
    val = z @ right_side
    assert val.domain == []
    assert len(val.controlled_domain) == 2
    assert val.op_domain[0].name == "AliasOfi_2"


def test_simple_vector_matrix(data):
    m = data
    """Test simple case where domain calculation is trivial
    vector x matrix"""
    i = Set(m, name="i", records=["i1", "i2", "i3"])
    j = Set(m, name="j", records=["j1", "j2", "j3"])

    a_recs = np.random.randint(1, 11, size=(3))
    b_recs = np.random.randint(1, 11, size=(3, 3))
    a = Parameter(m, name="a", domain=[i], records=a_recs, uels_on_axes=True)
    b = Parameter(
        m, name="b", domain=[i, j], records=b_recs, uels_on_axes=True
    )
    c = a @ b
    assert c.domain == [j]
    c = Parameter(m, name="c", domain=[j])
    c[...] = a @ b
    c_recs = c.toDense()
    assert np.allclose(c_recs, a_recs @ b_recs)

    a2 = Parameter(m, name="a2", domain=[j], records=a_recs, uels_on_axes=True)

    pytest.raises(ValidationError, lambda: a2 @ b)


def test_batched_matrix_matrix(data):
    m = data
    """Test batched matrix multiplication,
    batched matrix x batched matrix"""
    n = Set(m, name="n", records=["n1", "n2", "n3", "n4"])
    i = Set(m, name="i", records=["i1", "i2", "i3"])
    j = Set(m, name="j", records=["j1", "j2", "j3"])
    k = Set(m, name="k", records=["k1", "k2", "k3"])

    a_recs = np.random.randint(1, 11, size=(4, 3, 3))
    b_recs = np.random.randint(1, 11, size=(4, 3, 3))
    a = Parameter(
        m,
        name="a",
        domain=[n, i, j],
        records=a_recs,
        uels_on_axes=True,
    )
    b = Parameter(
        m,
        name="b",
        domain=[n, j, k],
        records=b_recs,
        uels_on_axes=True,
    )
    c = a @ b
    assert c.domain == [n, i, k]
    c = Parameter(m, name="c", domain=[n, i, k])
    c[...] = a @ b
    c_recs = c.toDense()
    assert np.allclose(c_recs, a_recs @ b_recs)

    a2 = Parameter(
        m,
        name="a2",
        domain=[n, j, i],
        records=a_recs,
        uels_on_axes=True,
    )
    pytest.raises(ValidationError, lambda: a2 @ b)


def test_batched_matrix_vector(data):
    m = data
    """Test batched matrix - vector multiplication,
    batched matrix x vector"""
    n = Set(m, name="n", records=["n1", "n2", "n3", "n4"])
    i = Set(m, name="i", records=["i1", "i2", "i3"])
    j = Set(m, name="j", records=["j1", "j2", "j3"])

    a_recs = np.random.randint(1, 11, size=(4, 3, 3))
    b_recs = np.random.randint(1, 11, size=(3))
    a = Parameter(
        m,
        name="a",
        domain=[n, i, j],
        records=a_recs,
        uels_on_axes=True,
    )
    b = Parameter(
        m,
        name="b",
        domain=[j],
        records=b_recs,
        uels_on_axes=True,
    )
    c = a @ b
    assert c.domain == [n, i]
    c = Parameter(m, name="c", domain=[n, i])
    c[...] = a @ b
    c_recs = c.toDense()
    assert np.allclose(c_recs, a_recs @ b_recs)

    a2 = Parameter(
        m,
        name="a2",
        domain=[n, j, i],
        records=a_recs,
        uels_on_axes=True,
    )
    pytest.raises(ValidationError, lambda: a2 @ b)


def test_batched_matrix_matrix_2(data):
    m = data
    """Test batched matrix multiplication,
    batched matrix x matrix"""
    n = Set(m, name="n", records=["n1", "n2", "n3", "n4"])
    i = Set(m, name="i", records=["i1", "i2", "i3"])
    j = Set(m, name="j", records=["j1", "j2", "j3"])
    k = Set(m, name="k", records=["k1", "k2", "k3"])

    a_recs = np.random.randint(1, 11, size=(4, 3, 3))
    b_recs = np.random.randint(1, 11, size=(3, 3))
    a = Parameter(
        m,
        name="a",
        domain=[n, i, j],
        records=a_recs,
        uels_on_axes=True,
    )
    b = Parameter(
        m, name="b", domain=[j, k], records=b_recs, uels_on_axes=True
    )
    c = a @ b
    assert c.domain == [n, i, k]
    c = Parameter(m, name="c", domain=[n, i, k])
    c[...] = a @ b
    c_recs = c.toDense()
    assert np.allclose(c_recs, a_recs @ b_recs)


def test_vector_batched_matrix(data):
    m = data
    """Test vector x batched_matrix"""
    n = Set(m, name="n", records=["n1", "n2", "n3", "n4"])
    i = Set(m, name="i", records=["i1", "i2", "i3"])
    j = Set(m, name="j", records=["j1", "j2", "j3"])

    a_recs = np.random.randint(1, 11, size=(4, 3, 3))
    b_recs = np.random.randint(1, 11, size=(3))
    a = Parameter(
        m,
        name="a",
        domain=[n, i, j],
        records=a_recs,
        uels_on_axes=True,
    )
    b = Parameter(m, name="b", domain=[i], records=b_recs, uels_on_axes=True)
    c = b @ a

    assert c.domain == [n, j]
    c = Parameter(m, name="c", domain=[n, j])
    c[...] = b @ a
    c_recs = c.toDense()
    assert np.allclose(c_recs, b_recs @ a_recs)

    b2 = Parameter(m, name="b2", domain=[j], records=b_recs, uels_on_axes=True)
    pytest.raises(ValidationError, lambda: b2 @ a)


def test_square_matrix_mult(data):
    m = data
    i = Set(m, name="i", records=["i1", "i2", "i3"])
    j = Alias(m, name="j", alias_with=i)
    k = Alias(m, name="k", alias_with=j)

    a_recs = np.random.randint(1, 11, size=(3, 3))
    b_recs = np.random.randint(1, 11, size=(3, 3))
    a = Parameter(
        m, name="a", domain=[i, i], records=a_recs, uels_on_axes=True
    )
    b = Parameter(
        m, name="b", domain=[i, i], records=b_recs, uels_on_axes=True
    )

    c2 = a[i, j] @ b[j, k]
    assert c2.domain == [i, k]
    c = Parameter(m, name="c", domain=[i, k])
    # reindexing is required in this case
    c[...] = (a @ b)[i, k]
    c_recs = c.toDense()
    assert np.allclose(c_recs, a_recs @ b_recs)


def test_square_matrix_mult_2(data):
    m = data
    n = Set(m, name="n", records=["n1", "n2", "n3", "n4"])
    i = Set(m, name="i", records=["i1", "i2", "i3"])
    j = Alias(m, name="j", alias_with=i)
    k = Alias(m, name="k", alias_with=j)

    a_recs = np.random.randint(1, 11, size=(4, 3, 3))
    b_recs = np.random.randint(1, 11, size=(4, 3, 3))
    a = Parameter(
        m,
        name="a",
        domain=[n, i, i],
        records=a_recs,
        uels_on_axes=True,
    )
    b = Parameter(
        m,
        name="b",
        domain=[n, i, i],
        records=b_recs,
        uels_on_axes=True,
    )

    c2 = a[n, i, j] @ b[n, j, k]
    assert c2.domain == [n, i, k]
    c = Parameter(m, name="c", domain=[n, i, k])
    c[...] = (a[n, i, j] @ b[n, j, k])[n, i, k]
    c_recs = c.toDense()
    assert np.allclose(c_recs, a_recs @ b_recs)


def test_square_matrix_mult_3(data):
    m = data
    i = Set(m, name="i", records=["i1", "i2", "i3"])
    j = Alias(m, name="j", alias_with=i)

    a_recs = np.random.randint(1, 11, size=(3, 3))
    b_recs = np.random.randint(1, 11, size=(3, 3))
    a = Parameter(
        m,
        name="a",
        domain=[i, i],
        records=a_recs,
        uels_on_axes=True,
    )
    b = Parameter(
        m,
        name="b",
        domain=[i, i],
        records=b_recs,
        uels_on_axes=True,
    )

    c2 = (a @ b) @ (a @ b)
    c = Parameter(m, name="c", domain=[i, i])
    c[i, j] = c2[i, j]
    c_recs = c.toDense()
    assert np.allclose(c_recs, (a_recs @ b_recs) @ (a_recs @ b_recs))


def test_square_matrix_mult_4(data):
    m = data
    a_recs = np.random.randint(1, 11, size=(3, 3))
    b_recs = np.random.randint(1, 11, size=(3, 3))
    a = Parameter(
        m,
        name="a",
        domain=dim([3, 3]),
        records=a_recs,
        uels_on_axes=True,
    )
    b = Parameter(
        m,
        name="b",
        domain=dim([3, 3]),
        records=b_recs,
        uels_on_axes=True,
    )

    c2 = (a @ b) + a
    c = Parameter(m, name="c", domain=dim([3, 3]))
    c[...] = c2
    c_recs = c.toDense()
    assert np.allclose(c_recs, (a_recs @ b_recs) + (a_recs))


def test_batch_size_matches(data):
    m = data
    n = Set(m, name="n", records=["n1", "n2", "n3"])
    z = Set(m, name="z", records=["m1", "m2", "m3"])
    i = Set(m, name="i", records=["i1", "i2", "i3"])
    j = Alias(m, name="j", alias_with=i)
    k = Alias(m, name="k", alias_with=j)

    a = Parameter(m, name="a", domain=[n, i, j])
    b = Parameter(m, name="b", domain=[z, j, k])
    c = Parameter(m, name="c", domain=[n, z, j, k])

    pytest.raises(ValidationError, lambda: a @ b)
    pytest.raises(ValidationError, lambda: a @ c)


def test_domain_conflict_resolution(data):
    m = data
    n = Set(m, name="n", records=["n1", "n2", "n3"])
    i = Set(m, name="i", records=["i1", "i2", "i3"])

    vec = Parameter(m, name="vec", domain=[i])
    mat = Parameter(m, name="mat", domain=[i, i])
    batched_mat = Parameter(m, name="batched_mat", domain=[n, i, i])

    r1 = vec @ vec
    assert r1.domain == []

    r2 = mat @ mat
    assert len(r2.domain) == 2
    assert len(r2.controlled_domain) == 1
    assert r2.domain[0] != r2.domain[1]

    r3 = vec @ mat
    assert len(r3.domain) == 1
    assert len(r3.controlled_domain) == 1

    r4 = mat @ vec
    assert len(r4.domain) == 1
    assert len(r4.controlled_domain) == 1

    r5 = vec @ batched_mat
    assert len(r5.domain) == 2
    assert len(r5.controlled_domain) == 1
    assert r5.domain[0] == batched_mat.domain[0]

    r6 = batched_mat @ vec
    assert len(r6.domain) == 2
    assert len(r6.controlled_domain) == 1
    assert r6.domain[0] == batched_mat.domain[0]

    r7 = batched_mat @ batched_mat
    assert len(r7.domain) == 3
    assert len(r7.controlled_domain) == 1
    assert r7.domain[-1] != r7.domain[-2]


def test_domain_conflict_resolution_2(data):
    m = data
    vec = Parameter(m, name="vec", domain=dim([3]))
    mat = Parameter(m, name="mat", domain=dim([3, 3]))
    batched_mat = Parameter(m, name="batched_mat", domain=dim([3, 3, 3]))

    r2 = mat @ mat
    assert len(r2.domain) == 2
    assert r2.domain[0] == mat.domain[0]
    assert r2.domain[1] == mat.domain[1]

    mat2 = Parameter(m, name="mat2", domain=[mat.domain[1], mat.domain[0]])
    r3 = vec @ mat2
    assert len(r3.domain) == 1
    assert r3.domain[0] == mat2.domain[1]

    r3 = vec @ mat
    assert len(r3.domain) == 1
    assert r3.domain[0] == mat.domain[1]

    r4 = mat @ vec
    assert len(r4.domain) == 1
    assert r4.domain[0] == mat.domain[0]

    r4 = mat2 @ vec
    assert len(r4.domain) == 1
    assert r4.domain[0] == mat2.domain[0]

    # Added an exception to make this one work
    r5 = vec @ batched_mat
    assert len(r5.domain) == 2
    assert len(r5.controlled_domain) == 1
    assert r5.domain[0] == batched_mat.domain[0]

    # This one does not work
    # r6 = batched_mat @ vec
    # assert(len(r6.domain)== 2)
    # assert(len(r6.controlled_domain)== 1)
    # assert(r6.domain[0]== batched_mat.domain[0])

    r7 = batched_mat @ batched_mat
    assert len(r7.domain) == 3
    assert r7.domain[0] == batched_mat.domain[0]
    assert r7.domain[1] == batched_mat.domain[1]
    assert r7.domain[2] == batched_mat.domain[2]


def test_trace_on_matrix(data):
    m = data
    identity = np.eye(3, 3)
    mat = Parameter(
        m,
        name="mat",
        domain=dim([3, 3]),
        records=identity,
        uels_on_axes=True,
    )

    trace_expr = trace(mat)
    assert trace_expr.domain == []
    sc = Parameter(m, name="sc", domain=[])
    sc[...] = trace(mat)
    assert np.trace(identity) == sc.toDense()

    rand_recs = np.random.randint(1, 11, size=(3, 3))
    mat.setRecords(rand_recs, uels_on_axes=True)
    sc[...] = trace(mat)
    assert np.trace(rand_recs) == sc.toDense()

    recs = np.ones((3, 5))
    rect = Parameter(
        m,
        name="rect",
        domain=dim([3, 5]),
        records=recs,
        uels_on_axes=True,
    )

    pytest.raises(ValidationError, lambda: trace(rect))


def test_trace_on_vector(data):
    m = data
    vec = Parameter(m, name="vec", domain=dim([3]))
    pytest.raises(ValidationError, lambda: trace(vec))


def test_trace_on_batched_matrix(data):
    m = data
    recs = np.random.randint(1, 11, size=(128, 3, 3))
    bm1 = Parameter(
        m,
        name="vec",
        domain=dim([128, 3, 3]),
        records=recs,
        uels_on_axes=True,
    )

    sc1 = Parameter(m, name="sc1", domain=dim([128]))
    expr1 = trace(bm1, axis1=1, axis2=2)
    assert expr1.domain[0].name == "DenseDim128_1"

    sc1[...] = expr1
    sc1_dens = sc1.toDense()
    assert np.allclose(sc1_dens, np.trace(recs, axis1=1, axis2=2))


def test_domain_relabeling(data):
    m = data
    n = Set(m, name="n", records=["n1", "n2", "n3", "n4"])
    i = Set(m, name="i", records=["i1", "i2", "i3"])
    i2 = Alias(m, name="i2", alias_with=i)
    j = Set(m, name="j", records=["j1", "j2", "j3"])
    k = Set(m, name="k", records=["k1", "k2", "k3"])
    k2 = Alias(m, name="k2", alias_with=k)

    a = Variable(m, name="a", domain=[n, i, j])
    b = Parameter(m, name="b", domain=[j, k])

    expr = a + b
    expr2 = expr[n, i2, j, k]
    assert expr.domain == [n, i, j, k]
    assert expr.gamsRepr() == "(a(n,i,j) + b(j,k))"
    assert expr2.domain == [n, i2, j, k]
    assert expr2.gamsRepr() == "(a(n,i2,j) + b(j,k))"
    pytest.raises(ValidationError, lambda: expr[n])

    expr3 = (a + a) + (b + b)
    assert expr3.gamsRepr() == "((a(n,i,j) + a(n,i,j)) + (b(j,k) + b(j,k)))"
    expr4 = expr3[n, i2, j, k]
    assert expr4.gamsRepr() == "((a(n,i2,j) + a(n,i2,j)) + (b(j,k) + b(j,k)))"

    expr5 = Sum(n, expr3)
    assert (
        expr5.gamsRepr()
        == "sum(n,((a(n,i,j) + a(n,i,j)) + (b(j,k) + b(j,k))))"
    )

    expr6 = expr5[i2, j, k]
    assert (
        expr6.gamsRepr()
        == "sum(n,((a(n,i2,j) + a(n,i2,j)) + (b(j,k) + b(j,k))))"
    )

    expr7 = Sum(j, expr6)
    assert (
        expr7.gamsRepr()
        == "sum(j,sum(n,((a(n,i2,j) + a(n,i2,j)) + (b(j,k) + b(j,k)))))"
    )

    expr8 = expr7[i, k2]
    assert (
        expr8.gamsRepr()
        == "sum(j,sum(n,((a(n,i,j) + a(n,i,j)) + (b(j,k2) + b(j,k2)))))"
    )


def test_vector_norm_not_implemented(data):
    m = data
    i = Set(m, name="i", records=["i1", "i2"])
    a = Variable(m, name="a", domain=[i])
    pytest.raises(ValidationError, lambda: vector_norm(a, ord=0))
    pytest.raises(ValidationError, lambda: vector_norm(a, ord=float("inf")))
    pytest.raises(ValidationError, lambda: vector_norm(a, ord=float("-inf")))


def test_vector_norm(data):
    m = data
    i = Set(m, name="i", records=["i1", "i2"])
    b = Parameter(m, name="b", domain=[i], records=[("i1", 3), ("i2", 4)])
    c = Parameter(m, name="c")

    n_expr = vector_norm(b, ord=2)
    c[...] = n_expr
    c_val = c.records.iloc[0, 0]
    assert math.isclose(c_val, 5, rel_tol=1e-4)

    n_expr = vector_norm(b, ord=2.0)
    c[...] = n_expr
    c_val = c.records.iloc[0, 0]
    assert math.isclose(c_val, 5, rel_tol=1e-4)

    # this is a special case
    norm_squared = n_expr**2
    assert isinstance(norm_squared, gp._algebra.operation.Operation)

    c[...] = vector_norm(b, ord=3)
    c_val = c.records.iloc[0, 0]
    assert math.isclose(c_val, 4.49794, rel_tol=1e-4)

    c[...] = vector_norm(b, ord=4)
    c_val = c.records.iloc[0, 0]
    assert math.isclose(c_val, 4.28457, rel_tol=1e-4)

    c[...] = vector_norm(b, ord=1)
    c_val = c.records.iloc[0, 0]
    assert math.isclose(c_val, 7.0, rel_tol=1e-4)


def test_vector_norm_2(data):
    m = data
    i = Set(m, name="i", records=["i1", "i2"])
    n = Set(m, name="n", records=["n1", "n2", "n3"])
    b = Parameter(
        m,
        name="b",
        domain=[n, i],
        records=[
            ("n1", "i1", 3),
            ("n1", "i2", 4),
            ("n2", "i1", 7),
            ("n2", "i2", 24),
            ("n3", "i1", 5),
            ("n3", "i2", 12),
        ],
    )
    c = Parameter(m, name="c", domain=[n])

    c[n] = vector_norm(b, dim=[1])
    assert math.isclose(c.records.iloc[0, 1], 5, rel_tol=1e-5)
    assert math.isclose(c.records.iloc[1, 1], 25, rel_tol=1e-5)
    assert math.isclose(c.records.iloc[2, 1], 13, rel_tol=1e-5)

    c[n] = vector_norm(b, dim=[i])
    assert math.isclose(c.records.iloc[0, 1], 5, rel_tol=1e-5)
    assert math.isclose(c.records.iloc[1, 1], 25, rel_tol=1e-5)
    assert math.isclose(c.records.iloc[2, 1], 13, rel_tol=1e-5)


def test_vector_norm_3(data):
    m = data
    i = Set(m, name="i", records=["i1", "i2"])
    n = Set(m, name="n", records=["n1", "n2"])
    a = Variable(m, name="a", domain=[n, i])

    assert vector_norm(a[:, "i1"]).domain == []
    assert vector_norm(a["n1", :]).domain == []
    pytest.raises(ValidationError, lambda: vector_norm(a["n1", "i1"]))


def test_vector_norm_dim(data):
    m = data
    i = Set(m, name="i", records=["i1", "i2"])
    a = Variable(m, name="a", domain=[i])
    pytest.raises(ValidationError, lambda: vector_norm(a, dim="asd"))
    pytest.raises(ValidationError, lambda: vector_norm(a, dim=[]))
    pytest.raises(ValidationError, lambda: vector_norm(a, dim=[0, i]))
    pytest.raises(ValidationError, lambda: vector_norm(a, dim=["asd"]))
    pytest.raises(ValidationError, lambda: vector_norm(a, dim=2))


def test_literal_indexing(data):
    m = data
    i = Set(m, name="i", records=["i1", "i2"])
    n = Set(m, name="n", records=["n1", "n2", "n3"])
    a = Variable(m, name="a", domain=[n, i])
    b = Parameter(m, name="b", domain=[n, i])

    # try simple case
    a_1 = a[:, "i1"]
    assert a_1.domain == [n]
    assert a_1.gamsRepr() == 'a(n,"i1")'

    # try simple case
    a_2 = a["n1", :]
    assert a_2.domain == [i]
    assert a_2.gamsRepr() == 'a("n1",i)'

    a_3 = a["n1", "i1"]
    assert a_3.domain == []
    assert a_3.gamsRepr() == 'a("n1","i1")'

    a_4 = a[:, "i1"]["n1"]
    assert a_4.domain == []
    assert a_4.gamsRepr() == 'a("n1","i1")'

    a_5 = a["n1", :]["i1"]
    assert a_5.domain == []
    assert a_5.gamsRepr() == 'a("n1","i1")'

    a_6 = -a["n1", :]
    assert a_6.domain == [i]
    assert a_6.gamsRepr() == '( - a("n1",i))'

    # try simple case
    b_1 = b[:, "i1"]
    assert b_1.domain == [n]
    assert b_1.gamsRepr() == 'b(n,"i1")'

    # try simple case
    b_2 = b["n1", :]
    assert b_2.domain == [i]
    assert b_2.gamsRepr() == 'b("n1",i)'

    b_3 = b["n1", "i1"]
    assert b_3.domain == []
    assert b_3.gamsRepr() == 'b("n1","i1")'

    b_4 = b[:, "i1"]["n1"]
    assert b_4.domain == []
    assert b_4.gamsRepr() == 'b("n1","i1")'

    b_5 = b["n1", :]["i1"]
    assert b_5.domain == []
    assert b_5.gamsRepr() == 'b("n1","i1")'

    b_6 = -b["n1", :]
    assert b_6.domain == [i]
    assert b_6.gamsRepr() == '( - b("n1",i))'


def test_literal_indexing_mix_permute_variable(data):
    m = data
    i = Set(m, name="i", records=["i1", "i2"])
    n = Set(m, name="n", records=["n1", "n2", "n3"])
    n2 = Set(m, name="n2", records=["n1", "n2", "n3"])
    a = Variable(m, name="a", domain=[n, n2, i])

    a_1 = a[:, :, "i1"].t()
    assert a_1.domain == [n2, n]
    assert a_1.gamsRepr() == 'a(n,n2,"i1")'

    a_1_2 = a.t()[:, "i1", :]
    assert a_1_2.domain == [n, n2]
    assert a_1_2.gamsRepr() == 'a(n,n2,"i1")'

    a_2 = a[:, "n1", :].t()
    assert a_2.domain == [i, n]
    assert a_2.gamsRepr() == 'a(n,"n1",i)'

    a_2_2 = a.t()[:, :, "n1"]
    assert a_2_2.domain == [n, i]
    assert a_2_2.gamsRepr() == 'a(n,"n1",i)'

    a_3 = a[:, "n1", "i1"]
    assert a_3.domain == [n]
    assert a_3.gamsRepr() == 'a(n,"n1","i1")'

    a_3_2 = a.t()[:, "i1", "n1"]
    assert a_3_2.domain == [n]
    assert a_3_2.gamsRepr() == 'a(n,"n1","i1")'

    a_4 = a[:, :, "i1"][:, "n1"]
    assert a_4.domain == [n]
    assert a_4.gamsRepr() == 'a(n,"n1","i1")'

    a_4_2 = a.t()[:, "i1", :][:, "n1"]
    assert a_4_2.domain == [n]
    assert a_4_2.gamsRepr() == 'a(n,"n1","i1")'

    a_4_3 = a[:, :, "i1"].t()[:, "n1"]
    assert a_4_3.domain == [n2]
    assert a_4_3.gamsRepr() == 'a("n1",n2,"i1")'

    a_5 = a[:, "n1", :][:, "i1"]
    assert a_5.domain == [n]
    assert a_5.gamsRepr() == 'a(n,"n1","i1")'

    a_5_2 = a.t()[:, :, "n1"][:, "i1"]
    assert a_5_2.domain == [n]
    assert a_5_2.gamsRepr() == 'a(n,"n1","i1")'

    a_5_3 = a[:, "n1", :].t()["i1", :]
    assert a_5_3.domain == [n]
    assert a_5_3.gamsRepr() == 'a(n,"n1","i1")'

    # a_6 = (-(a[:, "n1", :])).t()
    # assert(a_6.domain == [i== n])
    # assert(a_6.gamsRepr() == '-a(n,"n1",i)')

    a_6_2 = -((a[:, "n1", :]).t())
    assert a_6_2.domain == [i, n]
    assert a_6_2.gamsRepr() == '( - a(n,"n1",i))'

    # a_6_3 = ((-a)[:, "n1", :]).t()
    # assert(a_6_3.domain == [i== n])
    # assert(a_6_3.gamsRepr() == '-a(n,"n1",i)')

    # a_6_4 = ((-a).t())[:, :, "n1"]
    # assert(a_6_4.domain == [n== i])
    # assert(a_6_4.gamsRepr() == '-a(n,"n1",i)')

    a_6_5 = (-a.t())[:, :, "n1"]
    assert a_6_5.domain == [n, i]
    assert a_6_5.gamsRepr() == '( - a(n,"n1",i))'

    a_6_6 = -((a.t())[:, :, "n1"])
    assert a_6_6.domain == [n, i]
    assert a_6_6.gamsRepr() == '( - a(n,"n1",i))'


def test_literal_indexing_mix_permute_parameter(data):
    m = data
    i = Set(m, name="i", records=["i1", "i2"])
    n = Set(m, name="n", records=["n1", "n2", "n3"])
    n2 = Set(m, name="n2", records=["n1", "n2", "n3"])
    a = Parameter(m, name="a", domain=[n, n2, i])

    a_1 = a[:, :, "i1"].t()
    assert a_1.domain == [n2, n]
    assert a_1.gamsRepr() == 'a(n,n2,"i1")'

    a_1_2 = a.t()[:, "i1", :]
    assert a_1_2.domain == [n, n2]
    assert a_1_2.gamsRepr() == 'a(n,n2,"i1")'

    a_2 = a[:, "n1", :].t()
    assert a_2.domain == [i, n]
    assert a_2.gamsRepr() == 'a(n,"n1",i)'

    a_2_2 = a.t()[:, :, "n1"]
    assert a_2_2.domain == [n, i]
    assert a_2_2.gamsRepr() == 'a(n,"n1",i)'

    a_3 = a[:, "n1", "i1"]
    assert a_3.domain == [n]
    assert a_3.gamsRepr() == 'a(n,"n1","i1")'

    a_3_2 = a.t()[:, "i1", "n1"]
    assert a_3_2.domain == [n]
    assert a_3_2.gamsRepr() == 'a(n,"n1","i1")'

    a_4 = a[:, :, "i1"][:, "n1"]
    assert a_4.domain == [n]
    assert a_4.gamsRepr() == 'a(n,"n1","i1")'

    a_4_2 = a.t()[:, "i1", :][:, "n1"]
    assert a_4_2.domain == [n]
    assert a_4_2.gamsRepr() == 'a(n,"n1","i1")'

    a_4_3 = a[:, :, "i1"].t()[:, "n1"]
    assert a_4_3.domain == [n2]
    assert a_4_3.gamsRepr() == 'a("n1",n2,"i1")'

    a_5 = a[:, "n1", :][:, "i1"]
    assert a_5.domain == [n]
    assert a_5.gamsRepr() == 'a(n,"n1","i1")'

    a_5_2 = a.t()[:, :, "n1"][:, "i1"]
    assert a_5_2.domain == [n]
    assert a_5_2.gamsRepr() == 'a(n,"n1","i1")'

    a_5_3 = a[:, "n1", :].t()["i1", :]
    assert a_5_3.domain == [n]
    assert a_5_3.gamsRepr() == 'a(n,"n1","i1")'

    # a_6 = (-(a[:, "n1", :])).t()
    # assert(a_6.domain == [i== n])
    # assert(a_6.gamsRepr() == '-a(n,"n1",i)')

    a_6_2 = -((a[:, "n1", :]).t())
    assert a_6_2.domain == [i, n]
    assert a_6_2.gamsRepr() == '( - a(n,"n1",i))'

    # a_6_3 = ((-a)[:, "n1", :]).t()
    # assert(a_6_3.domain == [i== n])
    # assert(a_6_3.gamsRepr() == '-a(n,"n1",i)')

    # a_6_4 = ((-a).t())[:, :, "n1"]
    # assert(a_6_4.domain == [n== i])
    # assert(a_6_4.gamsRepr() == '-a(n,"n1",i)')

    a_6_5 = (-a.t())[:, :, "n1"]
    assert a_6_5.domain == [n, i]
    assert a_6_5.gamsRepr() == '( - a(n,"n1",i))'

    a_6_6 = -((a.t())[:, :, "n1"])
    assert a_6_6.domain == [n, i]
    assert a_6_6.gamsRepr() == '( - a(n,"n1",i))'


def test_shift_permute(data):
    m = data
    i = Set(m, name="i", records=["i1", "i2", "i3"])
    j = Set(m, name="j", records=["j1", "j2", "j3"])
    k = Set(m, name="k", records=["k1", "k2", "k3"])
    a = Parameter(m, name="a", domain=[i, j, k])
    # same permute
    a0 = permute(a, [0, 1, 2])
    assert a0.domain == [i, j, k]
    assert a0["i1", "j1", "k1"].gamsRepr() == 'a("i1","j1","k1")'

    # left shift
    a1 = permute(a, [1, 2, 0])
    assert a1.domain == [j, k, i]
    assert a1["j1", "k1", "i1"].gamsRepr() == 'a("i1","j1","k1")'

    a2 = permute(a1, [1, 2, 0])
    assert a2.domain == [k, i, j]
    assert a2["k1", "i1", "j1"].gamsRepr() == 'a("i1","j1","k1")'

    a3 = permute(a2, [1, 2, 0])
    assert a3.domain == [i, j, k]
    assert a3["i1", "j1", "k1"].gamsRepr() == 'a("i1","j1","k1")'

    # right shift
    a1 = permute(a, [2, 0, 1])
    assert a1.domain == [k, i, j]
    assert a1["k1", "i1", "j1"].gamsRepr() == 'a("i1","j1","k1")'

    a2 = permute(a1, [2, 0, 1])
    assert a2.domain == [j, k, i]
    assert a2["j1", "k1", "i1"].gamsRepr() == 'a("i1","j1","k1")'

    a3 = permute(a2, [2, 0, 1])
    assert a3.domain == [i, j, k]
    assert a3["i1", "j1", "k1"].gamsRepr() == 'a("i1","j1","k1")'


def test_permute_bruteforce_var(data):
    m = data
    i = Set(m, name="i", records=["i1", "i2", "i3"])
    j = Set(m, name="j", records=["j1", "j2", "j3"])
    k = Set(m, name="k", records=["k1", "k2", "k3"])
    l = Set(m, name="l", records=["l1", "l2", "l3"])
    a = Variable(m, name="a", domain=[i, j, k, l])
    ax = a
    for perm in itertools.permutations(range(4)):
        ax = permute(ax, perm)

    # computed via pytorch
    assert ax.domain == [k, l, i, j]


def test_permute_bruteforce_par(data):
    m = data
    i = Set(m, name="i", records=["i1", "i2", "i3"])
    j = Set(m, name="j", records=["j1", "j2", "j3"])
    k = Set(m, name="k", records=["k1", "k2", "k3"])
    l = Set(m, name="l", records=["l1", "l2", "l3"])
    a = Parameter(m, name="a", domain=[i, j, k, l])
    ax = a
    # we trust that itertools will give permutations always in same order
    for perm in itertools.permutations(range(4)):
        ax = permute(ax, perm)

    # computed via pytorch
    assert ax.domain == [k, l, i, j]


def test_permute_bad(data):
    m = data
    i = Set(m, name="i", records=["i1", "i2", "i3"])
    j = Set(m, name="j", records=["j1", "j2", "j3"])
    k = Set(m, name="k", records=["k1", "k2", "k3"])
    l = Set(m, name="l", records=["l1", "l2", "l3"])
    a = Parameter(m, name="a", domain=[i, j, k, l])
    pytest.raises(ValidationError, lambda: permute(a, [2, 2, 2, 2]))
    pytest.raises(ValidationError, lambda: permute(a, [5, 2, 2, 2]))
    pytest.raises(ValidationError, lambda: permute(a, [-1, 2, 2, 2]))
    pytest.raises(ValidationError, lambda: permute(a, [0, 2, 2, 3]))
    pytest.raises(ValidationError, lambda: permute(a, ["1", 2, 3, 4]))


def test_transpose(data):
    m = data
    i = Set(m, name="i", records=["i1", "i2", "i3"])
    j = Set(m, name="j", records=["j1", "j2", "j3"])
    k = Set(m, name="k", records=["k1", "k2", "k3"])
    l = Set(m, name="l", records=["l1", "l2", "l3"])
    a = Parameter(m, name="a", domain=[i, j, k, l])
    a_t = a.t()
    assert a_t.domain == [i, j, l, k]
    assert a_t["i1", "j1", "l1", "k1"].gamsRepr() == 'a("i1","j1","k1","l1")'

    a_t = a.T
    assert a_t.domain == [i, j, l, k]
    assert a_t["i1", "j1", "l1", "k1"].gamsRepr() == 'a("i1","j1","k1","l1")'

    b = Parameter(m, name="b", domain=[i])
    a2 = Variable(m, name="a2", domain=[i])
    pytest.raises(ValidationError, lambda: b.T)  # par
    pytest.raises(ValidationError, lambda: a2.T)  # var
    pytest.raises(ValidationError, lambda: b[i].T)  # imp par
    pytest.raises(ValidationError, lambda: a2[i].T)  # imp var


def test_domain_conflict_resolution_3(data):
    m = data
    mat1 = Parameter(m, name="mat1", domain=dim([30, 20, 30, 20]))
    mat2 = Parameter(m, name="mat2", domain=dim([20, 30]))

    expr = mat1 @ mat2
    assert len(expr.domain) == 4
    assert len(set([x.name for x in expr.domain])) == 4

    expr2 = mat2 @ mat1
    assert len(expr2.domain) == 4
    assert len(set([x.name for x in expr2.domain])) == 4
