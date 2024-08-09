from __future__ import annotations

import itertools
import math
import unittest

import gamspy as gp
import numpy as np
from gamspy import Alias, Container, Parameter, Set, Sum, Variable
from gamspy.exceptions import ValidationError
from gamspy.math import dim, permute, trace, vector_norm


class MatrixSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container()

    def test_matrix_mult_bad(self):
        i = Set(self.m, name="i", records=["i1", "i2", "i3"])
        j = Set(self.m, name="j", records=["j1", "j2", "j3"])

        a_recs = np.random.randint(1, 11, size=(3))
        b_recs = np.random.randint(1, 11, size=(3))
        a = Parameter(
            self.m, name="a", domain=[i], records=a_recs, uels_on_axes=True
        )
        b = Parameter(
            self.m, name="b", domain=[j], records=b_recs, uels_on_axes=True
        )
        c = Parameter(self.m, name="c", domain=[], uels_on_axes=True)

        self.assertRaises(ValidationError, lambda: c @ a)
        self.assertRaises(ValidationError, lambda: a @ c)
        self.assertRaises(ValidationError, lambda: a @ b)

    def test_simple_matrix_matrix(self):
        """Test simple case where domain calculation is trivial
        matrix x matrix"""
        i = Set(self.m, name="i", records=["i1", "i2", "i3"])
        j = Set(self.m, name="j", records=["j1", "j2", "j3"])
        k = Set(self.m, name="k", records=["k1", "k2", "k3"])

        a_recs = np.random.randint(1, 11, size=(3, 3))
        b_recs = np.random.randint(1, 11, size=(3, 3))
        a = Parameter(
            self.m, name="a", domain=[i, j], records=a_recs, uels_on_axes=True
        )
        b = Parameter(
            self.m, name="b", domain=[j, k], records=b_recs, uels_on_axes=True
        )
        c = a @ b
        self.assertEqual(c.domain, [i, k])

        c = Parameter(self.m, name="c", domain=[i, k])
        c[i, k] = a @ b
        c_recs = c.toDense()
        self.assertTrue(np.allclose(c_recs, a_recs @ b_recs))

        a2 = Parameter(
            self.m, name="a2", domain=[k, k], records=a_recs, uels_on_axes=True
        )
        # dims do not match
        self.assertRaises(ValidationError, lambda: a2 @ b)

    def test_simple_matrix_vector(self):
        """Test simple case where domain calculation is trivial
        matrix x vector"""
        i = Set(self.m, name="i", records=["i1", "i2", "i3"])
        j = Set(self.m, name="j", records=["j1", "j2", "j3"])

        a_recs = np.random.randint(1, 11, size=(3, 3))
        b_recs = np.random.randint(1, 11, size=(3))
        a = Parameter(
            self.m, name="a", domain=[i, j], records=a_recs, uels_on_axes=True
        )
        b = Parameter(
            self.m, name="b", domain=[j], records=b_recs, uels_on_axes=True
        )
        c = a @ b
        self.assertEqual(c.domain, [i])
        c = Parameter(self.m, name="c", domain=[i])
        c[i] = a @ b
        c_recs = c.toDense()
        self.assertTrue(np.allclose(c_recs, a_recs @ b_recs))

        a2 = Parameter(
            self.m, name="a2", domain=[j, i], records=a_recs, uels_on_axes=True
        )
        # dims do not match
        self.assertRaises(ValidationError, lambda: a2 @ b)

    def test_simple_vector_vector(self):
        """Test simple case where domain calculation is trivial
        vector x vector, aka inner product"""
        i = Set(self.m, name="i", records=["i1", "i2", "i3"])

        a_recs = np.random.randint(1, 11, size=(3))
        b_recs = np.random.randint(1, 11, size=(3))
        a = Parameter(
            self.m, name="a", domain=[i], records=a_recs, uels_on_axes=True
        )
        b = Parameter(
            self.m, name="b", domain=[i], records=b_recs, uels_on_axes=True
        )
        c = a @ b
        self.assertEqual(c.domain, [])
        c = Parameter(self.m, name="c", domain=[])
        c[...] = a @ b
        c_recs = c.toDense()
        self.assertTrue(np.allclose(c_recs, a_recs @ b_recs))

    def test_vector_vector_with_conflicting_sum_domain(self):
        i = Set(self.m, name="i", records=["i1", "i2", "i3"])
        j = Alias(self.m, name="j", alias_with=i)

        x_recs = np.random.randint(1, 11, size=(3, 3))
        y_recs = np.random.randint(1, 11, size=(3))
        x = Variable(
            self.m, name="x", domain=[j, i], records=x_recs, uels_on_axes=True
        )
        y = Parameter(
            self.m, name="y", domain=[i], records=y_recs, uels_on_axes=True
        )

        right_side = x @ y  # has controlled_domain domain of i
        self.assertEqual(right_side.controlled_domain, [i])

        z = Parameter(self.m, name="z", domain=[i])
        val = z @ right_side
        self.assertEqual(val.domain, [])
        self.assertEqual(len(val.controlled_domain), 2)
        self.assertEqual(val.op_domain[0].name, "AliasOfi_2")

    def test_simple_vector_matrix(self):
        """Test simple case where domain calculation is trivial
        vector x matrix"""
        i = Set(self.m, name="i", records=["i1", "i2", "i3"])
        j = Set(self.m, name="j", records=["j1", "j2", "j3"])

        a_recs = np.random.randint(1, 11, size=(3))
        b_recs = np.random.randint(1, 11, size=(3, 3))
        a = Parameter(
            self.m, name="a", domain=[i], records=a_recs, uels_on_axes=True
        )
        b = Parameter(
            self.m, name="b", domain=[i, j], records=b_recs, uels_on_axes=True
        )
        c = a @ b
        self.assertEqual(c.domain, [j])
        c = Parameter(self.m, name="c", domain=[j])
        c[...] = a @ b
        c_recs = c.toDense()
        self.assertTrue(np.allclose(c_recs, a_recs @ b_recs))

        a2 = Parameter(
            self.m, name="a2", domain=[j], records=a_recs, uels_on_axes=True
        )

        self.assertRaises(ValidationError, lambda: a2 @ b)

    def test_batched_matrix_matrix(self):
        """Test batched matrix multiplication,
        batched matrix x batched matrix"""
        n = Set(self.m, name="n", records=["n1", "n2", "n3", "n4"])
        i = Set(self.m, name="i", records=["i1", "i2", "i3"])
        j = Set(self.m, name="j", records=["j1", "j2", "j3"])
        k = Set(self.m, name="k", records=["k1", "k2", "k3"])

        a_recs = np.random.randint(1, 11, size=(4, 3, 3))
        b_recs = np.random.randint(1, 11, size=(4, 3, 3))
        a = Parameter(
            self.m,
            name="a",
            domain=[n, i, j],
            records=a_recs,
            uels_on_axes=True,
        )
        b = Parameter(
            self.m,
            name="b",
            domain=[n, j, k],
            records=b_recs,
            uels_on_axes=True,
        )
        c = a @ b
        self.assertEqual(c.domain, [n, i, k])
        c = Parameter(self.m, name="c", domain=[n, i, k])
        c[...] = a @ b
        c_recs = c.toDense()
        self.assertTrue(np.allclose(c_recs, a_recs @ b_recs))

        a2 = Parameter(
            self.m,
            name="a2",
            domain=[n, j, i],
            records=a_recs,
            uels_on_axes=True,
        )
        self.assertRaises(ValidationError, lambda: a2 @ b)

    def test_batched_matrix_vector(self):
        """Test batched matrix - vector multiplication,
        batched matrix x vector"""
        n = Set(self.m, name="n", records=["n1", "n2", "n3", "n4"])
        i = Set(self.m, name="i", records=["i1", "i2", "i3"])
        j = Set(self.m, name="j", records=["j1", "j2", "j3"])

        a_recs = np.random.randint(1, 11, size=(4, 3, 3))
        b_recs = np.random.randint(1, 11, size=(3))
        a = Parameter(
            self.m,
            name="a",
            domain=[n, i, j],
            records=a_recs,
            uels_on_axes=True,
        )
        b = Parameter(
            self.m,
            name="b",
            domain=[j],
            records=b_recs,
            uels_on_axes=True,
        )
        c = a @ b
        self.assertEqual(c.domain, [n, i])
        c = Parameter(self.m, name="c", domain=[n, i])
        c[...] = a @ b
        c_recs = c.toDense()
        self.assertTrue(np.allclose(c_recs, a_recs @ b_recs))

        a2 = Parameter(
            self.m,
            name="a2",
            domain=[n, j, i],
            records=a_recs,
            uels_on_axes=True,
        )
        self.assertRaises(ValidationError, lambda: a2 @ b)

    def test_batched_matrix_matrix_2(self):
        """Test batched matrix multiplication,
        batched matrix x matrix"""
        n = Set(self.m, name="n", records=["n1", "n2", "n3", "n4"])
        i = Set(self.m, name="i", records=["i1", "i2", "i3"])
        j = Set(self.m, name="j", records=["j1", "j2", "j3"])
        k = Set(self.m, name="k", records=["k1", "k2", "k3"])

        a_recs = np.random.randint(1, 11, size=(4, 3, 3))
        b_recs = np.random.randint(1, 11, size=(3, 3))
        a = Parameter(
            self.m,
            name="a",
            domain=[n, i, j],
            records=a_recs,
            uels_on_axes=True,
        )
        b = Parameter(
            self.m, name="b", domain=[j, k], records=b_recs, uels_on_axes=True
        )
        c = a @ b
        self.assertEqual(c.domain, [n, i, k])
        c = Parameter(self.m, name="c", domain=[n, i, k])
        c[...] = a @ b
        c_recs = c.toDense()
        self.assertTrue(np.allclose(c_recs, a_recs @ b_recs))

    def test_vector_batched_matrix(self):
        """Test vector x batched_matrix"""
        n = Set(self.m, name="n", records=["n1", "n2", "n3", "n4"])
        i = Set(self.m, name="i", records=["i1", "i2", "i3"])
        j = Set(self.m, name="j", records=["j1", "j2", "j3"])

        a_recs = np.random.randint(1, 11, size=(4, 3, 3))
        b_recs = np.random.randint(1, 11, size=(3))
        a = Parameter(
            self.m,
            name="a",
            domain=[n, i, j],
            records=a_recs,
            uels_on_axes=True,
        )
        b = Parameter(
            self.m, name="b", domain=[i], records=b_recs, uels_on_axes=True
        )
        c = b @ a

        self.assertEqual(c.domain, [n, j])
        c = Parameter(self.m, name="c", domain=[n, j])
        c[...] = b @ a
        c_recs = c.toDense()
        self.assertTrue(np.allclose(c_recs, b_recs @ a_recs))

        b2 = Parameter(
            self.m, name="b2", domain=[j], records=b_recs, uels_on_axes=True
        )
        self.assertRaises(ValidationError, lambda: b2 @ a)

    def test_square_matrix_mult(self):
        i = Set(self.m, name="i", records=["i1", "i2", "i3"])
        j = Alias(self.m, name="j", alias_with=i)
        k = Alias(self.m, name="k", alias_with=j)

        a_recs = np.random.randint(1, 11, size=(3, 3))
        b_recs = np.random.randint(1, 11, size=(3, 3))
        a = Parameter(
            self.m, name="a", domain=[i, i], records=a_recs, uels_on_axes=True
        )
        b = Parameter(
            self.m, name="b", domain=[i, i], records=b_recs, uels_on_axes=True
        )

        c2 = a[i, j] @ b[j, k]
        self.assertEqual(c2.domain, [i, k])
        c = Parameter(self.m, name="c", domain=[i, k])
        # reindexing is required in this case
        c[...] = (a @ b)[i, k]
        c_recs = c.toDense()
        self.assertTrue(np.allclose(c_recs, a_recs @ b_recs))

    def test_square_matrix_mult_2(self):
        n = Set(self.m, name="n", records=["n1", "n2", "n3", "n4"])
        i = Set(self.m, name="i", records=["i1", "i2", "i3"])
        j = Alias(self.m, name="j", alias_with=i)
        k = Alias(self.m, name="k", alias_with=j)

        a_recs = np.random.randint(1, 11, size=(4, 3, 3))
        b_recs = np.random.randint(1, 11, size=(4, 3, 3))
        a = Parameter(
            self.m,
            name="a",
            domain=[n, i, i],
            records=a_recs,
            uels_on_axes=True,
        )
        b = Parameter(
            self.m,
            name="b",
            domain=[n, i, i],
            records=b_recs,
            uels_on_axes=True,
        )

        c2 = a[n, i, j] @ b[n, j, k]
        self.assertEqual(c2.domain, [n, i, k])
        c = Parameter(self.m, name="c", domain=[n, i, k])
        c[...] = (a[n, i, j] @ b[n, j, k])[n, i, k]
        c_recs = c.toDense()
        self.assertTrue(np.allclose(c_recs, a_recs @ b_recs))

    def test_square_matrix_mult_3(self):
        i = Set(self.m, name="i", records=["i1", "i2", "i3"])
        j = Alias(self.m, name="j", alias_with=i)

        a_recs = np.random.randint(1, 11, size=(3, 3))
        b_recs = np.random.randint(1, 11, size=(3, 3))
        a = Parameter(
            self.m,
            name="a",
            domain=[i, i],
            records=a_recs,
            uels_on_axes=True,
        )
        b = Parameter(
            self.m,
            name="b",
            domain=[i, i],
            records=b_recs,
            uels_on_axes=True,
        )

        c2 = (a @ b) @ (a @ b)
        c = Parameter(self.m, name="c", domain=[i, i])
        c[i, j] = c2[i, j]
        c_recs = c.toDense()
        self.assertTrue(
            np.allclose(c_recs, (a_recs @ b_recs) @ (a_recs @ b_recs))
        )

    def test_square_matrix_mult_4(self):
        a_recs = np.random.randint(1, 11, size=(3, 3))
        b_recs = np.random.randint(1, 11, size=(3, 3))
        a = Parameter(
            self.m,
            name="a",
            domain=dim([3, 3]),
            records=a_recs,
            uels_on_axes=True,
        )
        b = Parameter(
            self.m,
            name="b",
            domain=dim([3, 3]),
            records=b_recs,
            uels_on_axes=True,
        )

        c2 = (a @ b) + a
        c = Parameter(self.m, name="c", domain=dim([3, 3]))
        c[...] = c2
        c_recs = c.toDense()
        self.assertTrue(np.allclose(c_recs, (a_recs @ b_recs) + (a_recs)))

    def test_batch_size_matches(self):
        n = Set(self.m, name="n", records=["n1", "n2", "n3"])
        m = Set(self.m, name="m", records=["m1", "m2", "m3"])
        i = Set(self.m, name="i", records=["i1", "i2", "i3"])
        j = Alias(self.m, name="j", alias_with=i)
        k = Alias(self.m, name="k", alias_with=j)

        a = Parameter(self.m, name="a", domain=[n, i, j])
        b = Parameter(self.m, name="b", domain=[m, j, k])
        c = Parameter(self.m, name="c", domain=[n, m, j, k])

        self.assertRaises(ValidationError, lambda: a @ b)
        self.assertRaises(ValidationError, lambda: a @ c)

    def test_domain_conflict_resolution(self):
        n = Set(self.m, name="n", records=["n1", "n2", "n3"])
        i = Set(self.m, name="i", records=["i1", "i2", "i3"])

        vec = Parameter(self.m, name="vec", domain=[i])
        mat = Parameter(self.m, name="mat", domain=[i, i])
        batched_mat = Parameter(self.m, name="batched_mat", domain=[n, i, i])

        r1 = vec @ vec
        self.assertEqual(r1.domain, [])

        r2 = mat @ mat
        self.assertEqual(len(r2.domain), 2)
        self.assertEqual(len(r2.controlled_domain), 1)
        self.assertNotEqual(r2.domain[0], r2.domain[1])

        r3 = vec @ mat
        self.assertEqual(len(r3.domain), 1)
        self.assertEqual(len(r3.controlled_domain), 1)

        r4 = mat @ vec
        self.assertEqual(len(r4.domain), 1)
        self.assertEqual(len(r4.controlled_domain), 1)

        r5 = vec @ batched_mat
        self.assertEqual(len(r5.domain), 2)
        self.assertEqual(len(r5.controlled_domain), 1)
        self.assertEqual(r5.domain[0], batched_mat.domain[0])

        r6 = batched_mat @ vec
        self.assertEqual(len(r6.domain), 2)
        self.assertEqual(len(r6.controlled_domain), 1)
        self.assertEqual(r6.domain[0], batched_mat.domain[0])

        r7 = batched_mat @ batched_mat
        self.assertEqual(len(r7.domain), 3)
        self.assertEqual(len(r7.controlled_domain), 1)
        self.assertNotEqual(r7.domain[-1], r7.domain[-2])

    def test_domain_conflict_resolution_2(self):
        vec = Parameter(self.m, name="vec", domain=dim([3]))
        mat = Parameter(self.m, name="mat", domain=dim([3, 3]))
        batched_mat = Parameter(
            self.m, name="batched_mat", domain=dim([3, 3, 3])
        )

        r2 = mat @ mat
        self.assertEqual(len(r2.domain), 2)
        self.assertEqual(r2.domain[0], mat.domain[0])
        self.assertEqual(r2.domain[1], mat.domain[1])

        # Added an exception to make this one work
        r3 = vec @ mat
        self.assertEqual(len(r3.domain), 1)
        self.assertEqual(r3.domain[0], vec.domain[0])

        r4 = mat @ vec
        self.assertEqual(len(r4.domain), 1)
        self.assertEqual(r4.domain[0], vec.domain[0])

        # Added an exception to make this one work
        r5 = vec @ batched_mat
        self.assertEqual(len(r5.domain), 2)
        self.assertEqual(len(r5.controlled_domain), 1)
        self.assertEqual(r5.domain[0], batched_mat.domain[0])

        # This one does not work
        # r6 = batched_mat @ vec
        # self.assertEqual(len(r6.domain), 2)
        # self.assertEqual(len(r6.controlled_domain), 1)
        # self.assertEqual(r6.domain[0], batched_mat.domain[0])

        r7 = batched_mat @ batched_mat
        self.assertEqual(len(r7.domain), 3)
        self.assertEqual(r7.domain[0], batched_mat.domain[0])
        self.assertEqual(r7.domain[1], batched_mat.domain[1])
        self.assertEqual(r7.domain[2], batched_mat.domain[2])

    def test_trace_on_matrix(self):
        identity = np.eye(3, 3)
        mat = Parameter(
            self.m,
            name="mat",
            domain=dim([3, 3]),
            records=identity,
            uels_on_axes=True,
        )

        trace_expr = trace(mat)
        self.assertEqual(trace_expr.domain, [])
        sc = Parameter(self.m, name="sc", domain=[])
        sc[...] = trace(mat)
        self.assertEqual(np.trace(identity), sc.toDense())

        rand_recs = np.random.randint(1, 11, size=(3, 3))
        mat.setRecords(rand_recs, uels_on_axes=True)
        sc[...] = trace(mat)
        self.assertEqual(np.trace(rand_recs), sc.toDense())

        recs = np.ones((3, 5))
        rect = Parameter(
            self.m,
            name="rect",
            domain=dim([3, 5]),
            records=recs,
            uels_on_axes=True,
        )

        self.assertRaises(ValidationError, lambda: trace(rect))

    def test_trace_on_vector(self):
        vec = Parameter(self.m, name="vec", domain=dim([3]))
        self.assertRaises(ValidationError, lambda: trace(vec))

    def test_trace_on_batched_matrix(self):
        recs = np.random.randint(1, 11, size=(128, 3, 3))
        bm1 = Parameter(
            self.m,
            name="vec",
            domain=dim([128, 3, 3]),
            records=recs,
            uels_on_axes=True,
        )

        sc1 = Parameter(self.m, name="sc1", domain=dim([128]))
        expr1 = trace(bm1, axis1=1, axis2=2)
        self.assertEqual(expr1.domain[0].name, "DenseDim128_1")

        sc1[...] = expr1
        sc1_dens = sc1.toDense()
        self.assertTrue(
            np.allclose(sc1_dens, np.trace(recs, axis1=1, axis2=2))
        )

    def test_domain_relabeling(self):
        n = Set(self.m, name="n", records=["n1", "n2", "n3", "n4"])
        i = Set(self.m, name="i", records=["i1", "i2", "i3"])
        i2 = Alias(self.m, name="i2", alias_with=i)
        j = Set(self.m, name="j", records=["j1", "j2", "j3"])
        k = Set(self.m, name="k", records=["k1", "k2", "k3"])
        k2 = Alias(self.m, name="k2", alias_with=k)

        a = Variable(self.m, name="a", domain=[n, i, j])
        b = Parameter(self.m, name="b", domain=[j, k])

        expr = a + b
        expr2 = expr[n, i2, j, k]
        self.assertEqual(expr.domain, [n, i, j, k])
        self.assertEqual(expr.gamsRepr(), "(a(n,i,j) + b(j,k))")
        self.assertEqual(expr2.domain, [n, i2, j, k])
        self.assertEqual(expr2.gamsRepr(), "(a(n,i2,j) + b(j,k))")
        self.assertRaises(ValidationError, lambda: expr[n])

        expr3 = (a + a) + (b + b)
        self.assertEqual(
            expr3.gamsRepr(), "((a(n,i,j) + a(n,i,j)) + (b(j,k) + b(j,k)))"
        )
        expr4 = expr3[n, i2, j, k]
        self.assertEqual(
            expr4.gamsRepr(), "((a(n,i2,j) + a(n,i2,j)) + (b(j,k) + b(j,k)))"
        )

        expr5 = Sum(n, expr3)
        self.assertEqual(
            expr5.gamsRepr(),
            "sum(n,((a(n,i,j) + a(n,i,j)) + (b(j,k) + b(j,k))))",
        )

        expr6 = expr5[i2, j, k]
        self.assertEqual(
            expr6.gamsRepr(),
            "sum(n,((a(n,i2,j) + a(n,i2,j)) + (b(j,k) + b(j,k))))",
        )

        expr7 = Sum(j, expr6)
        self.assertEqual(
            expr7.gamsRepr(),
            "sum(j,sum(n,((a(n,i2,j) + a(n,i2,j)) + (b(j,k) + b(j,k)))))",
        )

        expr8 = expr7[i, k2]
        self.assertEqual(
            expr8.gamsRepr(),
            "sum(j,sum(n,((a(n,i,j) + a(n,i,j)) + (b(j,k2) + b(j,k2)))))",
        )

    def test_vector_norm_not_implemented(self):
        i = Set(self.m, name="i", records=["i1", "i2"])
        a = Variable(self.m, name="a", domain=[i])
        self.assertRaises(ValidationError, lambda: vector_norm(a, ord=0))
        self.assertRaises(
            ValidationError, lambda: vector_norm(a, ord=float("inf"))
        )
        self.assertRaises(
            ValidationError, lambda: vector_norm(a, ord=float("-inf"))
        )

    def test_vector_norm(self):
        i = Set(self.m, name="i", records=["i1", "i2"])
        b = Parameter(
            self.m, name="b", domain=[i], records=[("i1", 3), ("i2", 4)]
        )
        c = Parameter(self.m, name="c")

        n_expr = vector_norm(b, ord=2)
        c[...] = n_expr
        c_val = c.records.iloc[0, 0]
        self.assertTrue(math.isclose(c_val, 5, rel_tol=1e-4))

        n_expr = vector_norm(b, ord=2.0)
        c[...] = n_expr
        c_val = c.records.iloc[0, 0]
        self.assertTrue(math.isclose(c_val, 5, rel_tol=1e-4))

        # this is a special case
        norm_squared = n_expr**2
        self.assertTrue(
            isinstance(norm_squared, gp._algebra.operation.Operation)
        )

        c[...] = vector_norm(b, ord=3)
        c_val = c.records.iloc[0, 0]
        self.assertTrue(math.isclose(c_val, 4.49794, rel_tol=1e-4))

        c[...] = vector_norm(b, ord=4)
        c_val = c.records.iloc[0, 0]
        self.assertTrue(math.isclose(c_val, 4.28457, rel_tol=1e-4))

        c[...] = vector_norm(b, ord=1)
        c_val = c.records.iloc[0, 0]
        self.assertTrue(math.isclose(c_val, 7.0, rel_tol=1e-4))

    def test_vector_norm_2(self):
        i = Set(self.m, name="i", records=["i1", "i2"])
        n = Set(self.m, name="n", records=["n1", "n2", "n3"])
        b = Parameter(
            self.m,
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
        c = Parameter(self.m, name="c", domain=[n])

        c[n] = vector_norm(b, dim=[1])
        self.assertTrue(math.isclose(c.records.iloc[0, 1], 5, rel_tol=1e-5))
        self.assertTrue(math.isclose(c.records.iloc[1, 1], 25, rel_tol=1e-5))
        self.assertTrue(math.isclose(c.records.iloc[2, 1], 13, rel_tol=1e-5))

        c[n] = vector_norm(b, dim=[i])
        self.assertTrue(math.isclose(c.records.iloc[0, 1], 5, rel_tol=1e-5))
        self.assertTrue(math.isclose(c.records.iloc[1, 1], 25, rel_tol=1e-5))
        self.assertTrue(math.isclose(c.records.iloc[2, 1], 13, rel_tol=1e-5))

    def test_vector_norm_3(self):
        i = Set(self.m, name="i", records=["i1", "i2"])
        n = Set(self.m, name="n", records=["n1", "n2"])
        a = Variable(self.m, name="a", domain=[n, i])

        self.assertEqual(vector_norm(a[:, "i1"]).domain, [])
        self.assertEqual(vector_norm(a["n1", :]).domain, [])
        self.assertRaises(ValidationError, lambda: vector_norm(a["n1", "i1"]))

    def test_vector_norm_dim(self):
        i = Set(self.m, name="i", records=["i1", "i2"])
        a = Variable(self.m, name="a", domain=[i])
        self.assertRaises(ValidationError, lambda: vector_norm(a, dim="asd"))
        self.assertRaises(ValidationError, lambda: vector_norm(a, dim=[]))
        self.assertRaises(ValidationError, lambda: vector_norm(a, dim=[0, i]))
        self.assertRaises(ValidationError, lambda: vector_norm(a, dim=["asd"]))
        self.assertRaises(ValidationError, lambda: vector_norm(a, dim=2))

    def test_literal_indexing(self):
        i = Set(self.m, name="i", records=["i1", "i2"])
        n = Set(self.m, name="n", records=["n1", "n2", "n3"])
        a = Variable(self.m, name="a", domain=[n, i])
        b = Parameter(self.m, name="b", domain=[n, i])

        # try simple case
        a_1 = a[:, "i1"]
        self.assertEqual(a_1.domain, [n])
        self.assertEqual(a_1.gamsRepr(), 'a(n,"i1")')

        # try simple case
        a_2 = a["n1", :]
        self.assertEqual(a_2.domain, [i])
        self.assertEqual(a_2.gamsRepr(), 'a("n1",i)')

        a_3 = a["n1", "i1"]
        self.assertEqual(a_3.domain, [])
        self.assertEqual(a_3.gamsRepr(), 'a("n1","i1")')

        a_4 = a[:, "i1"]["n1"]
        self.assertEqual(a_4.domain, [])
        self.assertEqual(a_4.gamsRepr(), 'a("n1","i1")')

        a_5 = a["n1", :]["i1"]
        self.assertEqual(a_5.domain, [])
        self.assertEqual(a_5.gamsRepr(), 'a("n1","i1")')

        a_6 = -a["n1", :]
        self.assertEqual(a_6.domain, [i])
        self.assertEqual(a_6.gamsRepr(), '( - a("n1",i))')

        # try simple case
        b_1 = b[:, "i1"]
        self.assertEqual(b_1.domain, [n])
        self.assertEqual(b_1.gamsRepr(), 'b(n,"i1")')

        # try simple case
        b_2 = b["n1", :]
        self.assertEqual(b_2.domain, [i])
        self.assertEqual(b_2.gamsRepr(), 'b("n1",i)')

        b_3 = b["n1", "i1"]
        self.assertEqual(b_3.domain, [])
        self.assertEqual(b_3.gamsRepr(), 'b("n1","i1")')

        b_4 = b[:, "i1"]["n1"]
        self.assertEqual(b_4.domain, [])
        self.assertEqual(b_4.gamsRepr(), 'b("n1","i1")')

        b_5 = b["n1", :]["i1"]
        self.assertEqual(b_5.domain, [])
        self.assertEqual(b_5.gamsRepr(), 'b("n1","i1")')

        b_6 = -b["n1", :]
        self.assertEqual(b_6.domain, [i])
        self.assertEqual(b_6.gamsRepr(), '( - b("n1",i))')

    def test_literal_indexing_mix_permute_variable(self):
        i = Set(self.m, name="i", records=["i1", "i2"])
        n = Set(self.m, name="n", records=["n1", "n2", "n3"])
        n2 = Set(self.m, name="n2", records=["n1", "n2", "n3"])
        a = Variable(self.m, name="a", domain=[n, n2, i])

        a_1 = a[:, :, "i1"].t()
        self.assertEqual(a_1.domain, [n2, n])
        self.assertEqual(a_1.gamsRepr(), 'a(n,n2,"i1")')

        a_1_2 = a.t()[:, "i1", :]
        self.assertEqual(a_1_2.domain, [n, n2])
        self.assertEqual(a_1_2.gamsRepr(), 'a(n,n2,"i1")')

        a_2 = a[:, "n1", :].t()
        self.assertEqual(a_2.domain, [i, n])
        self.assertEqual(a_2.gamsRepr(), 'a(n,"n1",i)')

        a_2_2 = a.t()[:, :, "n1"]
        self.assertEqual(a_2_2.domain, [n, i])
        self.assertEqual(a_2_2.gamsRepr(), 'a(n,"n1",i)')

        a_3 = a[:, "n1", "i1"]
        self.assertEqual(a_3.domain, [n])
        self.assertEqual(a_3.gamsRepr(), 'a(n,"n1","i1")')

        a_3_2 = a.t()[:, "i1", "n1"]
        self.assertEqual(a_3_2.domain, [n])
        self.assertEqual(a_3_2.gamsRepr(), 'a(n,"n1","i1")')

        a_4 = a[:, :, "i1"][:, "n1"]
        self.assertEqual(a_4.domain, [n])
        self.assertEqual(a_4.gamsRepr(), 'a(n,"n1","i1")')

        a_4_2 = a.t()[:, "i1", :][:, "n1"]
        self.assertEqual(a_4_2.domain, [n])
        self.assertEqual(a_4_2.gamsRepr(), 'a(n,"n1","i1")')

        a_4_3 = a[:, :, "i1"].t()[:, "n1"]
        self.assertEqual(a_4_3.domain, [n2])
        self.assertEqual(a_4_3.gamsRepr(), 'a("n1",n2,"i1")')

        a_5 = a[:, "n1", :][:, "i1"]
        self.assertEqual(a_5.domain, [n])
        self.assertEqual(a_5.gamsRepr(), 'a(n,"n1","i1")')

        a_5_2 = a.t()[:, :, "n1"][:, "i1"]
        self.assertEqual(a_5_2.domain, [n])
        self.assertEqual(a_5_2.gamsRepr(), 'a(n,"n1","i1")')

        a_5_3 = a[:, "n1", :].t()["i1", :]
        self.assertEqual(a_5_3.domain, [n])
        self.assertEqual(a_5_3.gamsRepr(), 'a(n,"n1","i1")')

        # a_6 = (-(a[:, "n1", :])).t()
        # self.assertEqual(a_6.domain, [i, n])
        # self.assertEqual(a_6.gamsRepr(), '-a(n,"n1",i)')

        a_6_2 = -((a[:, "n1", :]).t())
        self.assertEqual(a_6_2.domain, [i, n])
        self.assertEqual(a_6_2.gamsRepr(), '( - a(n,"n1",i))')

        # a_6_3 = ((-a)[:, "n1", :]).t()
        # self.assertEqual(a_6_3.domain, [i, n])
        # self.assertEqual(a_6_3.gamsRepr(), '-a(n,"n1",i)')

        # a_6_4 = ((-a).t())[:, :, "n1"]
        # self.assertEqual(a_6_4.domain, [n, i])
        # self.assertEqual(a_6_4.gamsRepr(), '-a(n,"n1",i)')

        a_6_5 = (-a.t())[:, :, "n1"]
        self.assertEqual(a_6_5.domain, [n, i])
        self.assertEqual(a_6_5.gamsRepr(), '( - a(n,"n1",i))')

        a_6_6 = -((a.t())[:, :, "n1"])
        self.assertEqual(a_6_6.domain, [n, i])
        self.assertEqual(a_6_6.gamsRepr(), '( - a(n,"n1",i))')

    def test_literal_indexing_mix_permute_parameter(self):
        i = Set(self.m, name="i", records=["i1", "i2"])
        n = Set(self.m, name="n", records=["n1", "n2", "n3"])
        n2 = Set(self.m, name="n2", records=["n1", "n2", "n3"])
        a = Parameter(self.m, name="a", domain=[n, n2, i])

        a_1 = a[:, :, "i1"].t()
        self.assertEqual(a_1.domain, [n2, n])
        self.assertEqual(a_1.gamsRepr(), 'a(n,n2,"i1")')

        a_1_2 = a.t()[:, "i1", :]
        self.assertEqual(a_1_2.domain, [n, n2])
        self.assertEqual(a_1_2.gamsRepr(), 'a(n,n2,"i1")')

        a_2 = a[:, "n1", :].t()
        self.assertEqual(a_2.domain, [i, n])
        self.assertEqual(a_2.gamsRepr(), 'a(n,"n1",i)')

        a_2_2 = a.t()[:, :, "n1"]
        self.assertEqual(a_2_2.domain, [n, i])
        self.assertEqual(a_2_2.gamsRepr(), 'a(n,"n1",i)')

        a_3 = a[:, "n1", "i1"]
        self.assertEqual(a_3.domain, [n])
        self.assertEqual(a_3.gamsRepr(), 'a(n,"n1","i1")')

        a_3_2 = a.t()[:, "i1", "n1"]
        self.assertEqual(a_3_2.domain, [n])
        self.assertEqual(a_3_2.gamsRepr(), 'a(n,"n1","i1")')

        a_4 = a[:, :, "i1"][:, "n1"]
        self.assertEqual(a_4.domain, [n])
        self.assertEqual(a_4.gamsRepr(), 'a(n,"n1","i1")')

        a_4_2 = a.t()[:, "i1", :][:, "n1"]
        self.assertEqual(a_4_2.domain, [n])
        self.assertEqual(a_4_2.gamsRepr(), 'a(n,"n1","i1")')

        a_4_3 = a[:, :, "i1"].t()[:, "n1"]
        self.assertEqual(a_4_3.domain, [n2])
        self.assertEqual(a_4_3.gamsRepr(), 'a("n1",n2,"i1")')

        a_5 = a[:, "n1", :][:, "i1"]
        self.assertEqual(a_5.domain, [n])
        self.assertEqual(a_5.gamsRepr(), 'a(n,"n1","i1")')

        a_5_2 = a.t()[:, :, "n1"][:, "i1"]
        self.assertEqual(a_5_2.domain, [n])
        self.assertEqual(a_5_2.gamsRepr(), 'a(n,"n1","i1")')

        a_5_3 = a[:, "n1", :].t()["i1", :]
        self.assertEqual(a_5_3.domain, [n])
        self.assertEqual(a_5_3.gamsRepr(), 'a(n,"n1","i1")')

        # a_6 = (-(a[:, "n1", :])).t()
        # self.assertEqual(a_6.domain, [i, n])
        # self.assertEqual(a_6.gamsRepr(), '-a(n,"n1",i)')

        a_6_2 = -((a[:, "n1", :]).t())
        self.assertEqual(a_6_2.domain, [i, n])
        self.assertEqual(a_6_2.gamsRepr(), '( - a(n,"n1",i))')

        # a_6_3 = ((-a)[:, "n1", :]).t()
        # self.assertEqual(a_6_3.domain, [i, n])
        # self.assertEqual(a_6_3.gamsRepr(), '-a(n,"n1",i)')

        # a_6_4 = ((-a).t())[:, :, "n1"]
        # self.assertEqual(a_6_4.domain, [n, i])
        # self.assertEqual(a_6_4.gamsRepr(), '-a(n,"n1",i)')

        a_6_5 = (-a.t())[:, :, "n1"]
        self.assertEqual(a_6_5.domain, [n, i])
        self.assertEqual(a_6_5.gamsRepr(), '( - a(n,"n1",i))')

        a_6_6 = -((a.t())[:, :, "n1"])
        self.assertEqual(a_6_6.domain, [n, i])
        self.assertEqual(a_6_6.gamsRepr(), '( - a(n,"n1",i))')

    def test_shift_permute(self):
        i = Set(self.m, name="i", records=["i1", "i2", "i3"])
        j = Set(self.m, name="j", records=["j1", "j2", "j3"])
        k = Set(self.m, name="k", records=["k1", "k2", "k3"])
        a = Parameter(self.m, name="a", domain=[i, j, k])
        # same permute
        a0 = permute(a, [0, 1, 2])
        self.assertEqual(a0.domain, [i, j, k])
        self.assertEqual(a0["i1", "j1", "k1"].gamsRepr(), 'a("i1","j1","k1")')

        # left shift
        a1 = permute(a, [1, 2, 0])
        self.assertEqual(a1.domain, [j, k, i])
        self.assertEqual(a1["j1", "k1", "i1"].gamsRepr(), 'a("i1","j1","k1")')

        a2 = permute(a1, [1, 2, 0])
        self.assertEqual(a2.domain, [k, i, j])
        self.assertEqual(a2["k1", "i1", "j1"].gamsRepr(), 'a("i1","j1","k1")')

        a3 = permute(a2, [1, 2, 0])
        self.assertEqual(a3.domain, [i, j, k])
        self.assertEqual(a3["i1", "j1", "k1"].gamsRepr(), 'a("i1","j1","k1")')

        # right shift
        a1 = permute(a, [2, 0, 1])
        self.assertEqual(a1.domain, [k, i, j])
        self.assertEqual(a1["k1", "i1", "j1"].gamsRepr(), 'a("i1","j1","k1")')

        a2 = permute(a1, [2, 0, 1])
        self.assertEqual(a2.domain, [j, k, i])
        self.assertEqual(a2["j1", "k1", "i1"].gamsRepr(), 'a("i1","j1","k1")')

        a3 = permute(a2, [2, 0, 1])
        self.assertEqual(a3.domain, [i, j, k])
        self.assertEqual(a3["i1", "j1", "k1"].gamsRepr(), 'a("i1","j1","k1")')

    def test_permute_bruteforce_var(self):
        i = Set(self.m, name="i", records=["i1", "i2", "i3"])
        j = Set(self.m, name="j", records=["j1", "j2", "j3"])
        k = Set(self.m, name="k", records=["k1", "k2", "k3"])
        l = Set(self.m, name="l", records=["l1", "l2", "l3"])
        a = Variable(self.m, name="a", domain=[i, j, k, l])
        ax = a
        for perm in itertools.permutations(range(4)):
            ax = permute(ax, perm)

        # computed via pytorch
        self.assertEqual(ax.domain, [k, l, i, j])

    def test_permute_bruteforce_par(self):
        i = Set(self.m, name="i", records=["i1", "i2", "i3"])
        j = Set(self.m, name="j", records=["j1", "j2", "j3"])
        k = Set(self.m, name="k", records=["k1", "k2", "k3"])
        l = Set(self.m, name="l", records=["l1", "l2", "l3"])
        a = Parameter(self.m, name="a", domain=[i, j, k, l])
        ax = a
        # we trust that itertools will give permutations always in same order
        for perm in itertools.permutations(range(4)):
            ax = permute(ax, perm)

        # computed via pytorch
        self.assertEqual(ax.domain, [k, l, i, j])

    def test_permute_bad(self):
        i = Set(self.m, name="i", records=["i1", "i2", "i3"])
        j = Set(self.m, name="j", records=["j1", "j2", "j3"])
        k = Set(self.m, name="k", records=["k1", "k2", "k3"])
        l = Set(self.m, name="l", records=["l1", "l2", "l3"])
        a = Parameter(self.m, name="a", domain=[i, j, k, l])
        self.assertRaises(ValidationError, lambda: permute(a, [2, 2, 2, 2]))
        self.assertRaises(ValidationError, lambda: permute(a, [5, 2, 2, 2]))
        self.assertRaises(ValidationError, lambda: permute(a, [-1, 2, 2, 2]))
        self.assertRaises(ValidationError, lambda: permute(a, [0, 2, 2, 3]))
        self.assertRaises(ValidationError, lambda: permute(a, ["1", 2, 3, 4]))

    def test_transpose(self):
        i = Set(self.m, name="i", records=["i1", "i2", "i3"])
        j = Set(self.m, name="j", records=["j1", "j2", "j3"])
        k = Set(self.m, name="k", records=["k1", "k2", "k3"])
        l = Set(self.m, name="l", records=["l1", "l2", "l3"])
        a = Parameter(self.m, name="a", domain=[i, j, k, l])
        a_t = a.t()
        self.assertEqual(a_t.domain, [i, j, l, k])
        self.assertEqual(
            a_t["i1", "j1", "l1", "k1"].gamsRepr(), 'a("i1","j1","k1","l1")'
        )

        a_t = a.T
        self.assertEqual(a_t.domain, [i, j, l, k])
        self.assertEqual(
            a_t["i1", "j1", "l1", "k1"].gamsRepr(), 'a("i1","j1","k1","l1")'
        )

        b = Parameter(self.m, name="b", domain=[i])
        a2 = Variable(self.m, name="a2", domain=[i])
        self.assertRaises(ValidationError, lambda: b.T)  # par
        self.assertRaises(ValidationError, lambda: a2.T)  # var
        self.assertRaises(ValidationError, lambda: b[i].T)  # imp par
        self.assertRaises(ValidationError, lambda: a2[i].T)  # imp var


def matrix_suite():
    suite = unittest.TestSuite()
    tests = [
        MatrixSuite(name)
        for name in dir(MatrixSuite)
        if name.startswith("test_")
    ]
    suite.addTests(tests)
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(matrix_suite())
