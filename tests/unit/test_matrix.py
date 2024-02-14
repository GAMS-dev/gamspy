from __future__ import annotations

import os
import unittest

import numpy as np
from gamspy import Alias, Container, Parameter, Set
from gamspy.exceptions import ValidationError


class MatrixSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
            delayed_execution=int(os.getenv("DELAYED_EXECUTION", False)),
        )

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
        c[...] = (a @ b)[i, k]  # TODO check this, this would not work o.w
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

    def test_batch_size_matches(self):
        n = Set(self.m, name="n", records=["n1", "n2", "n3"])
        m = Set(self.m, name="m", records=["m1", "m2", "m3"])
        i = Set(self.m, name="i", records=["i1", "i2", "i3"])
        j = Alias(self.m, name="j", alias_with=i)
        k = Alias(self.m, name="k", alias_with=j)

        a = Parameter(self.m, name="a", domain=[n, i, j])
        b = Parameter(self.m, name="b", domain=[m, j, k])

        self.assertRaises(ValidationError, lambda: a @ b)

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
