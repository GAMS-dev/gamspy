from __future__ import annotations

import os
import unittest

from gamspy import Alias
from gamspy import Container
from gamspy import Parameter
from gamspy import Set
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

        a = Parameter(self.m, name="a", domain=[i, j])
        b = Parameter(self.m, name="b", domain=[j, k])
        c = a @ b
        self.assertEqual(c.domain, [i, k])

    def test_simple_matrix_vector(self):
        """Test simple case where domain calculation is trivial
        matrix x vector"""
        i = Set(self.m, name="i", records=["i1", "i2", "i3"])
        j = Set(self.m, name="j", records=["j1", "j2", "j3"])

        a = Parameter(self.m, name="a", domain=[i, j])
        b = Parameter(self.m, name="b", domain=[j])
        c = a @ b
        self.assertEqual(c.domain, [i])

    def test_simple_vector_vector(self):
        """Test simple case where domain calculation is trivial
        vector x vector, aka inner product"""
        i = Set(self.m, name="i", records=["i1", "i2", "i3"])

        a = Parameter(self.m, name="a", domain=[i])
        b = Parameter(self.m, name="b", domain=[i])
        c = a @ b
        self.assertEqual(c.domain, [])

    def test_simple_vector_matrix(self):
        """Test simple case where domain calculation is trivial
        vector x matrix"""
        i = Set(self.m, name="i", records=["i1", "i2", "i3"])
        j = Set(self.m, name="j", records=["j1", "j2", "j3"])

        a = Parameter(self.m, name="a", domain=[i])
        b = Parameter(self.m, name="b", domain=[i, j])
        c = a @ b
        self.assertEqual(c.domain, [j])

    def test_batched_matrix_matrix(self):
        """Test batched matrix multiplication,
        batched matrix x batched matrix"""
        n = Set(self.m, name="n", records=["n1", "n2", "n3"])
        i = Set(self.m, name="i", records=["i1", "i2", "i3"])
        j = Set(self.m, name="j", records=["j1", "j2", "j3"])
        k = Set(self.m, name="k", records=["k1", "k2", "k3"])

        a = Parameter(self.m, name="a", domain=[n, i, j])
        b = Parameter(self.m, name="b", domain=[n, j, k])
        c = a @ b
        self.assertEqual(c.domain, [n, i, k])

    def test_batched_matrix_matrix_2(self):
        """Test batched matrix multiplication,
        batched matrix x matrix"""
        n = Set(self.m, name="n", records=["n1", "n2", "n3"])
        i = Set(self.m, name="i", records=["i1", "i2", "i3"])
        j = Set(self.m, name="j", records=["j1", "j2", "j3"])
        k = Set(self.m, name="k", records=["k1", "k2", "k3"])

        a = Parameter(self.m, name="a", domain=[n, i, j])
        b = Parameter(self.m, name="b", domain=[j, k])
        c = a @ b

        self.assertEqual(c.domain, [n, i, k])

    def test_square_matrix_mult(self):
        i = Set(self.m, name="i", records=["i1", "i2", "i3"])
        j = Alias(self.m, name="j", alias_with=i)
        k = Alias(self.m, name="k", alias_with=j)

        a = Parameter(self.m, name="a", domain=[i, i])
        b = Parameter(self.m, name="b", domain=[i, i])
        self.assertRaises(ValidationError, lambda: a @ b)

        # TODO what to do here?
        # c = a[i, j] @ b[j, i]
        # self.assertEqual(c.domain, [i, i])

        c2 = a[i, j] @ b[j, k]
        self.assertEqual(c2.domain, [i, k])

    def test_square_matrix_mult_2(self):
        n = Set(self.m, name="n", records=["n1", "n2", "n3"])
        i = Set(self.m, name="i", records=["i1", "i2", "i3"])
        j = Alias(self.m, name="j", alias_with=i)
        k = Alias(self.m, name="k", alias_with=j)

        a = Parameter(self.m, name="a", domain=[n, i, i])
        b = Parameter(self.m, name="b", domain=[n, i, i])
        self.assertRaises(ValidationError, lambda: a @ b)

        # TODO what to do here?
        # c = a[n, i, j] @ b[n, j, i]
        # self.assertEqual(c.domain, [n, i, i])

        c2 = a[n, i, j] @ b[n, j, k]
        self.assertEqual(c2.domain, [n, i, k])


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
