from __future__ import annotations

import os
import unittest

import gamspy.utils as utils
from gamspy import Container, Set
from gamspy.exceptions import GamspyException, ValidationError


class UtilsSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
        )

    def test_utils(self):
        string = "(bla))"
        self.assertRaises(
            AssertionError, utils._get_matching_paranthesis_indices, string
        )

        string2 = "((bla)"
        self.assertRaises(
            AssertionError, utils._get_matching_paranthesis_indices, string2
        )

        i = Set(self.m, "i", records=["i1", "i2"])
        self.assertEqual(utils._get_domain_str([i, "b", "*"]), '(i,"b",*)')
        self.assertRaises(ValidationError, utils._get_domain_str, [5])

        # invalid system directory
        self.assertRaises(GamspyException, utils._open_gdx_file, "bla", "bla")

        self.assertFalse(utils.checkAllSame([1, 2], [2]))
        self.assertFalse(utils.checkAllSame([1, 2], [2, 3]))
        self.assertTrue(utils.checkAllSame([1, 2], [1, 2]))

        # invalid load from path
        self.assertRaises(
            Exception, utils._open_gdx_file, self.m.system_directory, "bla.gdx"
        )

    def test_isin(self):
        i = Set(self.m, "i")
        j = Set(self.m, "j")
        k = Set(self.m, "k")
        symbols = [i, j]

        self.assertTrue(utils.isin(i, symbols))
        self.assertFalse(utils.isin(k, symbols))

    def test_available_solvers(self):
        available_solvers = utils.getAvailableSolvers()

        expected = [
            "BARON",
            "CBC",
            "CONOPT",
            "CONOPT3",
            "CONVERT",
            "COPT",
            "CPLEX",
            "DICOPT",
            "GUROBI",
            "HIGHS",
            "IPOPT",
            "IPOPTH",
            "KNITRO",
            "MINOS",
            "MOSEK",
            "MPSGE",
            "NLPEC",
            "PATH",
            "SBB",
            "SCIP",
            "SHOT",
            "SNOPT",
            "XPRESS",
        ]

        self.assertEqual(available_solvers, expected)


def utils_suite():
    suite = unittest.TestSuite()
    tests = [
        UtilsSuite(name)
        for name in dir(UtilsSuite)
        if name.startswith("test_")
    ]
    suite.addTests(tests)

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(utils_suite())
