import platform
import unittest

import gamspy.utils as utils
from gamspy import Container
from gamspy import Set
from gamspy._algebra.domain import DomainException
from gamspy.exceptions import GamspyException


class UtilsSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container(delayed_execution=True)

    def test_utils(self):
        string = "(bla))"
        self.assertRaises(
            AssertionError, utils._getMatchingParanthesisIndices, string
        )

        string2 = "((bla)"
        self.assertRaises(
            AssertionError, utils._getMatchingParanthesisIndices, string2
        )

        i = Set(self.m, "i", records=["i1", "i2"])
        self.assertEqual(utils._getDomainStr([i, "b", "*"]), '(i,"b",*)')
        self.assertRaises(DomainException, utils._getDomainStr, [5])

        # invalid system directory
        self.assertRaises(GamspyException, utils._openGdxFile, "bla", "bla")

        self.assertFalse(utils.checkAllSame([1, 2], [2]))
        self.assertFalse(utils.checkAllSame([1, 2], [2, 3]))
        self.assertTrue(utils.checkAllSame([1, 2], [1, 2]))

        # invalid load from path
        self.assertRaises(
            Exception, utils._openGdxFile, self.m.system_directory, "bla.gdx"
        )

    def test_isin(self):
        i = Set(self.m, "i")
        j = Set(self.m, "j")
        k = Set(self.m, "k")
        symbols = [i, j]

        self.assertTrue(utils.isin(i, symbols))
        self.assertFalse(utils.isin(k, symbols))

    def test_available_solvers(self):
        installed_solvers = utils.getInstalledSolvers()

        self.assertEqual(
            installed_solvers,
            [
                "CONOPT",
                "CONVERT",
                "CPLEX",
                "NLPEC",
                "PATH",
                "SBB",
            ],
        )

        def get_platform() -> str:
            operating_system = platform.system().lower()
            architecture = platform.machine()

            if operating_system == "darwin":
                return f"mac_{architecture}"

            return operating_system

        system = get_platform()

        available_solvers = utils.getAvailableSolvers()

        expected = [
            "NLPEC",
            "SBB",
            "CONOPT",
            "CONVERT",
            "CPLEX",
            "PATH",
            "BARON",
            "CONOPT4",
            "COPT",
            "DICOPT",
            "GUROBI",
            "HIGHS",
            "IPOPT",
            "IPOPTH",
            "KNITRO",
            "MINOS",
            "MOSEK",
            "SCIP",
            "SHOT",
            "SNOPT",
            "XPRESS",
        ]

        if system == "mac_arm64":
            expected.remove("XPRESS")

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
