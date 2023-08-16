import unittest

import gamspy.utils as utils
from gamspy import (
    Container,
    Set,
)


class UtilsSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container()

    def test_utils(self):
        string = "(bla))"
        self.assertRaises(
            Exception, utils._getMatchingParanthesisIndices, string
        )

        string2 = "((bla)"
        self.assertRaises(
            Exception, utils._getMatchingParanthesisIndices, string2
        )

        i = Set(self.m, "i", records=["i1", "i2"])
        self.assertEqual(utils._getDomainStr([i, "b", "*"]), '(i,"b",*)')
        self.assertRaises(Exception, utils._getDomainStr, [5])

        # invalid system directory
        self.assertRaises(utils.GdxException, utils._openGdxFile, "bla", "bla")

        self.assertFalse(utils.checkAllSame([1, 2], [2]))
        self.assertFalse(utils.checkAllSame([1, 2], [2, 3]))
        self.assertTrue(utils.checkAllSame([1, 2], [1, 2]))

        # invalid load from path
        self.assertRaises(
            Exception, utils._openGdxFile, self.m.system_directory, "bla.gdx"
        )

        # invalid symbol
        self.assertRaises(Exception, utils._getSymbolData, None, None, "i")

    def test_isin(self):
        i = Set(self.m, "i")
        j = Set(self.m, "j")
        k = Set(self.m, "k")
        symbols = [i, j]

        self.assertTrue(utils.isin(i, symbols))
        self.assertFalse(utils.isin(k, symbols))


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
