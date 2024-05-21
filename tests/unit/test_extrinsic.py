from __future__ import annotations

import os
import platform
import unittest

from gamspy import Container, Parameter


class ExtrinsicSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
        )

    def test_extrinsic_functions(self):
        directory = os.path.dirname(os.path.abspath(__file__))

        if platform.system() == "Linux":
            # This is a library which contains myNum=1.
            my_lib = self.m.importExtrinsicLibrary(
                f"{directory}/libextrinsic.so",
                "mine",
                functions={
                    "getMyNum": "getMyNum",
                    "multiplyMyNum": "multiplyMyNum",
                },
            )

            # Test extrinsic function with no argument
            d = Parameter(self.m, "d")
            d[...] = my_lib.getMyNum()
            self.assertEqual(int(d.toValue()), 1)

            # Test extrinsic function with one argument
            d2 = Parameter(self.m, "d2")
            d2[...] = my_lib.multiplyMyNum(5)
            self.assertEqual(int(d2.toValue()), 5)

            # Test the interaction with other components
            d3 = Parameter(self.m, "d3")
            d3[...] = my_lib.multiplyMyNum(5) * 5
            self.assertEqual(int(d3.toValue()), 25)

            d4 = Parameter(self.m, records=10)
            d4[...] = d4 * my_lib.multiplyMyNum(5)
            self.assertEqual(int(d4.toValue()), 50)


def extrinsic_suite():
    suite = unittest.TestSuite()
    tests = [
        ExtrinsicSuite(name)
        for name in dir(ExtrinsicSuite)
        if name.startswith("test_")
    ]
    suite.addTests(tests)

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(extrinsic_suite())
