from __future__ import annotations

import os
import platform
import unittest

from gamspy import Container, Parameter
from gamspy.exceptions import ValidationError


def get_default_platform():
    operating_system = platform.system().lower()
    architecture = platform.machine()

    if operating_system == "darwin":
        return f"mac_{architecture}"

    return operating_system


class ExtrinsicSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container()

    def test_extrinsic_functions(self):
        # Invalid path
        with self.assertRaises(FileNotFoundError):
            _ = self.m.importExtrinsicLibrary(
                "blablablalib.so",
                functions={
                    "myPi": "Pi",
                    "myCos": "Cosine",
                },
            )

        names = {
            "linux": "libtricclib64.so",
            "mac_x86_64": "libtricclib64x86.dylib",
            "mac_arm64": "libtricclib64arm.dylib",
            "windows": "tricclib64.dll",
        }
        directory = os.path.dirname(os.path.abspath(__file__))
        shared_object = os.path.join(directory, names[get_default_platform()])

        # This is a library which contains myNum=1.
        trilib = self.m.importExtrinsicLibrary(
            shared_object,
            functions={
                "myPi": "Pi",
                "myCos": "Cosine",
            },
        )

        # if attribute is not in functions, call the default getattr function of Python
        with self.assertRaises(AttributeError):
            _ = trilib.bla

        # Test extrinsic function with no argument
        d = Parameter(self.m, "d")
        d[...] = trilib.myPi
        self.assertEqual(d.toValue(), 3.141592653589793238462643)

        # Test extrinsic function with one argument
        d2 = Parameter(self.m, "d2")
        d2[...] = trilib.myCos(90)
        self.assertEqual(int(d2.toValue()), 0)

        # External functions do not accept keyword arguments
        with self.assertRaises(ValidationError):
            d2[...] = trilib.myCos(degree=90)

        # Test the interaction with other components
        d3 = Parameter(self.m, "d3")
        d3[...] = trilib.myCos(90, 1) * 3
        self.assertEqual(int(d3.toValue()), 0)


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
