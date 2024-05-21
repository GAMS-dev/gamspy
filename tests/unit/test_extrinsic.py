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
        names = {"Linux": "libtricclib64.so", "Darwin": "libtricclib64.dylib", "Windows": "tricclib64.dll"}
        directory = os.path.dirname(os.path.abspath(__file__))
        shared_object = os.path.join(directory, names[platform.system()])

        # This is a library which contains myNum=1.
        trilib = self.m.importExtrinsicLibrary(
            shared_object,
            "trilib",
            functions={
                "myPi": "Pi",
                "myCos": "Cosine",
            },
        )

        # Test extrinsic function with no argument
        d = Parameter(self.m, "d")
        d[...] = trilib.myPi
        self.assertEqual(d.toValue(), 3.141592653589793238462643)

        # Test extrinsic function with one argument
        d2 = Parameter(self.m, "d2")
        d2[...] = trilib.myCos(90)
        self.assertEqual(int(d2.toValue()), 0)

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
