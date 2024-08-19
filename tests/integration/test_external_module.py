from __future__ import annotations

import math
import os
import pathlib
import platform
import sys
import unittest

import gamspy as gp

try:
    from dotenv import load_dotenv

    load_dotenv(os.getcwd() + os.sep + ".env")
except Exception:
    pass


class ExternalModuleTestSuite(unittest.TestCase):
    def setUp(self):
        self.m = gp.Container()

        directory = str(pathlib.Path(__file__).parent.resolve())
        external_module = os.path.join(
            directory, "external_module", "build", "libsimple_ext_module"
        )

        if platform.system() == "Darwin" and platform.machine() == "arm64":
            external_module += "_arm64"

        self.external_module = external_module

    def test_sin_cos_example(self):
        y1 = gp.Variable(self.m, "y1")
        y2 = gp.Variable(self.m, "y2")
        x1 = gp.Variable(self.m, "x1")
        x2 = gp.Variable(self.m, "x2")

        eq1 = gp.Equation(self.m, "eq1", type="external")
        eq2 = gp.Equation(self.m, "eq2", type="external")

        eq1[...] = 1 * x1 + 3 * y1 == 1
        eq2[...] = 2 * x2 + 4 * y2 == 2

        model = gp.Model(
            container=self.m,
            name="sincos",
            equations=self.m.getEquations(),
            problem="NLP",
            sense="min",
            objective=y1 + y2,
            external_module=self.external_module,
        )

        model.solve(output=sys.stdout, solver="conopt")

        assert math.isclose(y1.toDense(), -1)
        assert math.isclose(y2.toDense(), -1)


def external_module_suite():
    suite = unittest.TestSuite()
    tests = [
        ExternalModuleTestSuite(name)
        for name in dir(ExternalModuleTestSuite)
        if name.startswith("test_")
    ]
    suite.addTests(tests)

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(external_module_suite())
