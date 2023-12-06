from __future__ import annotations

import unittest

from gams import GamsExceptionExecution
from pydantic import ValidationError

import gamspy.math as math
from gamspy import Container
from gamspy import Equation
from gamspy import Options
from gamspy import Parameter
from gamspy import Set
from gamspy import Sum
from gamspy import Variable


class OptionsSuite(unittest.TestCase):
    def test_options(self):
        with self.assertRaises(ValidationError):
            _ = Options(hold_fixed_variables=5)

        options = Options(hold_fixed_variables=True)
        self.assertEqual(options.hold_fixed_variables, 1)

        with self.assertRaises(ValidationError):
            _ = Options(report_solution=5)

        options = Options(report_solution=1)
        self.assertEqual(options.report_solution, 1)

        with self.assertRaises(ValidationError):
            _ = Options(merge_strategy=5)

        options = Options(merge_strategy="replace")
        self.assertEqual(options.merge_strategy, "replace")

        with self.assertRaises(ValidationError):
            _ = Options(step_summary=5)

        options = Options(step_summary=True)
        self.assertEqual(options.step_summary, True)

        with self.assertRaises(ValidationError):
            _ = Options(suppress_compiler_listing=5)

        options = Options(suppress_compiler_listing=True)
        self.assertEqual(options.suppress_compiler_listing, True)

        with self.assertRaises(ValidationError):
            _ = Options(report_solver_status=5)

        options = Options(report_solver_status=True)
        self.assertEqual(options.report_solver_status, True)

        with self.assertRaises(ValidationError):
            _ = Options(report_underflow=5)

        options = Options(report_underflow=True)
        self.assertEqual(options.report_underflow, True)

    def test_seed(self):
        m = Container(options=Options(seed=1))
        p1 = Parameter(m, "p1")
        p1[...] = math.normal(0, 1)
        self.assertEqual(p1.records.value.item(), 0.45286287828275534)

        p2 = Parameter(m, "p2")
        p2[...] = math.normal(0, 1)
        self.assertEqual(p2.records.value.item(), -0.4841775276628964)

        # change seed
        m = Container(options=Options(seed=5))
        p1 = Parameter(m, "p1")
        p1[...] = math.normal(0, 1)
        self.assertEqual(p1.records.value.item(), 0.14657004110784333)

        p2 = Parameter(m, "p2")
        p2[...] = math.normal(0, 1)
        self.assertEqual(p2.records.value.item(), 0.11165956511164217)

    def test_global_options(self):
        m = Container(
            delayed_execution=False, options=Options(write_listing_file=False)
        )

        # Prepare data
        distances = [
            ["seattle", "new-york", 2.5],
            ["seattle", "chicago", 1.7],
            ["seattle", "topeka", 1.8],
            ["san-diego", "new-york", 2.5],
            ["san-diego", "chicago", 1.8],
            ["san-diego", "topeka", 1.4],
        ]

        capacities = [["seattle", 350], ["san-diego", 600]]
        demands = [["new-york", 325], ["chicago", 300], ["topeka", 275]]

        # Set
        i = Set(m, name="i", records=["seattle", "san-diego"])
        j = Set(m, name="j", records=["new-york", "chicago", "topeka"])

        # Data
        a = Parameter(m, name="a", domain=[i], records=capacities)
        b = Parameter(m, name="b", domain=[j], records=demands)
        d = Parameter(m, name="d", domain=[i, j], records=distances)
        c = Parameter(m, name="c", domain=[i, j])
        c[i, j] = 90 * d[i, j] / 1000

        # Variable
        x = Variable(m, name="x", domain=[i, j], type="Positive")

        # Equation
        supply = Equation(m, name="supply", domain=[i])
        demand = Equation(m, name="demand", domain=[j])

        supply[i] = Sum(j, x[i, j]) <= a[i]
        with self.assertRaises(GamsExceptionExecution):
            demand[i] = Sum(i, x[i, j]) >= b[j]

    def test_gamspy_to_gams_options(self):
        options = Options(
            allow_suffix_in_equation=False,
            allow_suffix_in_limited_variables=False,
            merge_strategy="replace",
        )
        gams_options = options._getGamsCompatibleOptions()
        self.assertTrue(gams_options["suffixalgebravars"] == "off")
        self.assertTrue(gams_options["suffixdlvars"] == "off")
        self.assertTrue(gams_options["solveopt"] == 0)


def options_suite():
    suite = unittest.TestSuite()
    tests = [
        OptionsSuite(name)
        for name in dir(OptionsSuite)
        if name.startswith("test_")
    ]
    suite.addTests(tests)

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(options_suite())
