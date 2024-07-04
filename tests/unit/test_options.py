from __future__ import annotations

import os
import sys
import unittest

import gamspy.exceptions as exceptions
import gamspy.math as math
from gamspy import (
    Container,
    Equation,
    Model,
    Options,
    Parameter,
    Sense,
    Set,
    Sum,
    Variable,
)
from pydantic import ValidationError


class OptionsSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container(
            system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None)
        )
        self.canning_plants = ["seattle", "san-diego"]
        self.markets = ["new-york", "chicago", "topeka"]
        self.distances = [
            ["seattle", "new-york", 2.5],
            ["seattle", "chicago", 1.7],
            ["seattle", "topeka", 1.8],
            ["san-diego", "new-york", 2.5],
            ["san-diego", "chicago", 1.8],
            ["san-diego", "topeka", 1.4],
        ]
        self.capacities = [["seattle", 350], ["san-diego", 600]]
        self.demands = [["new-york", 325], ["chicago", 300], ["topeka", 275]]

    def test_options(self):
        with self.assertRaises(ValidationError):
            _ = Options(unknown_option=5)

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
        m = Container(
            system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None),
            options=Options(seed=1),
        )
        p1 = Parameter(m, "p1")
        p1[...] = math.normal(0, 1)
        self.assertEqual(p1.records.value.item(), 0.45286287828275534)

        p2 = Parameter(m, "p2")
        p2[...] = math.normal(0, 1)
        self.assertEqual(p2.records.value.item(), -0.4841775276628964)

        # change seed
        m = Container(
            system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None),
            options=Options(seed=5),
        )
        p1 = Parameter(m, "p1")
        p1[...] = math.normal(0, 1)
        self.assertEqual(p1.records.value.item(), 0.14657004110784333)

        p2 = Parameter(m, "p2")
        p2[...] = math.normal(0, 1)
        self.assertEqual(p2.records.value.item(), 0.11165956511164217)

    def test_global_options(self):
        options = Options(lp="conopt")
        m = Container(
            system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None),
            debugging_level="keep",
            options=options,
        )

        i = Set(m, name="i", records=self.canning_plants)
        j = Set(m, name="j", records=self.markets)

        a = Parameter(m, name="a", domain=[i], records=self.capacities)
        b = Parameter(m, name="b", domain=[j], records=self.demands)
        d = Parameter(m, name="d", domain=[i, j], records=self.distances)
        c = Parameter(m, name="c", domain=[i, j])
        c[i, j] = 90 * d[i, j] / 1000

        x = Variable(m, name="x", domain=[i, j], type="Positive")

        supply = Equation(m, name="supply", domain=[i])
        demand = Equation(m, name="demand", domain=[j])

        supply[i] = Sum(j, x[i, j]) <= a[i]
        demand[j] = Sum(i, x[i, j]) >= b[j]

        transport = Model(
            m,
            name="transport",
            equations=m.getEquations(),
            problem="LP",
            sense=Sense.MIN,
            objective=Sum((i, j), c[i, j] * x[i, j]),
        )
        transport.solve()

        with open(
            os.path.join(m.working_directory, m.gamsJobName() + ".pf")
        ) as file:
            self.assertTrue("lp = conopt" in file.read())

    def test_gamspy_to_gams_options(self):
        options = Options(
            allow_suffix_in_equation=False,
            allow_suffix_in_limited_variables=False,
            merge_strategy="replace",
        )
        gams_options = options._get_gams_compatible_options(output=None)
        self.assertTrue(gams_options["suffixalgebravars"] == "off")
        self.assertTrue(gams_options["suffixdlvars"] == "off")
        self.assertTrue(gams_options["solveopt"] == 0)

    def test_log_option(self):
        i = Set(self.m, name="i", records=self.canning_plants)
        j = Set(self.m, name="j", records=self.markets)

        a = Parameter(self.m, name="a", domain=[i], records=self.capacities)
        b = Parameter(self.m, name="b", domain=[j], records=self.demands)
        d = Parameter(self.m, name="d", domain=[i, j], records=self.distances)
        c = Parameter(self.m, name="c", domain=[i, j])
        c[i, j] = 90 * d[i, j] / 1000

        x = Variable(self.m, name="x", domain=[i, j], type="Positive")

        supply = Equation(self.m, name="supply", domain=[i])
        demand = Equation(self.m, name="demand", domain=[j])

        supply[i] = Sum(j, x[i, j]) <= a[i]
        demand[j] = Sum(i, x[i, j]) >= b[j]

        transport = Model(
            self.m,
            name="transport",
            equations=self.m.getEquations(),
            problem="LP",
            sense=Sense.MIN,
            objective=Sum((i, j), c[i, j] * x[i, j]),
        )

        transport.solve()  # logoption=0
        transport.solve(output=sys.stdout)  # logoption = 3
        logfile_name = os.path.join(os.getcwd(), "tmp", "log.txt")
        transport.solve(
            output=sys.stdout,
            options=Options(log_file=logfile_name),
        )  # logoption = 4

        # test logfile
        transport.solve(
            options=Options(log_file=logfile_name)
        )  # logoption = 2
        self.assertTrue(os.path.exists(logfile_name))

        # test listing file
        listing_file_name = os.path.join(os.getcwd(), "tmp", "listing.lst")
        transport.solve(options=Options(listing_file=listing_file_name))
        self.assertTrue(os.path.exists(listing_file_name))

        listing_file_name = os.path.join("tmp", "listing2.lst")
        transport.solve(options=Options(listing_file=listing_file_name))
        self.assertTrue(os.path.exists(listing_file_name))

    def test_from_file(self):
        option_file = os.path.join("tmp", "option_file")
        with open(option_file, "w") as file:
            file.write("lp = conopt\n\n")

        options = Options.from_file(option_file)
        self.assertEqual(options.lp, "conopt")

        with self.assertRaises(exceptions.ValidationError):
            _ = Options.from_file("unknown_path")

    def test_profile(self):
        # Set
        i = Set(
            self.m,
            name="i",
            records=self.canning_plants,
            description="canning plants",
        )
        j = Set(
            self.m,
            name="j",
            records=self.markets,
            description="markets",
        )

        # Data
        a = Parameter(
            self.m,
            name="a",
            domain=i,
            records=self.capacities,
            description="capacity of plant i in cases",
        )
        b = Parameter(
            self.m,
            name="b",
            domain=j,
            records=self.demands,
            description="demand at market j in cases",
        )
        d = Parameter(
            self.m,
            name="d",
            domain=[i, j],
            records=self.distances,
            description="distance in thousands of miles",
        )
        c = Parameter(
            self.m,
            name="c",
            domain=[i, j],
            description="transport cost in thousands of dollars per case",
        )
        c[i, j] = 90 * d[i, j] / 1000

        # Variable
        x = Variable(
            self.m,
            name="x",
            domain=[i, j],
            type="Positive",
            description="shipment quantities in cases",
        )

        # Equation
        supply = Equation(
            self.m,
            name="supply",
            domain=i,
            description="observe supply limit at plant i",
        )
        demand = Equation(
            self.m,
            name="demand",
            domain=j,
            description="satisfy demand at market j",
        )

        supply[i] = Sum(j, x[i, j]) <= a[i]
        demand[j] = Sum(i, x[i, j]) >= b[j]

        transport = Model(
            self.m,
            name="transport",
            equations=self.m.getEquations(),
            problem="LP",
            sense=Sense.MIN,
            objective=Sum((i, j), c[i, j] * x[i, j]),
        )

        profile_path = os.path.join("tmp", "bla.profile")
        transport.solve(
            output=sys.stdout,
            options=Options(
                profile=1,
                profile_file=profile_path,
                monitor_process_tree_memory=True,
            ),
        )
        self.assertTrue(os.path.exists(profile_path))


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
