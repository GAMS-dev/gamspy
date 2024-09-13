from __future__ import annotations

import os
import subprocess
import sys
import unittest

import gamspy.exceptions as exceptions
import gamspy.math as math
from gamspy import (
    Alias,
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
from gamspy.exceptions import GamspyException
from pydantic import ValidationError


class OptionsSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container()
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
        with self.assertRaises(exceptions.ValidationError):
            options = Options(generate_name_dict=True)
            _ = Container(options=options)

        with self.assertRaises(exceptions.ValidationError):
            options = Options(loadpoint="bla.gdx")
            _ = Container(options=options)

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

        options = Options(solve_link_type="disk")
        self.assertEqual(
            options._get_gams_compatible_options()["solvelink"], 2
        )

    def test_seed(self):
        m = Container(
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
            self.assertTrue('lp = "conopt"' in file.read())

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

        options = Options.fromFile(option_file)
        self.assertEqual(options.lp, "conopt")

        with self.assertRaises(exceptions.ValidationError):
            _ = Options.fromFile("unknown_path")

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

        # solprint should be 0 by default
        with open(self.m.gamsJobName() + ".lst") as file:
            self.assertTrue("---- EQU supply" not in file.read())

    def test_solprint(self):
        m = Container(options=Options(report_solution=1))

        # Set
        i = Set(
            m,
            name="i",
            records=self.canning_plants,
            description="canning plants",
        )
        j = Set(
            m,
            name="j",
            records=self.markets,
            description="markets",
        )

        # Data
        a = Parameter(
            m,
            name="a",
            domain=i,
            records=self.capacities,
            description="capacity of plant i in cases",
        )
        b = Parameter(
            m,
            name="b",
            domain=j,
            records=self.demands,
            description="demand at market j in cases",
        )
        d = Parameter(
            m,
            name="d",
            domain=[i, j],
            records=self.distances,
            description="distance in thousands of miles",
        )
        c = Parameter(
            m,
            name="c",
            domain=[i, j],
            description="transport cost in thousands of dollars per case",
        )
        c[i, j] = 90 * d[i, j] / 1000

        # Variable
        x = Variable(
            m,
            name="x",
            domain=[i, j],
            type="Positive",
            description="shipment quantities in cases",
        )

        # Equation
        supply = Equation(
            m,
            name="supply",
            domain=i,
            description="observe supply limit at plant i",
        )
        demand = Equation(
            m,
            name="demand",
            domain=j,
            description="satisfy demand at market j",
        )

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

        # solprint is 1
        with open(m.gamsJobName() + ".lst") as file:
            self.assertTrue("---- EQU supply" in file.read())

    def test_exception_on_solve_with_listing_file(self):
        subprocess.run(["gamspy", "uninstall", "license"], check=True)

        m = Container()

        n = 500

        # SET #
        nh = Set(
            m,
            name="nh",
            records=[str(i) for i in range(n + 1)],
            description="Number of subintervals",
        )

        # ALIAS #
        k = Alias(m, name="k", alias_with=nh)

        # SCALARS #
        tf = Parameter(m, name="tf", records=10, description="final time")
        x1_0 = Parameter(
            m, name="x1_0", records=1, description="initial value for x1"
        )
        x2_0 = Parameter(
            m, name="x2_0", records=5, description="initial value for x2"
        )
        x3_0 = Parameter(
            m, name="x3_0", records=0, description="initial value for x3"
        )
        x4_0 = Parameter(
            m, name="x4_0", records=0, description="initial value for x4"
        )
        x5_0 = Parameter(
            m, name="x5_0", records=1, description="initial value for x5"
        )
        h = Parameter(m, name="h")
        h[...] = tf / n

        # VARIABLES #
        x1 = Variable(m, name="x1", domain=nh)
        x2 = Variable(m, name="x2", domain=nh)
        x3 = Variable(m, name="x3", domain=nh)
        x4 = Variable(m, name="x4", domain=nh)
        x5 = Variable(m, name="x5", domain=nh)
        u = Variable(m, name="u", domain=nh, description="control variable")
        a1 = Variable(m, name="a1", domain=nh)
        a2 = Variable(m, name="a2", domain=nh)
        a3 = Variable(m, name="a3", domain=nh)

        # EQUATIONS #
        state1 = Equation(
            m,
            name="state1",
            type="regular",
            domain=nh,
            description="state equation 1",
        )
        state2 = Equation(
            m,
            name="state2",
            type="regular",
            domain=nh,
            description="state equation 2",
        )
        state3 = Equation(
            m,
            name="state3",
            type="regular",
            domain=nh,
            description="state equation 3",
        )
        state4 = Equation(
            m,
            name="state4",
            type="regular",
            domain=nh,
            description="state equation 4",
        )
        state5 = Equation(
            m,
            name="state5",
            type="regular",
            domain=nh,
            description="state equation 5",
        )
        ea1 = Equation(m, name="ea1", type="regular", domain=nh)
        ea2 = Equation(m, name="ea2", type="regular", domain=nh)
        ea3 = Equation(m, name="ea3", type="regular", domain=nh)

        eobj = x4[str(n)] * x5[str(n)]  # Objective function

        state1[nh[k.lead(1)]] = x1[k.lead(1)] == (
            x1[k]
            + (h / 2)
            * (
                a1[k] * x1[k]
                - u[k] * x1[k] / x5[k]
                + a1[k.lead(1)] * x1[k.lead(1)]
                - u[k.lead(1)] * x1[k.lead(1)] / x5[k.lead(1)]
            )
        )

        state2[nh[k.lead(1)]] = x2[k.lead(1)] == (
            x2[k]
            + (h / 2)
            * (
                -7.3 * a1[k] * x1[k]
                - u[k] * (x2[k] - 20) / x5[k]
                - 7.3 * a1[k.lead(1)] * x1[k.lead(1)]
                - u[k.lead(1)] * (x2[k.lead(1)] - 20) / x5[k.lead(1)]
            )
        )

        state3[nh[k.lead(1)]] = x3[k.lead(1)] == (
            x3[k]
            + (h / 2)
            * (
                a2[k] * x1[k]
                - u[k] * x3[k] / x5[k]
                + a2[k.lead(1)] * x1[k.lead(1)]
                - u[k.lead(1)] * x3[k.lead(1)] / x5[k.lead(1)]
            )
        )

        state4[nh[k.lead(1)]] = x4[k.lead(1)] == (
            x4[k]
            + (h / 2)
            * (
                a3[k] * (x3[k] - x4[k])
                - u[k] * x4[k] / x5[k]
                + a3[k.lead(1)] * (x3[k.lead(1)] - x4[k.lead(1)])
                - u[k.lead(1)] * x4[k.lead(1)] / x5[k.lead(1)]
            )
        )

        state5[nh[k.lead(1)]] = x5[k.lead(1)] == x5[k] + (h / 2) * (
            u[k] + u[k.lead(1)]
        )

        ea1[nh[k]] = a1[k] == 21.87 * x2[k] / ((x2[k] + 0.4) * (x2[k] + 62.5))
        ea2[nh[k]] = a2[k] == (x2[k] * math.exp(-5 * x2[k])) / (0.1 + x2[k])
        ea3[nh[k]] = a3[k] == 4.75 * a1[k] / (0.12 + a1[k])

        # Initial point
        x1.l[nh] = 1.0
        x2.l[nh] = 5.0
        x3.l[nh] = 0.0
        x4.l[nh] = 0.0
        x5.l[nh] = 1.0
        u.l[nh] = 0.0

        x1.fx["0"] = x1_0
        x2.fx["0"] = x2_0
        x3.fx["0"] = x3_0
        x4.fx["0"] = x4_0
        x5.fx["0"] = x5_0

        # Bounds
        u.lo[nh] = 0.0
        u.up[nh] = -5

        protein = Model(
            m,
            name="protein",
            equations=m.getEquations(),
            problem="nlp",
            sense="max",
            objective=eobj,
        )

        with self.assertRaises(GamspyException):
            protein.solve(
                solver="CONOPT",
                options=Options(
                    time_limit=60000,
                    iteration_limit=80000,
                    listing_file=os.path.join("tmp", "bla.lst"),
                ),
            )

        subprocess.run(
            ["gamspy", "install", "license", os.environ["LOCAL_LICENSE"]],
            check=True,
        )

    def test_model_attribute_options(self):
        m = Container()

        # Set
        i = Set(
            m,
            name="i",
            records=["seattle", "san-diego"],
            description="canning plants",
        )
        j = Set(
            m,
            name="j",
            records=["new-york", "chicago", "topeka"],
            description="markets",
        )

        # Data
        a = Parameter(
            m,
            name="a",
            domain=i,
            records=self.capacities,
            description="capacity of plant i in cases",
        )
        b = Parameter(
            m,
            name="b",
            domain=j,
            records=self.demands,
            description="demand at market j in cases",
        )
        d = Parameter(
            m,
            name="d",
            domain=[i, j],
            records=self.distances,
            description="distance in thousands of miles",
        )
        c = Parameter(
            m,
            name="c",
            domain=[i, j],
            description="transport cost in thousands of dollars per case",
        )
        c[i, j] = 90 * d[i, j] / 1000

        # Variable
        x = Variable(
            m,
            name="x",
            domain=[i, j],
            type="Positive",
            description="shipment quantities in cases",
        )

        # Equation
        supply = Equation(
            m,
            name="supply",
            domain=i,
            description="observe supply limit at plant i",
        )
        demand = Equation(
            m,
            name="demand",
            domain=j,
            description="satisfy demand at market j",
        )

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
        transport.solve(options=Options(infeasibility_tolerance=1e-6))
        transport.solve(options=Options(infeasibility_tolerance=1e-6))
        self.assertTrue(
            "transport.tolInfRep = 1e-06;"
            in m.generateGamsString(show_raw=True)
        )

    def test_scaling(self):
        m = Container()

        x1 = Variable(m, "x1", type="positive")
        x2 = Variable(m, "x2", type="positive")

        z = Variable(m, "z")
        eq = Equation(m, "eq")
        eq[...] = 200 * x1 + 0.5 * x2 == z

        x1.up[...] = 0.01
        x2.up[...] = 10
        x1.scale[...] = 0.01
        x2.scale[...] = 10

        eq.scale = 1e-6

        model = Model(m, "my_model", equations=[eq], sense="MIN", objective=z)
        listing_file_path = os.path.join("tmp", "scaling.lst")
        model.solve(
            options=Options(
                enable_scaling=True,
                equation_listing_limit=100,
                listing_file=listing_file_path,
            )
        )
        self.assertEqual(eq.records.scale.item(), 1e-6)
        with open(listing_file_path) as file:
            self.assertTrue(
                "eq..  2000000*x1 + 5000000*x2 - 1000000*z =E= 0"
                in file.read()
            )

    def test_loadpoint(self):
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

        transport.solve(options=Options(savepoint=1))
        self.assertEqual(transport.num_iterations, 4)

        transport.solve(options=Options(loadpoint="transport_p.gdx"))
        self.assertEqual(transport.num_iterations, 0)


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
