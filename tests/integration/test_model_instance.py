from __future__ import annotations

import os
import unittest

from gamspy import (
    Container,
    Equation,
    Model,
    ModelInstanceOptions,
    Options,
    Parameter,
    Sense,
    Set,
    Sum,
    Variable,
)
from gamspy.exceptions import ValidationError


class ModelInstanceSuite(unittest.TestCase):
    def setUp(self):
        self.canning_plants = ["seattle", "san-diego"]
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

    def test_parameter_change(self):
        m = Container(system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None))
        i = Set(m, name="i", records=["seattle", "san-diego"])
        j = Set(m, name="j", records=["new-york", "chicago", "topeka"])

        a = Parameter(m, name="a", domain=[i], records=self.capacities)
        b = Parameter(m, name="b", domain=[j], records=self.demands)
        d = Parameter(m, name="d", domain=[i, j], records=self.distances)
        c = Parameter(m, name="c", domain=[i, j])
        bmult = Parameter(m, name="bmult", records=1)
        c[i, j] = 90 * d[i, j] / 1000

        x = Variable(m, name="x", domain=[i, j], type="Positive")
        z = Variable(m, name="z")

        cost = Equation(m, name="cost")
        supply = Equation(m, name="supply", domain=[i])
        demand = Equation(m, name="demand", domain=[j])

        cost[...] = z == Sum((i, j), c[i, j] * x[i, j])
        supply[i] = Sum(j, x[i, j]) <= a[i]
        demand[j] = Sum(i, x[i, j]) >= bmult * b[j]

        transport = Model(
            m,
            name="transport",
            equations=m.getEquations(),
            problem="LP",
            sense=Sense.MIN,
            objective=z,
        )

        bmult_list = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
        results = [
            92.204,
            107.572,
            122.940,
            138.307,
            153.675,
            169.94250000000002,
            185.58,
            201.21750000000003,
        ]

        transport.freeze(modifiables=[bmult])

        for b_value, result in zip(bmult_list, results):
            bmult[...] = b_value
            transport.solve(solver="conopt")
            self.assertTrue("bmult_var" in m.data)
            self.assertAlmostEqual(z.toValue(), result, places=2)
            self.assertAlmostEqual(transport.objective_value, result, places=2)

        # different solver
        transport.solve(solver="cplex")
        self.assertAlmostEqual(
            transport.objective_value, 199.77750000000003, places=2
        )

        # invalid solver
        with self.assertRaises(ValidationError):
            transport.solve(solver="blablabla")

        transport.unfreeze()
        self.assertFalse(transport._is_frozen)

    def test_variable_change(self):
        m = Container(system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None))
        i = Set(m, name="i", records=["seattle", "san-diego"])
        j = Set(m, name="j", records=["new-york", "chicago", "topeka"])

        a = Parameter(m, name="a", domain=[i], records=self.capacities)
        b = Parameter(m, name="b", domain=[j], records=self.demands)
        d = Parameter(m, name="d", domain=[i, j], records=self.distances)
        c = Parameter(m, name="c", domain=[i, j])
        bmult = Parameter(m, name="bmult", records=1)
        c[i, j] = 90 * d[i, j] / 1000

        x = Variable(m, name="x", domain=[i, j], type="Positive")
        z = Variable(m, name="z")

        cost = Equation(m, name="cost")
        supply = Equation(m, name="supply", domain=[i])
        demand = Equation(m, name="demand", domain=[j])

        cost[...] = z == Sum((i, j), c[i, j] * x[i, j])
        supply[i] = Sum(j, x[i, j]) <= a[i]
        demand[j] = Sum(i, x[i, j]) >= bmult * b[j]

        transport = Model(
            m,
            name="transport",
            equations=m.getEquations(),
            problem="LP",
            sense=Sense.MIN,
            objective=z,
        )

        transport.freeze(modifiables=[x.up])
        transport.solve(solver="conopt")
        self.assertAlmostEqual(transport.objective_value, 153.675, places=3)

        x.records.loc[1, "upper"] = 0
        transport.solve(solver="conopt")
        self.assertAlmostEqual(transport.objective_value, 156.375, places=3)

        transport.unfreeze()

    def test_fx(self):
        m = Container(system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None))
        INCOME0 = Parameter(
            m, name="INCOME0", description="notional income level", records=3.5
        )

        IADJ = Variable(
            m,
            name="IADJ",
            description=(
                "investment scaling factor (for fixed capital formation)"
            ),
            type="Free",
        )
        MPSADJ = Variable(
            m,
            name="MPSADJ",
            description="savings rate scaling factor",
            type="Free",
        )

        BALANCE = Equation(
            m,
            name="BALANCE",
            description="notional balance constraint",
            definition=(1 + IADJ) + (1 + MPSADJ) == INCOME0,
        )

        mm = Model(m, name="mm", equations=[BALANCE], problem="MCP")
        mm.freeze(modifiables=[INCOME0, IADJ.fx, MPSADJ.fx])
        IADJ.setRecords({"lower": 0, "upper": 0, "scale": 1})
        mm.solve()

        self.assertEqual(MPSADJ.records["level"].tolist()[0], 1.5)

        MPSADJ.setRecords({"lower": 0, "upper": 0, "scale": 1})
        mm.solve()

        self.assertEqual(MPSADJ.records["level"].tolist()[0], 0)
        mm.unfreeze()

    def test_validations(self):
        m = Container(system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None))
        i = Set(m, name="i", records=["seattle", "san-diego"])
        j = Set(m, name="j", records=["new-york", "chicago", "topeka"])

        a = Parameter(m, name="a", domain=[i], records=self.capacities)
        b = Parameter(m, name="b", domain=[j], records=self.demands)
        d = Parameter(m, name="d", domain=[i, j], records=self.distances)
        c = Parameter(m, name="c", domain=[i, j])
        bmult = Parameter(m, name="bmult", records=1)
        c[i, j] = 90 * d[i, j] / 1000

        x = Variable(m, name="x", domain=[i, j], type="Positive")
        z = Variable(m, name="z")

        cost = Equation(m, name="cost")
        supply = Equation(m, name="supply", domain=[i])
        demand = Equation(m, name="demand", domain=[j])

        cost[...] = z == Sum((i, j), c[i, j] * x[i, j])
        supply[i] = Sum(j, x[i, j]) <= a[i]
        demand[j] = Sum(i, x[i, j]) >= bmult * b[j]

        transport = Model(
            m,
            name="transport",
            equations=m.getEquations(),
            problem="LP",
            sense=Sense.MIN,
            objective=z,
        )

        # modifiables is not an iterable.
        with self.assertRaises(ValidationError):
            transport.freeze(modifiables=bmult)

        # provide a set as a modifiable
        with self.assertRaises(ValidationError):
            transport.freeze(modifiables=[i])

        transport.freeze(modifiables=[x.up], options=Options(lp="conopt"))

        # Test model instance options
        transport.solve(
            solver="conopt",
            model_instance_options=ModelInstanceOptions(debug=True),
        )
        self.assertAlmostEqual(transport.objective_value, 153.675, places=3)
        self.assertTrue(os.path.exists("dict.txt"))
        self.assertTrue(os.path.exists("gams.gms"))

        # Test solver options
        transport.solve(solver="conopt", solver_options={"rtmaxv": "1.e12"})
        self.assertTrue(
            os.path.exists(os.path.join(m.working_directory, "conopt.123"))
        )


def model_instance_suite():
    suite = unittest.TestSuite()
    tests = [
        ModelInstanceSuite(name)
        for name in dir(ModelInstanceSuite)
        if name.startswith("test_")
    ]
    suite.addTests(tests)

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(model_instance_suite())
