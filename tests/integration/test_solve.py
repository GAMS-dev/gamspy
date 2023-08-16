import unittest

import os
import pandas as pd
from gamspy import (
    Container,
    Set,
    Parameter,
    Variable,
    Equation,
    Model,
    Sum,
    ModelStatus,
)


class SolveSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container()

    def test_read_on_demand(self):
        # Prepare data
        distances = pd.DataFrame(
            [
                ["seattle", "new-york", 2.5],
                ["seattle", "chicago", 1.7],
                ["seattle", "topeka", 1.8],
                ["san-diego", "new-york", 2.5],
                ["san-diego", "chicago", 1.8],
                ["san-diego", "topeka", 1.4],
            ]
        )
        capacities = [["seattle", 350], ["san-diego", 600]]
        demands = [["new-york", 325], ["chicago", 300], ["topeka", 275]]

        # Set
        i = Set(self.m, name="i", records=["seattle", "san-diego"])
        j = Set(self.m, name="j", records=["new-york", "chicago", "topeka"])
        k = Set(
            self.m, name="k", records=["seattle", "san-diego", "california"]
        )
        k["seattle"] = False
        self.assertTrue(k._is_dirty)
        self.assertEqual(
            k.records.loc[0, :].values.tolist(), ["san-diego", ""]
        )
        self.assertFalse(k._is_dirty)

        # Data
        a = Parameter(self.m, name="a", domain=[i], records=capacities)
        b = Parameter(self.m, name="b", domain=[j], records=demands)
        d = Parameter(self.m, name="d", domain=[i, j], records=distances)
        c = Parameter(self.m, name="c", domain=[i, j])
        e = Parameter(self.m, name="e")

        c[i, j] = 90 * d[i, j] / 1000
        self.assertTrue(c._is_dirty)
        self.assertEqual(
            c.records.values.tolist(),
            [
                ["seattle", "new-york", 0.225],
                ["seattle", "chicago", 0.153],
                ["seattle", "topeka", 0.162],
                ["san-diego", "new-york", 0.225],
                ["san-diego", "chicago", 0.162],
                ["san-diego", "topeka", 0.126],
            ],
        )
        self.assertFalse(c._is_dirty)

        e.assign = 5
        self.assertTrue(e._is_dirty)
        self.assertEqual(e.records.values.tolist(), [[5.0]])
        self.assertEqual(e.assign, 5)

        with self.assertRaises(TypeError):
            e.records = 5

        # Variable
        x = Variable(self.m, name="x", domain=[i, j], type="Positive")
        z = Variable(self.m, name="z")

        # Equation
        cost = Equation(self.m, name="cost")
        supply = Equation(self.m, name="supply", domain=[i])
        demand = Equation(self.m, name="demand", domain=[j])

        cost.definition = Sum((i, j), c[i, j] * x[i, j]) == z
        supply[i] = Sum(j, x[i, j]) <= a[i]
        demand[j] = Sum(i, x[i, j]) >= b[j]

        transport = Model(
            self.m,
            name="transport",
            equations=[cost, supply, demand],
            problem="LP",
            sense="min",
            objective=z,
        )
        transport.solve()

        # Test the columns of a set
        self.assertTrue(i.records.columns.tolist() == ["uni", "element_text"])

        # Test the columns of a parameter
        self.assertTrue(a.records.columns.tolist() == ["i", "value"])

        # Test the columns of scalar variable
        self.assertTrue(
            z.records.columns.tolist()
            == ["level", "marginal", "lower", "upper", "scale"]
        )

        # Test the columns of indexed variable
        self.assertTrue(
            x.records.columns.tolist()
            == ["i", "j", "level", "marginal", "lower", "upper", "scale"]
        )

        # Test the columns of equation
        self.assertTrue(
            cost.records.columns.tolist()
            == ["level", "marginal", "lower", "upper", "scale"]
        )

        # Test the columns of indexed equation
        self.assertTrue(
            supply.records.columns.tolist()
            == ["i", "level", "marginal", "lower", "upper", "scale"]
        )

    def test_after_first_solve(self):
        # Prepare data
        distances = pd.DataFrame(
            [
                ["seattle", "new-york", 2.5],
                ["seattle", "chicago", 1.7],
                ["seattle", "topeka", 1.8],
                ["san-diego", "new-york", 2.5],
                ["san-diego", "chicago", 1.8],
                ["san-diego", "topeka", 1.4],
            ]
        )
        capacities = pd.DataFrame([["seattle", 350], ["san-diego", 600]])
        demands = pd.DataFrame(
            [["new-york", 325], ["chicago", 300], ["topeka", 275]]
        )

        # Set
        i = Set(self.m, name="i", records=["seattle", "san-diego"])
        j = Set(self.m, name="j", records=["new-york", "chicago", "topeka"])

        # Data
        a = Parameter(self.m, name="a", domain=[i], records=capacities)
        b = Parameter(self.m, name="b", domain=[j], records=demands)
        d = Parameter(self.m, name="d", domain=[i, j], records=distances)
        c = Parameter(self.m, name="c", domain=[i, j])
        c[i, j] = 90 * d[i, j] / 1000

        # Variable
        x = Variable(self.m, name="x", domain=[i, j], type="Positive")
        z = Variable(self.m, name="z")
        z2 = Variable(self.m, name="z2")

        # Equation
        cost = Equation(self.m, name="cost")
        cost2 = Equation(self.m, name="cost2")
        supply = Equation(self.m, name="supply", domain=[i])
        demand = Equation(self.m, name="demand", domain=[j])

        cost.definition = Sum((i, j), c[i, j] * x[i, j]) == z
        supply[i] = Sum(j, x[i, j]) <= a[i]
        demand[j] = Sum(i, x[i, j]) >= b[j]
        cost2.definition = Sum((i, j), c[i, j] * x[i, j]) * 5 == z2

        transport = Model(
            self.m,
            name="transport",
            equations=[cost, supply, demand],
            problem="LP",
            sense="min",
            objective=z,
        )
        transport.solve()

        first_z2_value = z2.records["level"].values[0]
        self.assertEqual(first_z2_value, 0.0)

        transport2 = Model(
            self.m,
            name="transport2",
            equations=[cost2, supply, demand],
            problem="LP",
            sense="min",
            objective=z2,
        )
        transport2.solve()
        second_z2_value = z2.records["level"].values[0]
        self.assertAlmostEqual(second_z2_value, 768.375, 3)

    def test_solve(self):
        # Prepare data
        distances = pd.DataFrame(
            [
                ["seattle", "new-york", 2.5],
                ["seattle", "chicago", 1.7],
                ["seattle", "topeka", 1.8],
                ["san-diego", "new-york", 2.5],
                ["san-diego", "chicago", 1.8],
                ["san-diego", "topeka", 1.4],
            ]
        )
        capacities = pd.DataFrame([["seattle", 350], ["san-diego", 600]])
        demands = pd.DataFrame(
            [["new-york", 325], ["chicago", 300], ["topeka", 275]]
        )

        # Set
        i = Set(self.m, name="i", records=["seattle", "san-diego"])
        j = Set(self.m, name="j", records=["new-york", "chicago", "topeka"])

        # Data
        a = Parameter(self.m, name="a", domain=[i], records=capacities)
        b = Parameter(self.m, name="b", domain=[j], records=demands)
        d = Parameter(self.m, name="d", domain=[i, j], records=distances)
        c = Parameter(self.m, name="c", domain=[i, j])
        c[i, j] = 90 * d[i, j] / 1000

        # Variable
        x = Variable(self.m, name="x", domain=[i, j], type="Positive")
        z = Variable(self.m, name="z")

        # Equation
        cost = Equation(self.m, name="cost")
        supply = Equation(self.m, name="supply", domain=[i])
        demand = Equation(self.m, name="demand", domain=[j])

        cost.definition = Sum((i, j), c[i, j] * x[i, j]) == z
        supply[i] = Sum(j, x[i, j]) <= a[i]
        demand[j] = Sum(i, x[i, j]) >= b[j]

        transport = Model(
            self.m,
            name="transport",
            equations=[cost, supply, demand],
            problem="LP",
            sense="min",
            objective=z,
        )

        # Test output redirection
        with open("test.gms", "w") as file:
            _ = transport.solve(
                commandline_options={"resLim": 100},
                output=file,
            )

        self.assertTrue(os.path.exists("test.gms"))
        self.assertTrue(transport.status == ModelStatus.OptimalGlobal)
        for attr_name in transport._get_attribute_names().values():
            self.assertTrue(hasattr(transport, attr_name))

            # Make sure model attributes are not in the container
            self.assertFalse(attr_name in self.m.data.keys())

        # Make sure dummy variable and equation is not in the container
        self.assertFalse(any("dummy_" in name for name in self.m.data.keys()))

        # Test invalid problem
        self.assertRaises(ValueError, Model, self.m, "model", [cost], "bla")

        # Test invalid sense
        self.assertRaises(
            ValueError, Model, self.m, "model", [cost], "LP", "bla"
        )

        # Test invalid objective variable
        self.assertRaises(
            TypeError, Model, self.m, "model", [cost], "LP", "min", a
        )

        # Test invalid commandline options
        self.assertRaises(
            Exception,
            transport.solve,
            {"bla": 100},
        )

        self.assertRaises(Exception, transport.solve, 5)

        # Try to solve invalid model
        m = Container()
        cost = Equation(m, "cost")
        model = Model(m, "model", equations=[cost], problem="LP", sense="min")
        self.assertRaises(Exception, model.solve)

        # Test limited variables
        transport = Model(
            m,
            name="transport",
            equations=[cost, supply, demand],
            problem="LP",
            sense="min",
            objective=z,
            limited_variables=[x[i]],
        )

        self.assertEqual(
            transport.getStatement(),
            "Model transport / cost,supply,demand,x(i) /;",
        )


def solve_suite():
    suite = unittest.TestSuite()
    tests = [
        SolveSuite(name)
        for name in dir(SolveSuite)
        if name.startswith("test_")
    ]
    suite.addTests(tests)

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(solve_suite())
