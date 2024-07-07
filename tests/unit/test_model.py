from __future__ import annotations

import os
import unittest

from gamspy import (
    Container,
    Equation,
    Model,
    Parameter,
    Problem,
    Sense,
    Set,
    Sum,
    Variable,
)
from gamspy.exceptions import ValidationError


class ModelSuite(unittest.TestCase):
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

    def test_model(self):
        # No equations or matches
        self.assertRaises(ValidationError, Model, self.m)

        # Empty name
        self.assertRaises(ValueError, Model, self.m, "")

        i = Set(
            self.m,
            name="i",
            records=self.canning_plants,
            description="Canning Plants",
        )
        j = Set(self.m, name="j", records=self.markets, description="Markets")

        # Params
        a = Parameter(self.m, name="a", domain=[i], records=self.capacities)
        b = Parameter(self.m, name="b", domain=[j], records=self.demands)
        d = Parameter(self.m, name="d", domain=[i, j], records=self.distances)
        c = Parameter(self.m, name="c", domain=[i, j])
        c[i, j] = 90 * d[i, j] / 1000

        x = Variable(self.m, name="x", domain=[i, j], type="Positive")
        z = Variable(self.m, name="z")

        # Equation definition without an index
        cost = Equation(
            self.m,
            name="cost",
            description="define objective function",
        )
        cost[...] = Sum((i, j), c[i, j] * x[i, j]) == z

        # Equation definition with an index
        supply = Equation(
            self.m,
            name="supply",
            domain=[i],
            description="observe supply limit at plant i",
        )
        supply[i] = Sum(j, x[i, j]) <= a[i]

        demand = Equation(self.m, name="demand", domain=[j])
        demand[j] = Sum(i, x[i, j]) >= b[j]

        # Model with implicit objective
        test_model = Model(
            self.m,
            name="test_model",
            equations=[supply, demand],
            problem="LP",
            sense="min",
            objective=Sum((i, j), c[i, j] * x[i, j]),
        )
        test_model.solve(solver="CPLEX")
        self.assertEqual(
            list(self.m.data.keys()),
            [
                "i",
                "j",
                "a",
                "b",
                "d",
                "c",
                "x",
                "z",
                "cost",
                "supply",
                "demand",
                "test_model_objective_variable",
                "test_model_objective",
            ],
        )
        self.assertEqual(test_model.objective_value, 153.675)

        # Check if the name is reserved
        self.assertRaises(ValidationError, Model, self.m, "set", "LP")

        # Equation definition with more than one index
        bla = Equation(
            self.m,
            name="bla",
            domain=[i, j],
            description="observe supply limit at plant i",
        )
        bla[i, j] = x[i, j] <= a[i]

        # Test model with specific equations
        test_model2 = Model(
            self.m,
            name="test_model2",
            equations=[cost, supply],
            problem="LP",
            sense="min",
            objective=z,
        )
        self.assertEqual(
            test_model2.getDeclaration(),
            "Model test_model2 / cost,supply /;",
        )
        self.assertEqual(test_model2.equations, [cost, supply])

        test_model3 = Model(
            self.m,
            name="test_model3",
            equations=[cost],
            problem="LP",
            sense="min",
            objective=z,
        )
        test_model3.equations = [cost, supply]
        self.assertEqual(test_model3.equations, [cost, supply])

        test_model4 = self.m.addModel(
            name="test_model4",
            equations=[cost, supply],
            problem="LP",
            sense="min",
            objective=z,
        )

        self.assertTrue(test_model4.equations == test_model3.equations)

        test_model5 = self.m.addModel(
            name="test_model5",
            equations=[cost, supply],
            problem="LP",
            sense="min",
            objective=z,
            matches={supply: x, cost: z},
        )
        self.assertEqual(
            test_model5.getDeclaration(),
            "Model test_model5 / supply.x,cost.z /;",
        )

        # Equations provided as strings
        self.assertRaises(
            TypeError, Model, self.m, "test_model5", "LP", ["cost", "supply"]
        )

        # Test matches
        test_model6 = Model(
            self.m,
            name="test_model6",
            equations=[supply],
            matches={demand: x},
            problem="LP",
            sense="min",
        )
        self.assertEqual(
            test_model6.getDeclaration(),
            "Model test_model6 / supply,demand.x /;",
        )

        # Test repr and str
        self.assertTrue(
            test_model6.__repr__().startswith(f"<Model `{test_model6.name}`")
        )
        self.assertTrue(
            str(test_model6).startswith(
                f"Model {test_model6.name}:\n  Problem Type: LP\n  Sense: MIN\n  Equations:"
            )
        )

        # empty model name
        self.assertRaises(
            ValueError,
            Model,
            self.m,
            "test_model7",
            "",
            self.m.getEquations(),
            "min",
            Sum((i, j), c[i, j] * x[i, j]),
        )

        # model name too long
        self.assertRaises(
            ValueError,
            Model,
            self.m,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "LP",
            self.m.getEquations(),
            "min",
            Sum((i, j), c[i, j] * x[i, j]),
        )

        # model name is not an str
        self.assertRaises(
            TypeError,
            Model,
            self.m,
            5,
            "LP",
            self.m.getEquations(),
            "min",
            Sum((i, j), c[i, j] * x[i, j]),
        )

        # model name contains empty space
        self.assertRaises(
            ValidationError,
            Model,
            self.m,
            "test_model 8",
            "LP",
            self.m.getEquations(),
            "min",
            Sum((i, j), c[i, j] * x[i, j]),
        )

        # model name begins with underscore
        self.assertRaises(
            ValidationError,
            Model,
            self.m,
            "_test_model7",
            "LP",
            self.m.getEquations(),
            "min",
            Sum((i, j), c[i, j] * x[i, j]),
        )

    def test_feasibility(self):
        m = Container(
            system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None),
        )

        i = Set(m, name="i", records=["seattle", "san-diego"])
        j = Set(m, name="j", records=["new-york", "chicago", "topeka"])

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
            sense="feasibility",
        )
        transport.solve()
        self.assertIsNotNone(x.records)

        self.assertRaises(
            ValidationError,
            Model,
            m,
            "transport2",
            "LP",
            m.getEquations(),
            "feasibility",
            Sum((i, j), c[i, j] * x[i, j]),
        )

        self.assertRaises(
            ValidationError,
            Model,
            m,
            "transport2",
            "CNS",
            m.getEquations(),
            "feasibility",
        )

    def test_tuple_equations(self):
        i = Set(
            self.m,
            name="i",
            records=self.canning_plants,
            description="Canning Plants",
        )
        j = Set(self.m, name="j", records=self.markets, description="Markets")

        # Params
        a = Parameter(self.m, name="a", domain=[i], records=self.capacities)
        b = Parameter(self.m, name="b", domain=[j], records=self.demands)
        d = Parameter(self.m, name="d", domain=[i, j], records=self.distances)
        c = Parameter(self.m, name="c", domain=[i, j])
        c[i, j] = 90 * d[i, j] / 1000

        x = Variable(self.m, name="x", domain=[i, j], type="Positive")
        z = Variable(self.m, name="z")

        # Equation definition without an index
        cost = Equation(
            self.m,
            name="cost",
            description="define objective function",
        )
        cost.expr = Sum((i, j), c[i, j] * x[i, j]) == z

        # Equation definition with an index
        supply = Equation(
            self.m,
            name="supply",
            domain=[i],
            description="observe supply limit at plant i",
        )
        supply[i] = Sum(j, x[i, j]) <= a[i]

        demand = Equation(self.m, name="demand", domain=[j])
        demand[j] = Sum(i, x[i, j]) >= b[j]

        # Model with implicit objective
        test_model = Model(
            self.m,
            name="test_model",
            equations=[supply, demand],
            problem="LP",
            sense="min",
            objective=Sum((i, j), c[i, j] * x[i, j]),
        )
        test_model.solve()

    def test_compute_infeasibilities(self):
        m = Container(
            system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None),
        )

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

        x = Variable(
            m,
            name="x",
            domain=[i, j],
            type="Positive",
            description="shipment quantities in cases",
        )

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

        b[j] = 1.5 * b[j]
        transport.solve()

        infeasibilities = transport.compute_infeasibilities()
        columns = [
            "i",
            "level",
            "marginal",
            "lower",
            "upper",
            "scale",
            "infeasibility",
        ]
        self.assertEqual(
            list(infeasibilities.keys()),
            ["supply", "demand", "transport_objective"],
        )
        self.assertEqual(list(infeasibilities["supply"].columns), columns)
        self.assertEqual(
            infeasibilities["supply"].values.tolist(),
            [["san-diego", 1000.0, 0.0, float("-inf"), 600.0, 1.0, 400.0]],
        )

        self.assertEqual(
            x.compute_infeasibilities().values.tolist(),
            [
                [
                    "seattle",
                    "new-york",
                    -100.0,
                    0.0,
                    0.0,
                    float("inf"),
                    1.0,
                    100.0,
                ]
            ],
        )

        self.assertEqual(
            supply.compute_infeasibilities().values.tolist(),
            [["san-diego", 1000.0, 0.0, float("-inf"), 600.0, 1.0, 400.0]],
        )

    def test_equations(self):
        e = Equation(self.m, "e")
        e.l[...] = -10
        e.lo[...] = 5
        model = Model(
            self.m,
            "my",
            problem=Problem.LP,
            equations=[e],
            sense=Sense.FEASIBILITY,
        )

        with self.assertRaises(ValidationError):
            model.solve()


def model_suite():
    suite = unittest.TestSuite()
    tests = [
        ModelSuite(name)
        for name in dir(ModelSuite)
        if name.startswith("test_")
    ]
    suite.addTests(tests)

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(model_suite())
