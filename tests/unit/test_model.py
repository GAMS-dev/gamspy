from __future__ import annotations

import os
import unittest

import pandas as pd

from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Parameter
from gamspy import Set
from gamspy import Sum
from gamspy import Variable
from gamspy.exceptions import ValidationError


class ModelSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container(
            delayed_execution=int(os.getenv("DELAYED_EXECUTION", False))
        )

    def test_model(self):
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
        canning_plants = ["seattle", "san-diego"]
        markets = ["new-york", "chicago", "topeka"]
        capacities = pd.DataFrame([["seattle", 350], ["san-diego", 600]])
        demands = [["new-york", 325], ["chicago", 300], ["topeka", 275]]

        # Sets
        i = Set(
            self.m,
            name="i",
            records=canning_plants,
            description="Canning Plants",
        )
        j = Set(self.m, name="j", records=markets, description="Markets")

        # Params
        a = Parameter(self.m, name="a", domain=[i], records=capacities)
        b = Parameter(self.m, name="b", domain=[j], records=demands)
        d = Parameter(self.m, name="d", domain=[i, j], records=distances)
        c = Parameter(self.m, name="c", domain=[i, j])
        c[i, j] = 90 * d[i, j] / 1000

        # Variables
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
        test_model.solve()
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
            test_model2.getStatement(),
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
            test_model5.getStatement(),
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
            test_model6.getStatement(),
            "Model test_model6 / supply,demand.x /;",
        )

    def test_feasibility(self):
        m = Container(
            delayed_execution=int(os.getenv("DELAYED_EXECUTION", False))
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
        canning_plants = ["seattle", "san-diego"]
        markets = ["new-york", "chicago", "topeka"]
        capacities = pd.DataFrame([["seattle", 350], ["san-diego", 600]])
        demands = [["new-york", 325], ["chicago", 300], ["topeka", 275]]

        # Sets
        i = Set(
            self.m,
            name="i",
            records=canning_plants,
            description="Canning Plants",
        )
        j = Set(self.m, name="j", records=markets, description="Markets")

        # Params
        a = Parameter(self.m, name="a", domain=[i], records=capacities)
        b = Parameter(self.m, name="b", domain=[j], records=demands)
        d = Parameter(self.m, name="d", domain=[i, j], records=distances)
        c = Parameter(self.m, name="c", domain=[i, j])
        c[i, j] = 90 * d[i, j] / 1000

        # Variables
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
            equations=(supply, demand),
            problem="LP",
            sense="min",
            objective=Sum((i, j), c[i, j] * x[i, j]),
        )
        test_model.solve()


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
