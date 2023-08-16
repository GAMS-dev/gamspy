import unittest

import pandas as pd
from gamspy import Container, Set, Parameter, Variable, Model, Equation, Sum


class ModelSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container()

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
        cost.definition = Sum((i, j), c[i, j] * x[i, j]) == z

        # Equation definition with an index
        supply = Equation(
            self.m,
            name="supply",
            domain=[i],
            description="observe supply limit at plant i",
        )
        supply[i] = Sum(j, x[i, j]) <= a[i]

        # Equation definition with more than one index
        bla = Equation(
            self.m,
            name="bla",
            domain=[i, j],
            description="observe supply limit at plant i",
        )
        bla[i, j] = Sum((i, j), x[i, j]) <= a[i]

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
            test_model2.getStatement(), "Model test_model2 / cost,supply /;"
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
