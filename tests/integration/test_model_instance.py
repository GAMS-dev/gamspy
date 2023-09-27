import unittest

from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Parameter
from gamspy import Sense
from gamspy import Set
from gamspy import Sum
from gamspy import Variable


class ModelInstanceSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container()

    def test_parameter_change(self):
        m = Container()

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
        bmult = Parameter(m, name="bmult", records=1)
        c[i, j] = 90 * d[i, j] / 1000

        # Variable
        x = Variable(m, name="x", domain=[i, j], type="Positive")
        z = Variable(m, name="z")

        # Equation
        cost = Equation(m, name="cost")
        supply = Equation(m, name="supply", domain=[i])
        demand = Equation(m, name="demand", domain=[j])

        cost.expr = z == Sum((i, j), c[i, j] * x[i, j])
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
            164.002,
            164.519,
            166.117,
        ]

        transport.freeze(modifiables=[bmult])

        for b_value, result in zip(bmult_list, results):
            bmult.setRecords(b_value)
            transport.solve(model_instance_options={"solver": "conopt"})
            self.assertAlmostEqual(z.records["level"][0], result, places=2)

    def test_variable_change(self):
        m = Container()

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
        bmult = Parameter(m, name="bmult", records=1)
        c[i, j] = 90 * d[i, j] / 1000

        # Variable
        x = Variable(m, name="x", domain=[i, j], type="Positive")
        z = Variable(m, name="z")

        # Equation
        cost = Equation(m, name="cost")
        supply = Equation(m, name="supply", domain=[i])
        demand = Equation(m, name="demand", domain=[j])

        cost.expr = z == Sum((i, j), c[i, j] * x[i, j])
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
        transport.solve(model_instance_options={"solver": "conopt"})
        self.assertAlmostEqual(z.records["level"][0], 153.675, places=3)

        x.records.loc[1, "upper"] = 0
        transport.solve(model_instance_options={"solver": "conopt"})
        self.assertAlmostEqual(z.records["level"][0], 156.375, places=3)


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
