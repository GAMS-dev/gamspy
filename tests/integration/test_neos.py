from __future__ import annotations

import os
import sys
import unittest

import pandas as pd
from gamspy import (
    Container,
    Equation,
    Model,
    Parameter,
    Sense,
    Set,
    Sum,
    Variable,
)
from gamspy._backend.neos import NeosClient
from gamspy.exceptions import ValidationError

try:
    from dotenv import load_dotenv

    load_dotenv(os.getcwd() + os.sep + ".env")
except Exception:
    pass


class NeosSuite(unittest.TestCase):
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

    def test_neos_blocking(self):
        i = Set(self.m, name="i", records=["seattle", "san-diego"])
        j = Set(self.m, name="j", records=["new-york", "chicago", "topeka"])

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
        client = NeosClient(
            email=os.environ["NEOS_EMAIL"],
        )
        summary = transport.solve(backend="neos", client=client)
        self.assertTrue(isinstance(summary, pd.DataFrame))

        import math

        self.assertTrue(
            math.isclose(transport.objective_value, 153.675000, rel_tol=0.001)
        )

    def test_no_client(self):
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
        with self.assertRaises(ValidationError):
            transport.solve(backend="neos")

    def test_different_solver(self):
        i = Set(self.m, name="i", records=["seattle", "san-diego"])
        j = Set(self.m, name="j", records=["new-york", "chicago", "topeka"])

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
        client = NeosClient(
            email=os.environ["NEOS_EMAIL"],
        )
        transport.solve(backend="neos", client=client, solver="cplex")

        import math

        self.assertTrue(
            math.isclose(transport.objective_value, 153.675000, rel_tol=0.001)
        )

    def test_neos_non_blocking(self):
        i = Set(self.m, name="i", records=["seattle", "san-diego"])
        j = Set(self.m, name="j", records=["new-york", "chicago", "topeka"])

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
        client = NeosClient(
            email=os.environ["NEOS_EMAIL"],
            is_blocking=False,
        )
        transport.solve(backend="neos", client=client)

        job_number, job_password = client.jobs[-1]
        client.get_final_results(job_number, job_password)
        client.download_output(
            job_number,
            job_password,
            working_directory=f"tmp{os.sep}my_out_directory",
        )

        container = Container(
            load_from=f"tmp{os.sep}my_out_directory/output.gdx",
        )
        self.assertTrue("x" in container.data)
        x.setRecords(container["x"].records)
        self.assertTrue(x.records.equals(container["x"].records))

    def test_solver_options(self):
        # Set
        i = Set(
            self.m,
            name="i",
            records=["seattle", "san-diego"],
            description="canning plants",
        )
        j = Set(
            self.m,
            name="j",
            records=["new-york", "chicago", "topeka"],
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
        client = NeosClient(
            email=os.environ["NEOS_EMAIL"],
        )
        transport.solve(
            output=sys.stdout,
            solver="cplex",
            solver_options={"aggfill": "11"},
            backend="neos",
            client=client,
        )

        with open(os.path.join(self.m.working_directory, "solve.log")) as file:
            self.assertTrue(">>  aggfill 11" in file.read())

    def test_mip(self):
        mdl = Container()
        x1 = Variable(mdl, type="integer")
        x2 = Variable(mdl, type="integer")
        x3 = Variable(mdl, type="integer")

        eq1 = Equation(mdl)
        eq2 = Equation(mdl)
        eq3 = Equation(mdl)
        eq1[...] = x1 + 2 * x2 >= 3
        eq2[...] = x3 + x2 >= 5
        eq3[...] = x1 + x3 == 4
        obj = x1 + 3 * x2 + 3 * x3

        LP1 = Model(
            mdl,
            equations=mdl.getEquations(),
            problem="mip",
            sense="min",
            objective=obj,
        )

        client = NeosClient(email=os.environ["NEOS_EMAIL"])
        summary = LP1.solve(
            solver="cplex", output=sys.stdout, backend="neos", client=client
        )
        self.assertTrue(isinstance(summary, pd.DataFrame))


def neos_suite():
    suite = unittest.TestSuite()
    tests = [
        NeosSuite(name) for name in dir(NeosSuite) if name.startswith("test_")
    ]
    suite.addTests(tests)

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(neos_suite())
