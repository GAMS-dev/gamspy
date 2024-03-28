from __future__ import annotations

import os
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
        self.m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None)
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

    def test_neos_blocking(self):
        m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
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
        m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
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
        with self.assertRaises(ValidationError):
            transport.solve(backend="neos")

    def test_different_solver(self):
        m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
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
            sense=Sense.MIN,
            objective=Sum((i, j), c[i, j] * x[i, j]),
        )
        client = NeosClient(
            email=os.environ["NEOS_EMAIL"],
        )
        transport.solve(backend="neos", client=client, solver="CONOPT")

        import math

        self.assertTrue(
            math.isclose(transport.objective_value, 153.675000, rel_tol=0.001)
        )

    def test_neos_non_blocking(self):
        m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
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
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
            load_from=f"tmp{os.sep}my_out_directory/output.gdx",
        )
        self.assertTrue("x" in container.data)
        x.setRecords(container["x"].records)
        self.assertTrue(x.records.equals(container["x"].records))


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
