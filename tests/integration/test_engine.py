from __future__ import annotations

import os
import tempfile
import time
import unittest

import pandas as pd
from gams import GamsEngineConfiguration
from gamspy import (
    Container,
    EngineClient,
    Equation,
    Model,
    Parameter,
    Sense,
    Set,
    Sum,
    Variable,
)
from gamspy.exceptions import ValidationError

try:
    from dotenv import load_dotenv

    load_dotenv(os.getcwd() + os.sep + ".env")
except Exception:
    pass


class EngineSuite(unittest.TestCase):
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

    def test_engine(self):
        m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
        )

        # Set
        i = Set(m, name="i", records=["seattle", "san-diego"])
        j = Set(m, name="j", records=["new-york", "chicago", "topeka"])

        # Data
        a = Parameter(m, name="a", domain=[i], records=self.capacities)
        b = Parameter(m, name="b", domain=[j], records=self.demands)
        d = Parameter(m, name="d", domain=[i, j], records=self.distances)
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
            sense=Sense.MIN,
            objective=Sum((i, j), c[i, j] * x[i, j]),
        )

        client = EngineClient(
            host=os.environ["ENGINE_URL"],
            username=os.environ["ENGINE_USER"],
            password=os.environ["ENGINE_PASSWORD"],
            namespace=os.environ["ENGINE_NAMESPACE"],
        )

        self.assertTrue(
            isinstance(client._get_engine_config(), GamsEngineConfiguration)
        )

        transport.solve(backend="engine", client=client)

        self.assertEqual(transport.objective_value, 153.675)

        # invalid configuration
        client = EngineClient(
            host="http://localhost",
            username="bla",
            password="bla",
            namespace="bla",
        )

        transport3 = Model(
            m,
            name="transport3",
            equations=m.getEquations(),
            problem="LP",
            sense=Sense.MIN,
            objective=Sum((i, j), c[i, j] * x[i, j]),
        )
        self.assertRaises(
            ValidationError,
            transport3.solve,
            None,
            None,
            None,
            None,
            None,
            "engine",
        )

    def test_no_config(self):
        m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
        )

        # Set
        i = Set(m, name="i", records=["seattle", "san-diego"])
        j = Set(m, name="j", records=["new-york", "chicago", "topeka"])

        # Data
        a = Parameter(m, name="a", domain=[i], records=self.capacities)
        b = Parameter(m, name="b", domain=[j], records=self.demands)
        d = Parameter(m, name="d", domain=[i, j], records=self.distances)
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
            sense=Sense.MIN,
            objective=Sum((i, j), c[i, j] * x[i, j]),
        )

        with self.assertRaises(ValidationError):
            transport.solve(backend="engine")

    def test_extra_files(self):
        m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
        )

        # Set
        i = Set(m, name="i", records=["seattle", "san-diego"])
        j = Set(m, name="j", records=["new-york", "chicago", "topeka"])

        # Data
        a = Parameter(m, name="a", domain=[i], records=self.capacities)
        b = Parameter(m, name="b", domain=[j], records=self.demands)
        d = Parameter(m, name="d", domain=[i, j], records=self.distances)
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
            sense=Sense.MIN,
            objective=Sum((i, j), c[i, j] * x[i, j]),
        )

        with open(
            m.working_directory + os.sep + "test.txt", "w"
        ) as same_directory_file:
            client = EngineClient(
                host=os.environ["ENGINE_URL"],
                username=os.environ["ENGINE_USER"],
                password=os.environ["ENGINE_PASSWORD"],
                namespace=os.environ["ENGINE_NAMESPACE"],
                extra_model_files=[same_directory_file.name],
            )

            transport.solve(backend="engine", client=client)

        file = tempfile.NamedTemporaryFile(delete=False)
        client = EngineClient(
            host=os.environ["ENGINE_URL"],
            username=os.environ["ENGINE_USER"],
            password=os.environ["ENGINE_PASSWORD"],
            namespace=os.environ["ENGINE_NAMESPACE"],
            extra_model_files=[file.name],
        )

        with self.assertRaises(ValidationError):
            transport.solve(backend="engine", client=client)

        file.close()
        os.unlink(file.name)

    def test_solve_twice(self):
        m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
        )

        # Set
        i = Set(m, name="i", records=["seattle", "san-diego"])
        j = Set(m, name="j", records=["new-york", "chicago", "topeka"])

        # Data
        a = Parameter(m, name="a", domain=[i], records=self.capacities)
        b = Parameter(m, name="b", domain=[j], records=self.demands)
        d = Parameter(m, name="d", domain=[i, j], records=self.distances)
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
            sense=Sense.MIN,
            objective=Sum((i, j), c[i, j] * x[i, j]),
        )

        client = EngineClient(
            host=os.environ["ENGINE_URL"],
            username=os.environ["ENGINE_USER"],
            password=os.environ["ENGINE_PASSWORD"],
            namespace=os.environ["ENGINE_NAMESPACE"],
        )

        transport.solve(backend="engine", client=client)
        transport.solve(backend="engine", client=client)

    def test_summary(self):
        m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
        )

        # Set
        i = Set(m, name="i", records=["seattle", "san-diego"])
        j = Set(m, name="j", records=["new-york", "chicago", "topeka"])

        # Data
        a = Parameter(m, name="a", domain=[i], records=self.capacities)
        b = Parameter(m, name="b", domain=[j], records=self.demands)
        d = Parameter(m, name="d", domain=[i, j], records=self.distances)
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
            sense=Sense.MIN,
            objective=Sum((i, j), c[i, j] * x[i, j]),
        )
        summary = transport.solve()
        self.assertTrue(isinstance(summary, pd.DataFrame))

    def test_non_blocking(self):
        m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
            working_directory=".",
        )

        # Set
        i = Set(m, name="i", records=["seattle", "san-diego"])
        j = Set(m, name="j", records=["new-york", "chicago", "topeka"])

        # Data
        a = Parameter(m, name="a", domain=[i], records=self.capacities)
        b = Parameter(m, name="b", domain=[j], records=self.demands)
        d = Parameter(m, name="d", domain=[i, j], records=self.distances)
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
            sense=Sense.MIN,
            objective=Sum((i, j), c[i, j] * x[i, j]),
        )
        client = EngineClient(
            host=os.environ["ENGINE_URL"],
            username=os.environ["ENGINE_USER"],
            password=os.environ["ENGINE_PASSWORD"],
            namespace=os.environ["ENGINE_NAMESPACE"],
            is_blocking=False,
        )
        transport.solve(backend="engine", client=client)

        token = client.tokens[-1]

        job_status, _, exit_code = client.job.get(token)
        while job_status != 10:
            job_status, _, exit_code = client.job.get(token)

        self.assertEqual(exit_code, 0)

        client.job.get_results(token, "out_dir")

        gdx_out_path = os.path.join(
            "out_dir", os.path.basename(m.gdxOutputPath())
        )
        container = Container(load_from=gdx_out_path)
        self.assertTrue("x" in container.data)
        x.setRecords(container["x"].records)
        self.assertTrue(x.records.equals(container["x"].records))

    def test_api_job(self):
        client = EngineClient(
            host=os.environ["ENGINE_URL"],
            username=os.environ["ENGINE_USER"],
            password=os.environ["ENGINE_PASSWORD"],
            namespace=os.environ["ENGINE_NAMESPACE"],
            is_blocking=False,
        )

        gms_path = os.path.join(os.getcwd(), "dummy.gms")
        with open(gms_path, "w") as file:
            file.write("Set i / i1*i3 /;")

        token = client.job.post(os.getcwd(), gms_path)

        status, _, _ = client.job.get(token)
        while status != 10:
            status, _, _ = client.job.get(token)
            print(client.job.get_logs(token))

    def test_api_auth(self):
        # /api/auth -> post
        client = EngineClient(
            host=os.environ["ENGINE_URL"],
            username=os.environ["ENGINE_USER"],
            password=os.environ["ENGINE_PASSWORD"],
            namespace=os.environ["ENGINE_NAMESPACE"],
        )

        token = client.auth.post(scope=["JOBS", "AUTH"])
        self.assertTrue(token is not None and isinstance(token, str))

        # First get a JWT token, then send a job
        client = EngineClient(
            host=os.environ["ENGINE_URL"],
            username=os.environ["ENGINE_USER"],
            password=os.environ["ENGINE_PASSWORD"],
            namespace=os.environ["ENGINE_NAMESPACE"],
        )

        # /api/auth/login -> post
        jwt_token = client.auth.login(scope=["JOBS", "AUTH"])
        time.sleep(1)

        self.assertTrue(jwt_token is not None and isinstance(jwt_token, str))

        client = EngineClient(
            host=os.environ["ENGINE_URL"],
            namespace=os.environ["ENGINE_NAMESPACE"],
            jwt=jwt_token,
        )
        gms_path = os.path.join(os.getcwd(), "dummy2.gms")
        with open(gms_path, "w") as file:
            file.write("Set i / i1*i3 /;")

        token = client.job.post(os.getcwd(), gms_path)

        status, _, _ = client.job.get(token)
        while status != 10:
            status, _, _ = client.job.get(token)

        # /api/auth/logout -> post
        # message = client.auth.logout()
        # self.assertTrue(message is not None and isinstance(message, str))


def engine_suite():
    suite = unittest.TestSuite()
    tests = [
        EngineSuite(name)
        for name in dir(EngineSuite)
        if name.startswith("test_")
    ]
    suite.addTests(tests)

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(engine_suite())
