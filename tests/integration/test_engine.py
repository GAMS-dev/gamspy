from __future__ import annotations

import glob
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import pytest

from gamspy import (
    Container,
    EngineClient,
    Equation,
    Model,
    Options,
    Parameter,
    Sense,
    Set,
    Sum,
    Variable,
)
from gamspy._backend.engine import EngineConfiguration
from gamspy.exceptions import ValidationError

pytestmark = pytest.mark.engine
try:
    from dotenv import load_dotenv

    load_dotenv(os.getcwd() + os.sep + ".env")
except Exception:
    pass


@pytest.fixture
def data():
    os.makedirs("tmp", exist_ok=True)
    m = Container()
    canning_plants = ["seattle", "san-diego"]
    markets = ["new-york", "chicago", "topeka"]
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

    yield m, canning_plants, markets, capacities, demands, distances
    m.close()
    shutil.rmtree("tmp")
    files = glob.glob("_*")
    for file in files:
        if os.path.isfile(file):
            os.remove(file)


@pytest.fixture
def network_license():
    subprocess.run(
        [
            sys.executable,
            "-Bm",
            "gamspy",
            "install",
            "license",
            os.environ["NETWORK_LICENSE_NON_ACADEMIC"],
        ],
        check=True,
    )

    m = Container()
    canning_plants = ["seattle", "san-diego"]
    markets = ["new-york", "chicago", "topeka"]
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

    yield m, canning_plants, markets, capacities, demands, distances
    m.close()
    subprocess.run(
        [
            sys.executable,
            "-Bm",
            "gamspy",
            "install",
            "license",
            os.environ["LOCAL_LICENSE"],
        ],
        check=True,
    )


def test_network_license(network_license):
    m, canning_plants, markets, capacities, demands, distances = (
        network_license
    )
    i = Set(m, name="i", records=canning_plants)
    j = Set(m, name="j", records=markets)

    a = Parameter(m, name="a", domain=[i], records=capacities)
    b = Parameter(m, name="b", domain=[j], records=demands)
    d = Parameter(m, name="d", domain=[i, j], records=distances)
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
    client = EngineClient(
        host=os.environ["ENGINE_URL"],
        username=os.environ["ENGINE_USER"],
        password=os.environ["ENGINE_PASSWORD"],
        namespace=os.environ["ENGINE_NAMESPACE"],
    )

    with pytest.raises(ValidationError):
        transport.solve(backend="engine", client=client, solver="mpsge")

    summary = transport.solve(backend="engine", client=client)
    assert isinstance(summary, pd.DataFrame)

    import math

    assert math.isclose(transport.objective_value, 153.675000, rel_tol=0.001)


def test_engine(data):
    m, canning_plants, markets, capacities, demands, distances = data
    m = Container(debugging_level="keep")

    i = Set(m, name="i", records=canning_plants)
    j = Set(m, name="j", records=markets)

    a = Parameter(m, name="a", domain=[i], records=capacities)
    b = Parameter(m, name="b", domain=[j], records=demands)
    d = Parameter(m, name="d", domain=[i, j], records=distances)
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

    with pytest.raises(ValidationError):
        _ = EngineClient(
            host="blabla_dummy_host",
            username=os.environ["ENGINE_USER"],
            password=os.environ["ENGINE_PASSWORD"],
            namespace="stupid_namespace",
        )

    client = EngineClient(
        host=os.environ["ENGINE_URL"],
        username=os.environ["ENGINE_USER"],
        password=os.environ["ENGINE_PASSWORD"],
        namespace=os.environ["ENGINE_NAMESPACE"],
        remove_results=True,
    )

    assert isinstance(client._engine_config, EngineConfiguration)

    transport.solve(backend="engine", client=client, output=sys.stdout)

    assert transport.objective_value == 153.675

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
    pytest.raises(
        ValidationError,
        transport3.solve,
        None,
        None,
        None,
        None,
        None,
        "engine",
    )


def test_logoption(data):
    m, canning_plants, markets, capacities, demands, distances = data
    i = Set(m, name="i", records=canning_plants)
    j = Set(m, name="j", records=markets)

    a = Parameter(m, name="a", domain=[i], records=capacities)
    b = Parameter(m, name="b", domain=[j], records=demands)
    d = Parameter(m, name="d", domain=[i, j], records=distances)
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

    client = EngineClient(
        host=os.environ["ENGINE_URL"],
        username=os.environ["ENGINE_USER"],
        password=os.environ["ENGINE_PASSWORD"],
        namespace=os.environ["ENGINE_NAMESPACE"],
    )

    # logoption=2
    log_file_path = os.path.join("tmp", "bla.log")
    transport.solve(
        backend="engine",
        client=client,
        options=Options(log_file=log_file_path),
    )
    assert os.path.exists(log_file_path)

    # logoption=3
    transport.solve(backend="engine", client=client, output=sys.stdout)
    assert not os.path.exists(
        os.path.join(m.working_directory, "log_stdout.txt")
    )

    # logoption=4
    log_file_path = os.path.join("tmp", "bla2.log")
    transport.solve(
        backend="engine",
        client=client,
        output=sys.stdout,
        options=Options(log_file=log_file_path),
    )
    assert os.path.exists(log_file_path)


def test_no_config(data):
    m, canning_plants, markets, capacities, demands, distances = data
    i = Set(m, name="i", records=canning_plants)
    j = Set(m, name="j", records=markets)

    a = Parameter(m, name="a", domain=[i], records=capacities)
    b = Parameter(m, name="b", domain=[j], records=demands)
    d = Parameter(m, name="d", domain=[i, j], records=distances)
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

    with pytest.raises(ValidationError):
        transport.solve(backend="engine")


def test_extra_files(data):
    m, canning_plants, markets, capacities, demands, distances = data
    i = Set(m, name="i", records=canning_plants)
    j = Set(m, name="j", records=markets)

    a = Parameter(m, name="a", domain=[i], records=capacities)
    b = Parameter(m, name="b", domain=[j], records=demands)
    d = Parameter(m, name="d", domain=[i, j], records=distances)
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

    with tempfile.NamedTemporaryFile() as file:
        client = EngineClient(
            host=os.environ["ENGINE_URL"],
            username=os.environ["ENGINE_USER"],
            password=os.environ["ENGINE_PASSWORD"],
            namespace=os.environ["ENGINE_NAMESPACE"],
            extra_model_files=[file.name],
        )

        with pytest.raises(ValidationError):
            transport.solve(backend="engine", client=client)


def test_solve_twice(data):
    m, canning_plants, markets, capacities, demands, distances = data
    i = Set(m, name="i", records=canning_plants)
    j = Set(m, name="j", records=markets)

    a = Parameter(m, name="a", domain=[i], records=capacities)
    b = Parameter(m, name="b", domain=[j], records=demands)
    d = Parameter(m, name="d", domain=[i, j], records=distances)
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

    client = EngineClient(
        host=os.environ["ENGINE_URL"],
        username=os.environ["ENGINE_USER"],
        password=os.environ["ENGINE_PASSWORD"],
        namespace=os.environ["ENGINE_NAMESPACE"],
    )

    transport.solve(backend="engine", client=client)
    transport.solve(backend="engine", client=client)


def test_summary(data):
    m, canning_plants, markets, capacities, demands, distances = data
    i = Set(m, name="i", records=canning_plants)
    j = Set(m, name="j", records=markets)

    a = Parameter(m, name="a", domain=[i], records=capacities)
    b = Parameter(m, name="b", domain=[j], records=demands)
    d = Parameter(m, name="d", domain=[i, j], records=distances)
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
    summary = transport.solve()
    assert isinstance(summary, pd.DataFrame)


def test_non_blocking(data):
    m, canning_plants, markets, capacities, demands, distances = data
    i = Set(m, name="i", records=canning_plants)
    j = Set(m, name="j", records=markets)

    a = Parameter(m, name="a", domain=[i], records=capacities)
    b = Parameter(m, name="b", domain=[j], records=demands)
    d = Parameter(m, name="d", domain=[i, j], records=distances)
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
    client = EngineClient(
        host=os.environ["ENGINE_URL"],
        username=os.environ["ENGINE_USER"],
        password=os.environ["ENGINE_PASSWORD"],
        namespace=os.environ["ENGINE_NAMESPACE"],
        is_blocking=False,
    )
    # send jobs asynchronously
    for _ in range(3):
        transport.solve(backend="engine", client=client)

    # gather the results
    for i in range(3):
        token = client.tokens[i]

        job_status, _, exit_code = client.job.get(token)
        while job_status != 10:
            job_status, _, exit_code = client.job.get(token)

        assert exit_code == 0

        client.job.get_results(token, f"tmp{os.sep}out_dir{i}")

    gdx_out_path = os.path.join(
        f"tmp{os.sep}out_dir0", os.path.basename(m.gdxOutputPath())
    )
    container = Container(load_from=gdx_out_path)
    assert "x" in container.data
    x.setRecords(container["x"].records)
    assert x.records.equals(container["x"].records)


def test_api_job(data):
    client = EngineClient(
        host=os.environ["ENGINE_URL"],
        username=os.environ["ENGINE_USER"],
        password=os.environ["ENGINE_PASSWORD"],
        namespace=os.environ["ENGINE_NAMESPACE"],
        is_blocking=False,
    )

    gms_path = os.path.join(os.getcwd(), "tmp", "dummy.gms")
    with open(gms_path, "w") as file:
        file.write("Set i / i1*i3 /;")

    token = client.job.post(os.getcwd() + os.sep + "tmp", gms_path)

    status, _, _ = client.job.get(token)
    while status != 10:
        status, _, _ = client.job.get(token)
        print(client.job.get_logs(token))

    client.job.delete_results(token)


def test_api_auth(data):
    # /api/auth -> post
    client = EngineClient(
        host=os.environ["ENGINE_URL"],
        username=os.environ["ENGINE_USER"],
        password=os.environ["ENGINE_PASSWORD"],
        namespace=os.environ["ENGINE_NAMESPACE"],
    )

    with pytest.raises(ValidationError):
        _ = client.auth.post(scope=["Blabla"])

    token = client.auth.post(scope=["JOBS", "AUTH"])
    assert token is not None and isinstance(token, str)

    # First get a JWT token, then send a job
    client = EngineClient(
        host=os.environ["ENGINE_URL"],
        username=os.environ["ENGINE_USER"],
        password=os.environ["ENGINE_PASSWORD"],
        namespace=os.environ["ENGINE_NAMESPACE"],
    )

    # /api/auth/login -> post
    with pytest.raises(ValidationError):
        _ = client.auth.login(scope=["Blabla"])
    jwt_token = client.auth.login(scope=["JOBS", "AUTH"])
    time.sleep(1)

    assert jwt_token is not None and isinstance(jwt_token, str)

    client = EngineClient(
        host=os.environ["ENGINE_URL"],
        namespace=os.environ["ENGINE_NAMESPACE"],
        jwt=jwt_token,
    )
    gms_path = os.path.join(os.getcwd(), "tmp", "dummy2.gms")
    with open(gms_path, "w") as file:
        file.write("Set i / i1*i3 /;")

    token = client.job.post(os.getcwd() + os.sep + "tmp", gms_path)

    status, _, _ = client.job.get(token)
    while status != 10:
        status, _, _ = client.job.get(token)

    # /api/auth/logout -> post
    # logout only on Python 3.12 to avoid unauthorized calls on parallel jobs.
    if platform.system() == "Linux" and sys.version_info.minor == 12:
        message = client.auth.logout()
        assert message is not None and isinstance(message, str)


def test_solver_options(data):
    m, canning_plants, markets, capacities, demands, distances = data
    # Set
    i = Set(
        m,
        name="i",
        records=canning_plants,
        description="canning plants",
    )
    j = Set(
        m,
        name="j",
        records=markets,
        description="markets",
    )

    # Data
    a = Parameter(
        m,
        name="a",
        domain=i,
        records=capacities,
        description="capacity of plant i in cases",
    )
    b = Parameter(
        m,
        name="b",
        domain=j,
        records=demands,
        description="demand at market j in cases",
    )
    d = Parameter(
        m,
        name="d",
        domain=[i, j],
        records=distances,
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
    client = EngineClient(
        host=os.environ["ENGINE_URL"],
        username=os.environ["ENGINE_USER"],
        password=os.environ["ENGINE_PASSWORD"],
        namespace=os.environ["ENGINE_NAMESPACE"],
    )
    transport.solve(
        output=sys.stdout,
        solver="conopt",
        solver_options={"rtmaxv": "1.e12"},
        backend="engine",
        client=client,
    )

    with open(m.gamsJobName() + ".lst") as file:
        assert ">>  rtmaxv 1.e12" in file.read()


def test_savepoint(data):
    m, canning_plants, markets, capacities, demands, distances = data
    # Set
    i = Set(
        m,
        name="i",
        records=canning_plants,
        description="canning plants",
    )
    j = Set(
        m,
        name="j",
        records=markets,
        description="markets",
    )

    # Data
    a = Parameter(
        m,
        name="a",
        domain=i,
        records=capacities,
        description="capacity of plant i in cases",
    )
    b = Parameter(
        m,
        name="b",
        domain=j,
        records=demands,
        description="demand at market j in cases",
    )
    d = Parameter(
        m,
        name="d",
        domain=[i, j],
        records=distances,
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
    client = EngineClient(
        host=os.environ["ENGINE_URL"],
        username=os.environ["ENGINE_USER"],
        password=os.environ["ENGINE_PASSWORD"],
        namespace=os.environ["ENGINE_NAMESPACE"],
    )
    transport.solve(
        output=sys.stdout,
        client=client,
        backend="engine",
        options=Options(savepoint=1),
    )
    assert transport.num_iterations == 4

    savepoint_path = os.path.join(m.working_directory, "transport_p.gdx")

    transport.solve(
        output=sys.stdout,
        client=client,
        backend="engine",
        options=Options(loadpoint=savepoint_path),
    )
    assert transport.num_iterations == 0

    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False
    ) as temp_file:
        transport.solve(
            output=temp_file, options=Options(loadpoint=Path(savepoint_path))
        )
        temp_file.close()
        with open(temp_file.name) as file:
            content = file.read()
            assert "GDX File (execute_load)" in content

        os.remove(temp_file.name)


def test_external_equation_on_engine(data):
    directory = str(pathlib.Path(__file__).parent.resolve())
    external_module = os.path.relpath(
        os.path.join(
            directory, "external_module", "build", "libsimple_ext_module"
        ),
        os.getcwd(),
    )

    if platform.system() == "Darwin":
        if platform.machine() == "arm64":
            external_module += "_arm64"

        external_module += ".dylib"
    elif platform.system() == "Linux":
        external_module += ".so"
    elif platform.system() == "Windows":
        external_module += ".dll"

    if platform.system() == "Linux" and platform.machine() == "aarch64":
        return

    if platform.system() == "Linux":
        m = Container(working_directory=".")
        y1 = Variable(m, "y1")
        y2 = Variable(m, "y2")
        x1 = Variable(m, "x1")
        x2 = Variable(m, "x2")

        eq1 = Equation(m, "eq1", type="external")
        eq2 = Equation(m, "eq2", type="external")

        eq1[...] = 1 * x1 + 3 * y1 == 1
        eq2[...] = 2 * x2 + 4 * y2 == 2

        model = Model(
            container=m,
            name="sincos",
            equations=m.getEquations(),
            problem="NLP",
            sense="min",
            objective=y1 + y2,
            external_module=external_module,
        )

        client = EngineClient(
            host=os.environ["ENGINE_URL"],
            username=os.environ["ENGINE_USER"],
            password=os.environ["ENGINE_PASSWORD"],
            namespace=os.environ["ENGINE_NAMESPACE"],
        )

        model.solve(
            output=sys.stdout,
            solver="conopt",
            backend="engine",
            client=client,
        )
        assert y1.toDense() == -1.0

        m.close()
