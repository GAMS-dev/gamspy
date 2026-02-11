from __future__ import annotations

import glob
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import gamspy as gp
import gamspy.math as gams_math
from gamspy import (
    Alias,
    Card,
    Container,
    Equation,
    Model,
    Number,
    Options,
    Ord,
    Parameter,
    Problem,
    Product,
    Sense,
    Set,
    Sum,
    Variable,
)
from gamspy._backend.neos import NeosClient
from gamspy.exceptions import ValidationError
from gamspy.math import cos, sin

pytestmark = pytest.mark.neos

try:
    from dotenv import load_dotenv

    load_dotenv(os.getcwd() + os.sep + ".env")
except Exception:
    pass


def fx(t):
    return sin(t) * cos(t - t * t)


def fy(t):
    return t * sin(t)


@pytest.fixture
def data():
    # Arrange
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

    # Act and assert
    yield m, canning_plants, markets, capacities, demands, distances

    # Cleanup
    m.close()
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
    files = glob.glob("_*")
    for file in files:
        if os.path.isfile(file):
            os.remove(file)


def test_network_license(network_license):
    m, canning_plants, markets, capacities, demands, distances = network_license
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
    client = NeosClient(
        email=os.environ["NEOS_EMAIL"],
    )
    with pytest.raises(ValidationError):
        transport.solve(backend="neos", client=client, solver="mpsge")

    with pytest.raises(ValidationError):
        transport.solve(backend="neos", client=client, solver="kestrel")

    summary = transport.solve(
        output=sys.stdout, solver="cplex", backend="neos", client=client
    )
    assert isinstance(summary, pd.DataFrame)

    import math

    assert math.isclose(transport.objective_value, 153.675000, rel_tol=0.001)


def test_neos_blocking(data):
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
    client = NeosClient(
        email=os.environ["NEOS_EMAIL"],
    )
    summary = transport.solve(backend="neos", client=client)
    assert isinstance(summary, pd.DataFrame)

    import math

    assert math.isclose(transport.objective_value, 153.675000, rel_tol=0.001)

    summary = transport.solve(solver="cbc", backend="neos", client=client)
    assert isinstance(summary, pd.DataFrame)
    assert math.isclose(transport.objective_value, 153.675000, rel_tol=0.001)


def test_no_client(data):
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
        transport.solve(backend="neos")


def test_different_solver(data):
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
    client = NeosClient(
        email=os.environ["NEOS_EMAIL"],
    )
    transport.solve(backend="neos", client=client, solver="cplex")

    import math

    assert math.isclose(transport.objective_value, 153.675000, rel_tol=0.001)


def test_neos_non_blocking(data):
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
    client = NeosClient(
        email=os.environ["NEOS_EMAIL"],
        is_blocking=False,
    )
    transport.solve(solver="cplex", backend="neos", client=client)

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
    assert "x" in container.data
    x.setRecords(container["x"].records)
    assert x.records.equals(container["x"].records)


def test_solver_options(data, tmp_path):
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
    client = NeosClient(
        email=os.environ["NEOS_EMAIL"],
    )

    log_path = str(tmp_path / "neos.log")
    transport.solve(
        solver="cplex",
        solver_options={"aggfill": "11"},
        backend="neos",
        client=client,
        options=Options(log_file=log_path),
    )

    with open(log_path) as file:
        assert ">>  aggfill 11" in file.read()

    # Read solver options from an existing file
    log_path2 = str(tmp_path / "neos2.log")
    gp.set_options({"SOLVER_OPTION_VALIDATION": 0})
    with tempfile.TemporaryDirectory() as tmpdir:
        options_path = os.path.join(tmpdir, "my_solver_options.opt")
        with open(options_path, "w") as file:
            file.write("aggfill 55")

        transport.solve(
            solver="cplex",
            solver_options=options_path,
            backend="neos",
            client=client,
            options=Options(log_file=log_path2),
        )

    with open(log_path2) as file:
        content = file.read()
        assert ">>  aggfill 55" in content

    gp.set_options({"SOLVER_OPTION_VALIDATION": 1})


def test_lp(data):
    mdl = Container()
    x1 = Variable(mdl)
    x2 = Variable(mdl)
    x3 = Variable(mdl)

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
        problem="lp",
        sense="min",
        objective=obj,
    )

    client = NeosClient(email=os.environ["NEOS_EMAIL"])
    summary = LP1.solve(
        solver="cplex", output=sys.stdout, backend="neos", client=client
    )
    assert isinstance(summary, pd.DataFrame)


def test_mip(data):
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

    MIP1 = Model(
        mdl,
        equations=mdl.getEquations(),
        problem="mip",
        sense="min",
        objective=obj,
    )

    client = NeosClient(email=os.environ["NEOS_EMAIL"])
    summary = MIP1.solve(
        solver="cplex", output=sys.stdout, backend="neos", client=client
    )
    assert isinstance(summary, pd.DataFrame)


def test_rmip(data):
    m, *_ = data

    # Sets
    i = Set(
        m,
        "i",
        records=[f"w{idx}" for idx in range(1, 5)],
        description="widths",
    )
    p = Set(
        m,
        "p",
        records=[f"p{idx}" for idx in range(1, 1001)],
        description="possible patterns",
    )
    pp = Set(m, "pp", domain=p, description="dynamic subset of p")

    # Parameters
    r = Parameter(m, "r", records=100, description="raw width")
    w = Parameter(
        m,
        "w",
        domain=i,
        records=[["w1", 45], ["w2", 36], ["w3", 31], ["w4", 14]],
        description="width",
    )
    d = Parameter(
        m,
        "d",
        domain=i,
        records=[["w1", 97], ["w2", 610], ["w3", 395], ["w4", 211]],
        description="demand",
    )
    aip = Parameter(
        m,
        "aip",
        domain=[i, p],
        description="number of width i in pattern growing in p",
    )

    # Master model variables
    xp = Variable(m, "xp", domain=p, type="integer", description="patterns used")
    z = Variable(m, "z", description="objective variable")
    xp.up[p] = Sum(i, d[i])

    # Master model equations
    numpat = Equation(
        m,
        "numpat",
        definition=z == Sum(pp, xp[pp]),
        description="number of patterns used",
    )
    demand = Equation(m, "demand", domain=i, description="meet demand")
    demand[i] = Sum(pp, aip[i, pp] * xp[pp]) >= d[i]

    master = Model(
        m,
        "master",
        equations=[numpat, demand],
        problem="rmip",
        sense=Sense.MIN,
        objective=z,
    )

    # Pricing model variables
    y = Variable(m, "y", domain=i, type="integer", description="new pattern")
    y.up[i] = gams_math.ceil(r / w[i])

    defobj = Equation(m, "defobj", definition=z == (1 - Sum(i, demand.m[i] * y[i])))
    knapsack = Equation(
        m,
        "knapsack",
        description="knapsack constraint",
        definition=Sum(i, w[i] * y[i]) <= r,
    )

    pricing = Model(
        m,
        "pricing",
        equations=[defobj, knapsack],
        problem="mip",
        sense=Sense.MIN,
        objective=z,
    )

    pp[p] = Ord(p) <= Card(i)
    aip[i, pp[p]].where[Ord(i) == Ord(p)] = gams_math.floor(r / w[i])

    pi = Set(m, "pi", domain=p, description="set of the last pattern")
    pi[p] = Ord(p) == Card(pp) + 1

    client = NeosClient(email=os.environ["NEOS_EMAIL"])
    while len(pp) < len(p):
        master.solve(
            backend="neos",
            client=client,
            options=Options(relative_optimality_gap=0),
        )
        pricing.solve(
            backend="neos",
            client=client,
            options=Options(relative_optimality_gap=0),
        )

        if z.records["level"].values[0] >= -0.001:
            break

        aip[i, pi] = gams_math.Round(y.l[i])
        pp[pi] = True
        pi[p] = pi[p.lag(1)]

    master.problem = "mip"
    master.solve(
        backend="neos",
        client=client,
        options=Options(relative_optimality_gap=0),
    )

    import math

    assert math.isclose(master.objective_value, 453.0000, rel_tol=0.001)


def test_nlp(data):
    mdl = Container()
    x1 = Variable(mdl)
    x2 = Variable(mdl)
    x3 = Variable(mdl)

    eq1 = Equation(mdl)
    eq2 = Equation(mdl)
    eq3 = Equation(mdl)
    eq1[...] = x1 * x2 >= 3
    eq2[...] = x3 + x2 >= 5
    eq3[...] = x1 + x3 == 4
    obj = x1 + 3 * x2 + 3 * x3

    NLP1 = Model(
        mdl,
        equations=mdl.getEquations(),
        problem="nlp",
        sense="min",
        objective=obj,
    )

    client = NeosClient(email=os.environ["NEOS_EMAIL"])
    summary = NLP1.solve(
        solver="CONOPT", output=sys.stdout, backend="neos", client=client
    )
    assert isinstance(summary, pd.DataFrame)


def test_dnlp(data):
    m, *_ = data

    # Set
    i = Set(
        m,
        name="i",
        records=["1", "2", "3", "4"],
        description="corner points of square",
    )

    # Variable
    t = Variable(
        m,
        name="t",
        domain=i,
        description="position of square corner points on curve",
    )
    x = Variable(
        m,
        name="x",
        description=("x-coordinate of lower-left corner of square (=fx(t('1')))"),
    )
    y = Variable(
        m,
        name="y",
        description=("y-coordinate of lower-left corner of square (=fy(t('1')))"),
    )
    a = Variable(
        m,
        name="a",
        type="Positive",
        description=(
            "horizontal distance between lower-left and lower-right corner of square"
        ),
    )
    b = Variable(
        m,
        name="b",
        type="Positive",
        description=(
            "vertical distance between lower-left and lower-right corner of square"
        ),
    )

    t.lo[i] = -math.pi
    t.up[i] = math.pi

    # Equation
    e1x = Equation(
        m, name="e1x", description="define x-coordinate of lower-left corner"
    )
    e1y = Equation(
        m, name="e1y", description="define y-coordinate of lower-left corner"
    )
    e2x = Equation(
        m, name="e2x", description="define x-coordinate of lower-right corner"
    )
    e2y = Equation(
        m, name="e2y", description="define y-coordinate of lower-right corner"
    )
    e3x = Equation(
        m, name="e3x", description="define x-coordinate of upper-left corner"
    )
    e3y = Equation(
        m, name="e3y", description="define y-coordinate of upper-left corner"
    )
    e4x = Equation(
        m, name="e4x", description="define x-coordinate of upper-right corner"
    )
    e4y = Equation(
        m, name="e4y", description="define y-coordinate of upper-right corner"
    )

    obj = a**2 + b**2  # Area of square to be maximized

    e1x[...] = fx(t["1"]) == x
    e1y[...] = fy(t["1"]) == y
    e2x[...] = fx(t["2"]) == x + a
    e2y[...] = fy(t["2"]) == y + b
    e3x[...] = fx(t["3"]) == x - b
    e3y[...] = fy(t["3"]) == y + a
    e4x[...] = fx(t["4"]) == x + a - b
    e4y[...] = fy(t["4"]) == y + a + b

    square = Model(
        m,
        name="square",
        equations=m.getEquations(),
        problem="DNLP",
        sense=Sense.MAX,
        objective=obj,
    )

    t.l[i] = -math.pi + (Ord(i) - 1) * 2 * math.pi / Card(i)
    x.l[...] = fx(t.l["1"])
    y.l[...] = fy(t.l["1"])
    a.l[...] = 1
    b.l[...] = 1

    client = NeosClient(email=os.environ["NEOS_EMAIL"])
    summary = square.solve(
        solver="conopt", output=sys.stdout, backend="neos", client=client
    )
    assert isinstance(summary, pd.DataFrame)


def test_mcp():
    cont = Container(
        load_from=str(Path(__file__).parent.absolute()) + "/models/qp6.gdx",
    )

    # Sets
    days, stocks = cont.getSymbols(["days", "stocks"])

    # Parameters
    returns, _val = cont.getSymbols(["return", "val"])

    # Set
    d = Set(cont, name="d", domain=[days], description="selected days")
    s = Set(cont, name="s", domain=[stocks], description="selected stocks")

    # select subset of stocks and periods
    d[days] = (Ord(days) > 1) & (Ord(days) < 31)
    s[stocks] = Ord(stocks) < 51

    # Parameter
    mean = Parameter(
        cont, name="mean", domain=stocks, description="mean of daily return"
    )
    dev = Parameter(cont, name="dev", domain=[stocks, days], description="deviations")
    totmean = Parameter(cont, name="totmean", description="total mean return")

    mean[s] = Sum(d, returns[s, d]) / Card(d)
    dev[s, d] = returns[s, d] - mean[s]
    totmean[...] = Sum(s, mean[s]) / (Card(s))

    # Variable
    x = Variable(
        cont,
        name="x",
        type="positive",
        domain=stocks,
        description="investments",
    )
    w = Variable(
        cont,
        name="w",
        type="free",
        domain=days,
        description="intermediate variables",
    )

    # Equation
    budget = Equation(cont, name="budget")
    retcon = Equation(cont, name="retcon", description="returns constraint")
    wdef = Equation(cont, name="wdef", domain=days)

    wdef[d] = w[d] == Sum(s, x[s] * dev[s, d])

    budget[...] = Sum(s, x[s]) == Number(1.0)

    retcon[...] = Sum(s, mean[s] * x[s]) >= totmean * 1.25

    # Equation
    d_x = Equation(cont, name="d_x", domain=stocks)
    d_w = Equation(cont, name="d_w", domain=days)

    # Variable
    m_budget = Variable(cont, name="m_budget", type="free")
    m_wdef = Variable(cont, name="m_wdef", type="free", domain=days)

    # Positive Variable
    m_retcon = Variable(cont, name="m_retcon", type="positive")

    m_wdef.fx[days].where[~d[days]] = 0

    d_x[s] = Sum(d, m_wdef[d] * dev[s, d]) >= m_retcon * mean[s] + m_budget

    d_w[d] = 2 * w[d] / (Card(d) - 1) == m_wdef[d]

    qp6 = Model(
        cont,
        name="qp6",
        matches={
            d_x: x,
            d_w: w,
            retcon: m_retcon,
            budget: m_budget,
            wdef: m_wdef,
        },
        problem="mcp",
    )

    client = NeosClient(email=os.environ["NEOS_EMAIL"])
    summary = qp6.solve(solver="path", backend="neos", client=client)
    assert isinstance(summary, pd.DataFrame)


def test_cns():
    data = [
        [0.00000, 0.00000, 0.00000],
        [0.01000, 0.03920, 0.05000],
        [0.02000, 0.07000, 0.91000],
        [0.13000, 0.29000, 0.58000],
        [0.00000, 0.00000, 0.00000],
        [0.00000, 0.00000, 0.00000],
        [0.10000, 0.22751, 0.08084],
        [0.61447, 1.60111, 0.52019],
        [0.33263, 0.43486, 0.23251],
        [0.90909, 0.81466, 0.92521],
        [1.00000, 1.00000, 1.00000],
        [2.00000, 0.66000, 0.40000],
        [0.24820, 0.05111, 0.00001],
        [1.59539, 1.34652, 1.01839],
        [2.00000, 2.00000, 2.00000],
        [0.86628, 0.84602, 0.82436],
        [3.85424, 3.51886, 3.23592],
    ]

    columns = ["agricult", "industry", "services"]
    indexes = [
        "depr",
        "itax",
        "gles",
        "kio",
        "dstr",
        "te",
        "tm",
        "ad",
        "pwts",
        "pwm",
        "pwe",
        "sigc",
        "delta",
        "ac",
        "sigt",
        "gamma",
        "at",
    ]
    zz_df = pd.DataFrame(data=data, columns=columns, index=indexes)
    zz_df = zz_df.reset_index().melt(
        id_vars="index", var_name="Category", value_name="Value"
    )

    # sectres
    data = [
        [1.0000, 1.0000, 1.0000],
        [1.0000, 1.0000, 1.0000],
        [0.7370, 0.2911, 0.6625],
        [711.6443, 930.3509, 497.4428],
        [657.3677, 840.0500, 515.4296],
        [641.7037, 812.2222, 492.0307],
        [15.6639, 27.8278, 23.3988],
        [69.9406, 118.1287, 5.4120],
        [657.5754, 338.7076, 1548.5192],
        [256.6450, 464.1656, 156.2598],
        [452.1765, 307.8561, 202.0416],
        [2.8230, 9.8806, 128.4482],
        [0.0000, 148.4488, 10.6931],
        [0.0000, 0.0000, 0.0000],
        [20.6884, 46.1511, 92.3023],
        [1.0000, 1.0000, 1.0000],
        [1.0000, 1.0000, 1.0000],
        [1.0000, 1.0000, 1.0000],
        [1.0000, 1.0000, 1.0000],
    ]
    columns = ["agricult", "industry", "services"]
    indexes = [
        "pd1",
        "pk",
        "pva",
        "x",
        "xd",
        "xxd",
        "e",
        "m",
        "k",
        "intr",
        "cd",
        "gd",
        "id",
        "dst",
        "dk",
        "pm",
        "pe",
        "px",
        "p",
    ]
    sectres_df = pd.DataFrame(data=data, columns=columns, index=indexes)
    sectres_df = sectres_df.reset_index().melt(
        id_vars="index", var_name="Category", value_name="Value"
    )

    # Model's container
    cont = Container()

    # Sets
    i = Set(
        cont,
        name="i",
        records=["agricult", "industry", "services"],
        description="sectors",
    )
    hh = Set(
        cont,
        name="hh",
        records=["lab_hh", "cap_hh"],
        description="household type",
    )
    lc = Set(
        cont,
        name="lc",
        records=["labor1", "labor2", "labor3"],
        description="labor categories",
    )
    it = Set(cont, name="it", domain=i, description="traded sectors")
    inn = Set(cont, name="inn", domain=i, description="nontraded sectors")

    j = Alias(cont, name="j", alias_with=i)

    # Parameters
    delta = Parameter(
        cont,
        name="delta",
        domain=i,
        description="Armington function share parameter",
    )
    ac = Parameter(
        cont,
        name="ac",
        domain=i,
        description="Armington function shift parameter",
    )
    rhoc = Parameter(
        cont, name="rhoc", domain=i, description="Armington function exponent"
    )
    rhot = Parameter(cont, name="rhot", domain=i, description="cet function exponent")
    at = Parameter(
        cont, name="at", domain=i, description="cet function shift parameter"
    )
    gamma = Parameter(
        cont,
        name="gamma",
        domain=i,
        description="cet function share parameter",
    )
    ad = Parameter(
        cont,
        name="ad",
        domain=i,
        description="production function shift parameter",
    )
    gles = Parameter(
        cont,
        name="gles",
        domain=i,
        description="government consumption shares",
    )
    depr = Parameter(cont, name="depr", domain=i, description="depreciation rates")
    dstr = Parameter(
        cont,
        name="dstr",
        domain=i,
        description="ratio of inventory investment to gross output",
    )
    kio = Parameter(
        cont,
        name="kio",
        domain=i,
        description="shares of investment by sector of destination",
    )
    te = Parameter(cont, name="te", domain=i, description="export duty rates")
    itax = Parameter(cont, name="itax", domain=i, description="indirect tax rates")
    htax = Parameter(
        cont,
        name="htax",
        domain=hh,
        description="income tax rate by household type",
    )
    pwm = Parameter(
        cont,
        name="pwm",
        domain=i,
        description="world market price of imports    (in dollars)",
    )
    pwe = Parameter(
        cont,
        name="pwe",
        domain=i,
        description="world market price of exports    (in dollars)",
    )
    tm = Parameter(cont, name="tm", domain=i, description="tariff rates on imports")
    pwts = Parameter(cont, name="pwts", domain=i, description="cpi weights")

    htax["lab_hh"] = 0.08910
    htax["cap_hh"] = 0.08910

    alphl = Parameter(
        cont,
        name="alphl",
        domain=[i, lc],
        records=np.array(
            [
                [0.38258, 0.06740, 0.00000],
                [0.00000, 0.53476, 0.00000],
                [0.00000, 0.16234, 0.42326],
            ]
        ),
        description="labor share parameter in production function",
    )

    io = Parameter(
        cont,
        name="io",
        domain=[i, j],
        records=np.array(
            [
                [0.12591, 0.19834, 0.01407],
                [0.10353, 0.35524, 0.18954],
                [0.02358, 0.11608, 0.08390],
            ]
        ),
        description="input-output coefficients",
    )

    imat = Parameter(
        cont,
        name="imat",
        domain=[i, j],
        records=np.array(
            [
                [0.00000, 0.00000, 0.00000],
                [0.93076, 0.93774, 0.93080],
                [0.06924, 0.06226, 0.06920],
            ]
        ),
        description="capital composition matrix",
    )

    wdist = Parameter(
        cont,
        name="wdist",
        domain=[i, lc],
        records=np.array(
            [
                [1.00000, 0.52780, 0.00000],
                [0.00000, 1.21879, 0.00000],
                [0.00000, 1.11541, 1.00000],
            ]
        ),
        description="wage proportionality factors",
    )

    cles = Parameter(
        cont,
        name="cles",
        domain=[i, hh],
        records=np.array([[0.47000, 0.47000], [0.31999, 0.31999], [0.21001, 0.21001]]),
        description="private consumption shares",
    )

    zz = Parameter(
        cont,
        name="zz",
        domain=["*", i],
        records=zz_df,
        description="miscellaneous parameters",
    )

    depr[i] = zz["depr", i]
    itax[i] = zz["itax", i]
    gles[i] = zz["gles", i]
    kio[i] = zz["kio", i]
    dstr[i] = zz["dstr", i]
    te[i] = zz["te", i]
    tm[i] = zz["tm", i]
    ad[i] = zz["ad", i]
    pwts[i] = zz["pwts", i]
    pwm[i] = zz["pwm", i]
    pwe[i] = zz["pwe", i]
    rhoc[i] = (1 / zz["sigc", i]) - 1
    delta[i] = zz["delta", i]
    ac[i] = zz["ac", i]
    rhot[i] = (1 / zz["sigt", i]) + 1
    gamma[i] = zz["gamma", i]
    at[i] = zz["at", i]

    # Model Definition
    # Variables
    # prices block
    er = Variable(
        cont,
        name="er",
        type="free",
        description=("real exchange rate                          (won per dollar)"),
    )
    pd1 = Variable(
        cont, name="pd1", type="free", domain=i, description="domestic prices"
    )
    pm = Variable(
        cont,
        name="pm",
        type="free",
        domain=i,
        description="domestic price of imports",
    )
    pe = Variable(
        cont,
        name="pe",
        type="free",
        domain=i,
        description="domestic price of exports",
    )
    pk = Variable(
        cont,
        name="pk",
        type="free",
        domain=i,
        description="rate of capital rent by sector",
    )
    px = Variable(
        cont,
        name="px",
        type="free",
        domain=i,
        description="average output price by sector",
    )
    p = Variable(
        cont,
        name="p",
        type="free",
        domain=i,
        description="price of composite goods",
    )
    pva = Variable(
        cont,
        name="pva",
        type="free",
        domain=i,
        description="value added price by sector",
    )
    pr = Variable(cont, name="pr", type="free", description="import premium")
    pindex = Variable(
        cont, name="pindex", type="free", description="general price level"
    )

    # production block
    x = Variable(
        cont,
        name="x",
        type="free",
        domain=i,
        description=("composite goods supply                        ('68 bill won)"),
    )
    xd = Variable(
        cont,
        name="xd",
        type="free",
        domain=i,
        description=("domestic output by sector                     ('68 bill won)"),
    )
    xxd = Variable(
        cont,
        name="xxd",
        type="free",
        domain=i,
        description=("domestic sales                                ('68 bill won)"),
    )
    e = Variable(
        cont,
        name="e",
        type="free",
        domain=i,
        description=("exports by sector                             ('68 bill won)"),
    )
    m = Variable(
        cont,
        name="m",
        type="free",
        domain=i,
        description=("imports                                       ('68 bill won)"),
    )

    # factors block
    k = Variable(
        cont,
        name="k",
        type="free",
        domain=i,
        description=("capital stock by sector                       ('68 bill won)"),
    )
    wa = Variable(
        cont,
        name="wa",
        type="free",
        domain=lc,
        description=("average wage rate by labor category     (mill won pr person)"),
    )
    ls = Variable(
        cont,
        name="ls",
        type="free",
        domain=lc,
        description=("labor supply by labor category                (1000 persons)"),
    )
    l = Variable(
        cont,
        name="l",
        type="free",
        domain=[i, lc],
        description=("employment by sector and labor category       (1000 persons)"),
    )

    # demand block
    intr = Variable(
        cont,
        name="intr",
        type="free",
        domain=i,
        description=("intermediates uses                            ('68 bill won)"),
    )
    cd = Variable(
        cont,
        name="cd",
        type="free",
        domain=i,
        description=("final demand for private consumption          ('68 bill won)"),
    )
    gd = Variable(
        cont,
        name="gd",
        type="free",
        domain=i,
        description=("final demand for government consumption       ('68 bill won)"),
    )
    id = Variable(
        cont,
        name="id",
        type="free",
        domain=i,
        description=("final demand for productive investment        ('68 bill won)"),
    )
    dst = Variable(
        cont,
        name="dst",
        type="free",
        domain=i,
        description=("inventory investment by sector                ('68 bill won)"),
    )
    y = Variable(
        cont,
        name="y",
        type="free",
        description=("private gdp                                       (bill won)"),
    )
    gr = Variable(
        cont,
        name="gr",
        type="free",
        description=("government revenue                                (bill won)"),
    )
    tariff = Variable(
        cont,
        name="tariff",
        type="free",
        description=("tariff revenue                                    (bill won)"),
    )
    indtax = Variable(
        cont,
        name="indtax",
        type="free",
        description=("indirect tax revenue                              (bill won)"),
    )
    netsub = Variable(
        cont,
        name="netsub",
        type="free",
        description=("export duty revenue                               (bill won)"),
    )
    gdtot = Variable(
        cont,
        name="gdtot",
        type="free",
        description=("total volume of government consumption        ('68 bill won)"),
    )
    hhsav = Variable(
        cont,
        name="hhsav",
        type="free",
        description=("total household savings                           (bill won)"),
    )
    govsav = Variable(
        cont,
        name="govsav",
        type="free",
        description=("government savings                                (bill won)"),
    )
    deprecia = Variable(
        cont,
        name="deprecia",
        type="free",
        description=("total depreciation expenditure                    (bill won)"),
    )
    invest = Variable(
        cont,
        name="invest",
        type="free",
        description=("total investment                                  (bill won)"),
    )
    savings = Variable(
        cont,
        name="savings",
        type="free",
        description=("total savings                                     (bill won)"),
    )
    mps = Variable(
        cont,
        name="mps",
        type="free",
        domain=hh,
        description="marginal propensity to save by household type",
    )
    fsav = Variable(
        cont,
        name="fsav",
        type="free",
        description=("foreign savings                               (bill dollars)"),
    )
    dk = Variable(
        cont,
        name="dk",
        type="free",
        domain=i,
        description=("volume of investment by sector of destination ('68 bill won)"),
    )
    ypr = Variable(
        cont,
        name="ypr",
        type="free",
        description=("total premium income accruing to capitalists      (bill won)"),
    )
    remit = Variable(
        cont,
        name="remit",
        type="free",
        description=("net remittances from abroad                   (bill dollars)"),
    )
    fbor = Variable(
        cont,
        name="fbor",
        type="free",
        description=("net flow of foreign borrowing                 (bill dollars)"),
    )
    yh = Variable(
        cont,
        name="yh",
        type="free",
        domain=hh,
        description=("total income by household type                    (bill won)"),
    )
    tothhtax = Variable(
        cont,
        name="tothhtax",
        type="free",
        description=("household tax revenue                             (bill won)"),
    )

    # welfare indicator for objective function
    omega = Variable(
        cont,
        name="omega",
        type="free",
        description=("objective function variable                   ('68 bill won)"),
    )

    er.l[...] = 1.0000
    pr.l[...] = 0.0000
    pindex.l[...] = 1.0000
    gr.l[...] = 194.0449
    tariff.l[...] = 28.6572
    indtax.l[...] = 65.2754
    netsub.l[...] = 0.0000
    gdtot.l[...] = 141.1519
    hhsav.l[...] = 61.4089
    govsav.l[...] = 52.8930
    deprecia.l[...] = 0.0000
    savings.l[...] = 159.1419
    invest.l[...] = 159.1419
    fsav.l[...] = 39.1744
    fbor.l[...] = 58.7590
    remit.l[...] = 0.0000
    tothhtax.l[...] = 100.1122
    y.l[...] = 1123.5941

    labres1 = Parameter(
        cont,
        name="labres1",
        domain=[i, lc],
        records=np.array(
            [
                [2515.900, 442.643, 0.000],
                [0.000, 767.776, 0.000],
                [0.000, 355.568, 948.100],
            ]
        ),
        description="summary matrix with sectoral employment results",
    )

    labres2 = Parameter(
        cont,
        name="labres2",
        domain=["*", lc],
        records=pd.DataFrame(
            [
                ["wa", "labor1", 0.074],
                ["ls", "labor1", 2515.9],
                ["wa", "labor2", 0.14],
                ["ls", "labor2", 1565.987],
                ["wa", "labor3", 0.152],
                ["ls", "labor3", 948.1],
            ]
        ),
        description="summary matrix with aggregate employment results",
    )

    hhres = Parameter(
        cont,
        name="hhres",
        domain=["*", hh],
        records=pd.DataFrame(
            [
                ["yh", "lab_hh", 548.7478],
                ["mps", "lab_hh", 0.06],
                ["yh", "cap_hh", 574.8463],
                ["mps", "cap_hh", 0.06],
            ]
        ),
        description="summary matrix with household results",
    )

    l.l[i, lc] = labres1[i, lc]
    ls.l[lc] = labres2["ls", lc]
    wa.l[lc] = labres2["wa", lc]
    mps.l[hh] = hhres["mps", hh]
    yh.l[hh] = hhres["yh", hh]

    sectres = Parameter(
        cont,
        name="sectres",
        domain=["*", i],
        records=sectres_df,
        description="summary matrix with sectoral results",
    )

    pd1.l[i] = sectres["pd1", i]
    pm.l[i] = sectres["pm", i]
    pe.l[i] = sectres["pe", i]
    pk.l[i] = sectres["pk", i]
    px.l[i] = sectres["px", i]
    p.l[i] = sectres["p", i]
    pva.l[i] = sectres["pva", i]
    x.l[i] = sectres["x", i]
    xd.l[i] = sectres["xd", i]
    xxd.l[i] = sectres["xxd", i]
    e.l[i] = sectres["e", i]
    m.l[i] = sectres["m", i]
    k.l[i] = sectres["k", i]
    intr.l[i] = sectres["intr", i]
    cd.l[i] = sectres["cd", i]
    gd.l[i] = sectres["gd", i]
    id.l[i] = sectres["id", i]
    dst.l[i] = sectres["dst", i]
    dk.l[i] = sectres["dk", i]
    it[i] = Number(1).where[e.l[i] | m.l[i]]
    inn[i] = ~it[i]
    k.fx[i] = k.l[i]
    m.fx[inn] = 0
    e.fx[inn] = 0
    l.fx[i, lc].where[l.l[i, lc] == 0] = 0

    p.lo[i] = 0.01
    pd1.lo[i] = 0.01
    pm.lo[it] = 0.01
    pk.lo[i] = 0.01
    px.lo[i] = 0.01
    x.lo[i] = 0.01
    xd.lo[i] = 0.01
    m.lo[it] = 0.01
    xxd.lo[it] = 0.01
    wa.lo[lc] = 0.01
    intr.lo[i] = 0.01
    y.lo[...] = 0.01
    e.lo[it] = 0.01
    l.lo[i, lc].where[l.l[i, lc] != 0] = 0.01

    # Equation Definitions
    # price block
    pmdef = Equation(
        cont,
        name="pmdef",
        domain=i,
        description="definition of domestic import prices",
    )
    pedef = Equation(
        cont,
        name="pedef",
        domain=i,
        description="definition of domestic export prices",
    )
    absorption = Equation(
        cont,
        name="absorption",
        domain=i,
        description="value of domestic sales",
    )
    sales = Equation(
        cont, name="sales", domain=i, description="value of domestic output"
    )
    actp = Equation(
        cont,
        name="actp",
        domain=i,
        description="definition of activity prices",
    )
    pkdef = Equation(
        cont,
        name="pkdef",
        domain=i,
        description="definition of capital goods price",
    )
    pindexdef = Equation(
        cont, name="pindexdef", description="definition of general price level"
    )

    # output block
    activity = Equation(
        cont, name="activity", domain=i, description="production function"
    )
    profitmax = Equation(
        cont,
        name="profitmax",
        domain=[i, lc],
        description="first order condition for profit maximum",
    )
    lmequil = Equation(
        cont, name="lmequil", domain=lc, description="labor market equilibrium"
    )
    cet = Equation(cont, name="cet", domain=i, description="cet function")
    esupply = Equation(cont, name="esupply", domain=i, description="export supply")
    armington = Equation(
        cont,
        name="armington",
        domain=i,
        description="composite good aggregation function",
    )
    costmin = Equation(
        cont,
        name="costmin",
        domain=i,
        description="f.o.c. for cost minimization of composite good",
    )
    xxdsn = Equation(
        cont,
        name="xxdsn",
        domain=i,
        description="domestic sales for nontraded sectors",
    )
    xsn = Equation(
        cont,
        name="xsn",
        domain=i,
        description="composite good agg. for nontraded sectors",
    )

    # demand block
    inteq = Equation(
        cont, name="inteq", domain=i, description="total intermediate uses"
    )
    cdeq = Equation(
        cont, name="cdeq", domain=i, description="private consumption behavior"
    )
    dsteq = Equation(cont, name="dsteq", domain=i, description="inventory investment")
    gdp = Equation(cont, name="gdp", description="private gdp")
    labory = Equation(cont, name="labory", description="total income accruing to labor")
    capitaly = Equation(
        cont, name="capitaly", description="total income accruing to capital"
    )
    hhtaxdef = Equation(
        cont,
        name="hhtaxdef",
        description="total household taxes collected by govt.",
    )
    gdeq = Equation(
        cont,
        name="gdeq",
        domain=i,
        description="government consumption shares",
    )
    greq = Equation(cont, name="greq", description="government revenue")
    tariffdef = Equation(cont, name="tariffdef", description="tariff revenue")
    premium = Equation(cont, name="premium", description="total import premium income")
    indtaxdef = Equation(
        cont,
        name="indtaxdef",
        description="indirect taxes on domestic production",
    )
    netsubdef = Equation(cont, name="netsubdef", description="export duties")

    # savings-investment block
    hhsaveq = Equation(cont, name="hhsaveq", description="household savings")
    gruse = Equation(cont, name="gruse", description="government savings")
    depreq = Equation(cont, name="depreq", description="depreciation expenditure")
    totsav = Equation(cont, name="totsav", description="total savings")
    prodinv = Equation(
        cont,
        name="prodinv",
        domain=i,
        description="investment by sector of destination",
    )
    ieq = Equation(
        cont,
        name="ieq",
        domain=i,
        description="investment by sector of origin",
    )

    # balance of payments
    caeq = Equation(
        cont, name="caeq", description="current account balance (bill dollars)"
    )

    # market clearing
    equil = Equation(
        cont, name="equil", domain=i, description="goods market equilibrium"
    )

    # objective function
    obj = Equation(cont, name="obj", description="objective function")

    # price block
    pmdef[it] = pm[it] == pwm[it] * er * (1 + tm[it] + pr)

    pedef[it] = pe[it] == pwe[it] * (1 + te[it]) * er

    absorption[i] = p[i] * x[i] == pd1[i] * xxd[i] + (pm[i] * m[i]).where[it[i]]

    sales[i] = px[i] * xd[i] == pd1[i] * xxd[i] + (pe[i] * e[i]).where[it[i]]

    actp[i] = px[i] * (1 - itax[i]) == pva[i] + Sum(j, io[j, i] * p[j])

    pkdef[i] = pk[i] == Sum(j, p[j] * imat[j, i])

    pindexdef[...] = pindex == Sum(i, pwts[i] * p[i])

    # output and factors of production block
    activity[i] = xd[i] == ad[i] * Product(
        lc.where[wdist[i, lc]], l[i, lc] ** alphl[i, lc]
    ) * k[i] ** (1 - Sum(lc, alphl[i, lc]))

    profitmax[i, lc].where[wdist[i, lc]] = (
        wa[lc] * wdist[i, lc] * l[i, lc] == xd[i] * pva[i] * alphl[i, lc]
    )

    lmequil[lc] = Sum(i, l[i, lc]) == ls[lc]

    cet[it] = xd[it] == at[it] * (
        gamma[it] * e[it] ** rhot[it] + (1 - gamma[it]) * xxd[it] ** rhot[it]
    ) ** (1 / rhot[it])

    esupply[it] = e[it] / xxd[it] == (
        pe[it] / pd1[it] * (1 - gamma[it]) / gamma[it]
    ) ** (1 / (rhot[it] - 1))

    armington[it] = x[it] == ac[it] * (
        delta[it] * m[it] ** (rhoc[it] * (-1))
        + (1 - delta[it]) * xxd[it] ** (rhoc[it] * (-1))
    ) ** (-1 / rhoc[it])

    costmin[it] = m[it] / xxd[it] == (
        pd1[it] / pm[it] * delta[it] / (1 - delta[it])
    ) ** (1 / (1 + rhoc[it]))

    xxdsn[inn] = xxd[inn] == xd[inn]

    xsn[inn] = x[inn] == xxd[inn]

    # demand block
    inteq[i] = intr[i] == Sum(j, io[i, j] * xd[j])

    dsteq[i] = dst[i] == dstr[i] * xd[i]

    cdeq[i] = p[i] * cd[i] == Sum(
        hh, cles[i, hh] * (1 - mps[hh]) * yh[hh] * (1 - htax[hh])
    )

    gdp[...] = y == Sum(hh, yh[hh])

    labory[...] = yh["lab_hh"] == Sum(lc, wa[lc] * ls[lc]) + remit * er

    capitaly[...] = (
        yh["cap_hh"]
        == Sum(i, pva[i] * xd[i])
        - deprecia
        - Sum(lc, wa[lc] * ls[lc])
        + fbor * er
        + ypr
    )

    hhsaveq[...] = hhsav == Sum(hh, mps[hh] * yh[hh] * (1 - htax[hh]))

    greq[...] = gr == tariff - netsub + indtax + tothhtax

    gruse[...] = gr == Sum(i, p[i] * gd[i]) + govsav

    gdeq[i] = gd[i] == gles[i] * gdtot

    tariffdef[...] = tariff == Sum(it, tm[it] * m[it] * pwm[it]) * er

    indtaxdef[...] = indtax == Sum(i, itax[i] * px[i] * xd[i])

    netsubdef[...] = netsub == Sum(it, te[it] * e[it] * pwe[it]) * er

    premium[...] = ypr == Sum(it, pwm[it] * m[it]) * er * pr

    hhtaxdef[...] = tothhtax == Sum(hh, htax[hh] * yh[hh])

    depreq[...] = deprecia == Sum(i, depr[i] * pk[i] * k[i])

    totsav[...] = savings == hhsav + govsav + deprecia + fsav * er

    prodinv[i] = pk[i] * dk[i] == kio[i] * invest - kio[i] * Sum(j, dst[j] * p[j])

    ieq[i] = id[i] == Sum(j, imat[i, j] * dk[j])

    # balance of payments
    caeq[...] = (
        Sum(it, pwm[it] * m[it]) == Sum(it, pwe[it] * e[it]) + fsav + remit + fbor
    )
    # market clearing
    equil[i] = x[i] == intr[i] + cd[i] + gd[i] + id[i] + dst[i]

    # objective function
    obj[...] = omega == Product(i.where[cles[i, "lab_hh"]], cd[i] ** cles[i, "lab_hh"])

    er.fx[...] = er.l
    fsav.fx[...] = fsav.l
    remit.fx[...] = remit.l
    fbor.fx[...] = fbor.l
    pindex.fx[...] = pindex.l
    mps.fx[hh] = mps.l[hh]
    gdtot.fx[...] = gdtot.l
    ls.fx[lc] = ls.l[lc]

    model1 = Model(cont, name="model1", equations=cont.getEquations(), problem="cns")
    client = NeosClient(email=os.environ["NEOS_EMAIL"])
    summary = model1.solve(
        solver="path", output=sys.stdout, backend="neos", client=client
    )
    assert isinstance(summary, pd.DataFrame)


def test_minlp(data):
    m, *_ = data

    # Set
    t = Set(
        m,
        name="t",
        records=["period-1", "period-2", "period-3"],
        description="scheduling periods (2hrs)",
    )

    # Data
    load = Parameter(
        m,
        name="load",
        domain=t,
        records=pd.DataFrame([["period-1", 400], ["period-2", 900], ["period-3", 700]]),
        description="system load",
    )
    initlev = Parameter(
        m,
        name="initlev",
        domain=t,
        records=pd.DataFrame([["period-1", 3000]]),
        description="initial level of the oil storage tank",
    )

    # Variable
    status = Variable(
        m,
        name="status",
        domain=t,
        type="Binary",
        description="on or off status of the oil based generating unit",
    )
    poil = Variable(
        m,
        name="poil",
        domain=t,
        description="generation level of oil based unit",
    )
    others = Variable(m, name="others", domain=t, description="other generation")
    oil = Variable(
        m, name="oil", domain=t, type="Positive", description="oil consumption"
    )
    volume = Variable(
        m,
        name="volume",
        domain=t,
        type="Positive",
        description="the volume of oil in the storage tank",
    )

    volume.up[t] = 4000
    volume.lo[t].where[Ord(t) == Card(t)] = 2000

    others.lo[t] = 50
    others.up[t] = 700

    # Equation
    lowoil = Equation(
        m,
        name="lowoil",
        domain=t,
        description="lower limit on oil generating unit",
    )
    maxoil = Equation(
        m,
        name="maxoil",
        domain=t,
        description="upper limit on oil generating unit",
    )
    floweq = Equation(
        m,
        name="floweq",
        domain=t,
        description="the oil flow balance in the storage tank",
    )
    demcons = Equation(
        m,
        name="demcons",
        domain=t,
        description="total generation must meet the load",
    )
    oileq = Equation(
        m, name="oileq", domain=t, description="calculation of oil consumption"
    )

    cost = Sum(t, 300 + 6 * others[t] + 0.0025 * (others[t] ** 2))
    lowoil[t] = poil[t] >= 100 * status[t]
    maxoil[t] = poil[t] <= 500 * status[t]
    floweq[t] = volume[t] == volume[t.lag(1)] + 500 - oil[t] + initlev[t]
    oileq[t] = oil[t] == 50 * status[t] + poil[t] + 0.005 * (poil[t] ** 2)
    demcons[t] = poil[t] + others[t] >= load[t]

    model = Model(
        m,
        name="ucom",
        equations=m.getEquations(),
        problem="MINLP",
        sense=Sense.MIN,
        objective=cost,
    )
    poil.l[t] = 100

    client = NeosClient(email=os.environ["NEOS_EMAIL"])
    summary = model.solve(
        solver="baron", output=sys.stdout, backend="neos", client=client
    )
    assert isinstance(summary, pd.DataFrame)


def test_qcp(data):
    m, *_ = data

    # Set
    i = Set(m, name="i", records=[str(idx) for idx in range(181)])
    omega_stop = Set(
        m,
        name="omega_stop",
        domain=i,
        records=[str(idx) for idx in range(120, 181)],
    )
    omega_pass = Set(
        m,
        name="omega_pass",
        domain=i,
        records=[str(idx) for idx in range(91)],
    )
    k = Set(m, name="k", records=[str(idx) for idx in range(11)])

    # Parameter
    beta = Parameter(m, name="beta", records=0.01)
    step = Parameter(m, name="step", records=math.pi / 180)
    n = Parameter(m, name="n", records=20)
    omega = Parameter(m, name="omega", domain=i)
    omega[i] = (Ord(i) - 1) * step

    # Variable
    h = Variable(m, name="h", domain=k)
    t = Variable(m, name="t")
    v2 = Variable(m, name="v2", description="for conic variable u - t")
    v3 = Variable(m, name="v3", type="Positive", description="for conic variable u + t")
    u = Variable(m, name="u", type="Positive")
    v = Variable(m, name="v", type="Positive")

    # Equation
    passband_up_bnds = Equation(m, name="passband_up_bnds", domain=i)
    cone_lhs = Equation(m, name="cone_lhs")
    cone_rhs = Equation(m, name="cone_rhs")
    so = Equation(m, name="so")
    passband_lo_bnds = Equation(m, name="passband_lo_bnds", domain=i)
    stopband_bnds = Equation(m, name="stopband_bnds", domain=i)
    stopband_bnds2 = Equation(m, name="stopband_bnds2", domain=i)

    passband_up_bnds[i].where[omega_pass[i]] = (
        2
        * Sum(
            k.where[Ord(k) < Card(k)],
            h[k] * gams_math.cos((Ord(k) - 1 - (n - 1) / 2) * omega[i]),
        )
        <= t
    )

    cone_rhs[...] = v2 == u - t
    cone_lhs[...] = v3 == u + t
    so[...] = v3**2 >= v**2 + v2**2

    passband_lo_bnds[i].where[omega_pass[i]] = u <= 2 * Sum(
        k.where[Ord(k) < Card(k)],
        h[k] * gams_math.cos((Ord(k) - 1 - (n - 1) / 2) * omega[i]),
    )

    stopband_bnds[i].where[omega_stop[i]] = -beta <= 2 * Sum(
        k.where[Ord(k) < Card(k)],
        h[k] * gams_math.cos((Ord(k) - 1 - (n - 1) / 2) * omega[i]),
    )

    stopband_bnds2[i].where[omega_stop[i]] = (
        2
        * Sum(
            k.where[Ord(k) < Card(k)],
            h[k] * gams_math.cos((Ord(k) - 1 - (n - 1) / 2) * omega[i]),
        )
        <= beta
    )

    t.lo[...] = 1
    v.fx[...] = 2

    fir_socp = Model(
        m,
        name="fir_socp",
        equations=m.getEquations(),
        problem=Problem.QCP,
        sense=Sense.MIN,
        objective=t,
    )

    client = NeosClient(email=os.environ["NEOS_EMAIL"])
    summary = fir_socp.solve(
        solver="scip", output=sys.stdout, backend="neos", client=client
    )
    assert isinstance(summary, pd.DataFrame)
