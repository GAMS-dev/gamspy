from __future__ import annotations

import concurrent.futures
import logging
import math
import os
import platform
import shutil
import sys
import time

import numpy as np
import pytest

import gamspy._validation as validation
import gamspy.math as gamspy_math
from gamspy import (
    Alias,
    Container,
    Equation,
    Model,
    ModelStatus,
    Options,
    Parameter,
    Problem,
    Sense,
    Set,
    SolveStatus,
    Sum,
    Variable,
    VariableType,
)
from gamspy.exceptions import GamspyException, ValidationError

pytestmark = pytest.mark.integration


@pytest.fixture
def data():
    # Arrange
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

    # Act and assert
    yield m, canning_plants, markets, capacities, demands, distances

    # Cleanup
    m.close()
    shutil.rmtree("tmp")


def transport(f_value):
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
    i = Set(
        m,
        name="i",
        records=["seattle", "san-diego"],
        description="canning plants",
    )
    j = Set(
        m,
        name="j",
        records=["new-york", "chicago", "topeka"],
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
    f = Parameter(m, name="f", records=f_value)
    c[i, j] = f * d[i, j] / 1000

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
        m, name="demand", domain=j, description="satisfy demand at market j"
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
    transport.solve()

    m.close()

    return transport.objective_value


def transport2(f_value):
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
    i = Set(
        m,
        name="i",
        records=["seattle", "san-diego"],
        description="canning plants",
    )
    j = Set(
        m,
        name="j",
        records=["new-york", "chicago", "topeka"],
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
    f = Parameter(m, name="f", records=f_value)
    c[i, j] = f * d[i, j] / 1000

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
        m, name="demand", domain=j, description="satisfy demand at market j"
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
    m.addGamsCode("scalar bla; bla = sleep(180);")
    transport.solve()

    m.close()

    return transport.objective_value


def test_uel_order(data):
    m, *_ = data
    i = Set(m, "i")
    p = m.addParameter("base", [i])
    d = Parameter(m, "d")
    i.setRecords(["i1", "i2"])
    d[...] = 0
    i.setRecords(["i0", "i1"])
    p[i] = i.ord
    assert p.records.values.tolist() == [["i1", 1.0], ["i0", 2.0]]


def test_records(data):
    m, canning_plants, markets, capacities, demands, distances = data
    i = Set(m, name="i", records=canning_plants)
    j = Set(m, name="j", records=markets)
    k = Set(m, name="k", records=["seattle", "san-diego", "california"])
    k["seattle"] = False
    assert k.records.loc[0, :].values.tolist() == ["san-diego", ""]

    a = Parameter(m, name="a", domain=[i], records=capacities)
    b = Parameter(m, name="b", domain=[j], records=demands)
    d = Parameter(m, name="d", domain=[i, j], records=distances)
    c = Parameter(m, name="c", domain=[i, j])
    assert c.records is None
    e = Parameter(m, name="e")

    c[i, j] = 90 * d[i, j] / 1000
    assert c.records.values.tolist() == [
        ["seattle", "new-york", 0.225],
        ["seattle", "chicago", 0.153],
        ["seattle", "topeka", 0.162],
        ["san-diego", "new-york", 0.225],
        ["san-diego", "chicago", 0.162],
        ["san-diego", "topeka", 0.126],
    ]
    assert c[i, j].records.values.tolist() == [
        ["seattle", "new-york", 0.225],
        ["seattle", "chicago", 0.153],
        ["seattle", "topeka", 0.162],
        ["san-diego", "new-york", 0.225],
        ["san-diego", "chicago", 0.162],
        ["san-diego", "topeka", 0.126],
    ]
    assert c[...].records.values.tolist() == [
        ["seattle", "new-york", 0.225],
        ["seattle", "chicago", 0.153],
        ["seattle", "topeka", 0.162],
        ["san-diego", "new-york", 0.225],
        ["san-diego", "chicago", 0.162],
        ["san-diego", "topeka", 0.126],
    ]
    assert c[:, :].records.values.tolist() == [
        ["seattle", "new-york", 0.225],
        ["seattle", "chicago", 0.153],
        ["seattle", "topeka", 0.162],
        ["san-diego", "new-york", 0.225],
        ["san-diego", "chicago", 0.162],
        ["san-diego", "topeka", 0.126],
    ]
    assert c[i, "new-york"].records.values.tolist() == [
        ["seattle", "new-york", 0.225],
        ["san-diego", "new-york", 0.225],
    ]
    assert c["san-diego", j].records.values.tolist() == [
        ["san-diego", "new-york", 0.225],
        ["san-diego", "chicago", 0.162],
        ["san-diego", "topeka", 0.126],
    ]
    assert c["san-diego", "new-york"].records == 0.225
    e[...] = 5
    assert e.records.values.tolist() == [[5.0]]

    with pytest.raises(TypeError):
        e.records = 5

    x = Variable(m, name="x", domain=[i, j], type="Positive")
    assert x.records is None
    assert x.l[i, j].records is None
    assert x.l[i, "new-york"].records is None
    assert x.l["san-diego", j].records is None
    assert x.l["san-diego", "new-york"].records is None
    z = Variable(m, name="z")

    cost = Equation(m, name="cost")
    supply = Equation(m, name="supply", domain=[i])
    assert supply.records is None
    assert supply.l[i].records is None
    assert supply.l["seattle"].records is None

    demand = Equation(m, name="demand", domain=[j])

    cost[...] = Sum((i, j), c[i, j] * x[i, j]) == z
    supply[i] = Sum(j, x[i, j]) <= a[i]
    demand[j] = Sum(i, x[i, j]) >= b[j]

    transport = Model(
        m,
        name="transport",
        equations=[cost, supply, demand],
        problem="LP",
        sense="min",
        objective=z,
    )
    transport.solve()

    # Test the columns of a set
    assert i.records.columns.tolist() == ["uni", "element_text"]

    # Test the columns of a parameter
    assert a.records.columns.tolist() == ["i", "value"]

    # Test the columns of scalar variable
    assert z.records.columns.tolist() == [
        "level",
        "marginal",
        "lower",
        "upper",
        "scale",
    ]

    # Test the columns of indexed variable
    assert x.records.columns.tolist() == [
        "i",
        "j",
        "level",
        "marginal",
        "lower",
        "upper",
        "scale",
    ]

    # Test the columns of the attribute records
    assert x.l[i, j].records.columns.tolist() == ["i", "j", "level"]
    assert x.m[i, j].records.columns.tolist() == ["i", "j", "marginal"]
    assert x.up[i, j].records.columns.tolist() == ["i", "j", "upper"]
    assert x.lo[i, j].records.columns.tolist() == ["i", "j", "lower"]
    assert x.scale[i, j].records.columns.tolist() == ["i", "j", "scale"]

    # Test the records of the filtered attribute records
    assert x.l.records.values.tolist() == [
        ["seattle", "new-york", 50.0],
        ["seattle", "chicago", 300.0],
        ["seattle", "topeka", 0.0],
        ["san-diego", "new-york", 275.0],
        ["san-diego", "chicago", 0.0],
        ["san-diego", "topeka", 275.0],
    ]
    assert x.l[i, j].records.values.tolist() == [
        ["seattle", "new-york", 50.0],
        ["seattle", "chicago", 300.0],
        ["seattle", "topeka", 0.0],
        ["san-diego", "new-york", 275.0],
        ["san-diego", "chicago", 0.0],
        ["san-diego", "topeka", 275.0],
    ]
    assert x.l[...].records.values.tolist() == [
        ["seattle", "new-york", 50.0],
        ["seattle", "chicago", 300.0],
        ["seattle", "topeka", 0.0],
        ["san-diego", "new-york", 275.0],
        ["san-diego", "chicago", 0.0],
        ["san-diego", "topeka", 275.0],
    ]
    assert x.l[i, "new-york"].records.values.tolist() == [
        ["seattle", "new-york", 50.0],
        ["san-diego", "new-york", 275.0],
    ]
    assert x.l["san-diego", j].records.values.tolist() == [
        ["san-diego", "new-york", 275.0],
        ["san-diego", "chicago", 0.0],
        ["san-diego", "topeka", 275.0],
    ]
    assert x.l["san-diego", "new-york"].records == 275.0
    assert z.l.records == 153.675

    # Test the columns of equation
    assert cost.records.columns.tolist() == [
        "level",
        "marginal",
        "lower",
        "upper",
        "scale",
    ]

    # Test the columns of indexed equation
    assert supply.records.columns.tolist() == [
        "i",
        "level",
        "marginal",
        "lower",
        "upper",
        "scale",
    ]

    # Test the columns of the attribute records
    assert supply.l[i].records.columns.tolist() == ["i", "level"]
    assert supply.m[i].records.columns.tolist() == ["i", "marginal"]
    assert supply.up[i].records.columns.tolist() == ["i", "upper"]
    assert supply.lo[i].records.columns.tolist() == ["i", "lower"]
    assert supply.scale[i].records.columns.tolist() == ["i", "scale"]

    # Test the records of the filtered attribute records
    assert supply.l[i].records.values.tolist() == [
        ["seattle", 350.0],
        ["san-diego", 550.0],
    ]
    assert supply.l["seattle"].records == 350.0

    m = Container()
    i1 = Set(m, name="i1", records=range(2))
    i2 = Set(m, name="i2", records=range(2))
    i3 = Set(m, name="i3", records=range(2))
    i4 = Set(m, name="i4", records=range(2))
    v1 = Variable(m, "v1", domain=[i1, i2, i3, i4])
    v1.generateRecords(seed=1)
    assert v1.l[i1, i2, i3, i4].records.values.tolist() == [
        ["0", "0", "0", "0", 0.5118216247002567],
        ["0", "0", "0", "1", 0.9504636963259353],
        ["0", "0", "1", "0", 0.14415961271963373],
        ["0", "0", "1", "1", 0.9486494471372439],
        ["0", "1", "0", "0", 0.31183145201048545],
        ["0", "1", "0", "1", 0.42332644897257565],
        ["0", "1", "1", "0", 0.8277025938204418],
        ["0", "1", "1", "1", 0.4091991363691613],
        ["1", "0", "0", "0", 0.5495936876730595],
        ["1", "0", "0", "1", 0.027559113243068367],
        ["1", "0", "1", "0", 0.7535131086748066],
        ["1", "0", "1", "1", 0.5381433132192782],
        ["1", "1", "0", "0", 0.32973171649909216],
        ["1", "1", "0", "1", 0.7884287034284043],
        ["1", "1", "1", "0", 0.303194829291645],
        ["1", "1", "1", "1", 0.4534978894806515],
    ]

    assert v1.l[i1, :, i3, i4].records.values.tolist() == [
        ["0", "0", "0", "0", 0.5118216247002567],
        ["0", "0", "0", "1", 0.9504636963259353],
        ["0", "0", "1", "0", 0.14415961271963373],
        ["0", "0", "1", "1", 0.9486494471372439],
        ["0", "1", "0", "0", 0.31183145201048545],
        ["0", "1", "0", "1", 0.42332644897257565],
        ["0", "1", "1", "0", 0.8277025938204418],
        ["0", "1", "1", "1", 0.4091991363691613],
        ["1", "0", "0", "0", 0.5495936876730595],
        ["1", "0", "0", "1", 0.027559113243068367],
        ["1", "0", "1", "0", 0.7535131086748066],
        ["1", "0", "1", "1", 0.5381433132192782],
        ["1", "1", "0", "0", 0.32973171649909216],
        ["1", "1", "0", "1", 0.7884287034284043],
        ["1", "1", "1", "0", 0.303194829291645],
        ["1", "1", "1", "1", 0.4534978894806515],
    ]

    assert v1.l[i1, ..., i4].records.values.tolist() == [
        ["0", "0", "0", "0", 0.5118216247002567],
        ["0", "0", "0", "1", 0.9504636963259353],
        ["0", "0", "1", "0", 0.14415961271963373],
        ["0", "0", "1", "1", 0.9486494471372439],
        ["0", "1", "0", "0", 0.31183145201048545],
        ["0", "1", "0", "1", 0.42332644897257565],
        ["0", "1", "1", "0", 0.8277025938204418],
        ["0", "1", "1", "1", 0.4091991363691613],
        ["1", "0", "0", "0", 0.5495936876730595],
        ["1", "0", "0", "1", 0.027559113243068367],
        ["1", "0", "1", "0", 0.7535131086748066],
        ["1", "0", "1", "1", 0.5381433132192782],
        ["1", "1", "0", "0", 0.32973171649909216],
        ["1", "1", "0", "1", 0.7884287034284043],
        ["1", "1", "1", "0", 0.303194829291645],
        ["1", "1", "1", "1", 0.4534978894806515],
    ]

    assert v1.l["0", ..., "1"].records.values.tolist() == [
        ["0", "0", "0", "1", 0.9504636963259353],
        ["0", "0", "1", "1", 0.9486494471372439],
        ["0", "1", "0", "1", 0.42332644897257565],
        ["0", "1", "1", "1", 0.4091991363691613],
    ]

    i = Set(m, "i", records=[f"i{i}" for i in range(2)])
    j = Set(m, "j", records=[f"j{i}" for i in range(2)])
    k = Set(m, "k", records=[f"k{i}" for i in range(2)])
    l = Set(m, "l", records=[f"l{i}" for i in range(2)])
    a = Set(m, domain=[i, j, k, l])
    a.generateRecords()
    assert a.records.values.tolist() == [
        ["i0", "j0", "k0", "l0", ""],
        ["i0", "j0", "k0", "l1", ""],
        ["i0", "j0", "k1", "l0", ""],
        ["i0", "j0", "k1", "l1", ""],
        ["i0", "j1", "k0", "l0", ""],
        ["i0", "j1", "k0", "l1", ""],
        ["i0", "j1", "k1", "l0", ""],
        ["i0", "j1", "k1", "l1", ""],
        ["i1", "j0", "k0", "l0", ""],
        ["i1", "j0", "k0", "l1", ""],
        ["i1", "j0", "k1", "l0", ""],
        ["i1", "j0", "k1", "l1", ""],
        ["i1", "j1", "k0", "l0", ""],
        ["i1", "j1", "k0", "l1", ""],
        ["i1", "j1", "k1", "l0", ""],
        ["i1", "j1", "k1", "l1", ""],
    ]
    assert a["i0", ...].records.values.tolist() == [
        ["i0", "j0", "k0", "l0", ""],
        ["i0", "j0", "k0", "l1", ""],
        ["i0", "j0", "k1", "l0", ""],
        ["i0", "j0", "k1", "l1", ""],
        ["i0", "j1", "k0", "l0", ""],
        ["i0", "j1", "k0", "l1", ""],
        ["i0", "j1", "k1", "l0", ""],
        ["i0", "j1", "k1", "l1", ""],
    ]
    assert a["i0", :, "k1", "l0"].records.values.tolist() == [
        ["i0", "j0", "k1", "l0", ""],
        ["i0", "j1", "k1", "l0", ""],
    ]


def test_after_first_solve(data):
    m, canning_plants, markets, capacities, demands, distances = data
    i = Set(m, name="i", records=canning_plants)
    j = Set(m, name="j", records=markets)

    a = Parameter(m, name="a", domain=[i], records=capacities)
    b = Parameter(m, name="b", domain=[j], records=demands)
    d = Parameter(m, name="d", domain=[i, j], records=distances)
    c = Parameter(m, name="c", domain=[i, j])
    c[i, j] = 90 * d[i, j] / 1000

    x = Variable(m, name="x", domain=[i, j], type="Positive")
    z = Variable(m, name="z")
    z2 = Variable(m, name="z2")

    cost = Equation(m, name="cost")
    cost2 = Equation(m, name="cost2")
    supply = Equation(m, name="supply", domain=[i])
    demand = Equation(m, name="demand", domain=[j])

    cost[...] = Sum((i, j), c[i, j] * x[i, j]) == z
    supply[i] = Sum(j, x[i, j]) <= a[i]
    demand[j] = Sum(i, x[i, j]) >= b[j]
    cost2[...] = Sum((i, j), c[i, j] * x[i, j]) * 5 == z2

    transport = Model(
        m,
        name="transport",
        equations=[cost, supply, demand],
        problem="LP",
        sense="min",
        objective=z,
    )
    transport.solve()

    assert z.records is not None
    assert x.records is not None
    assert cost.records is not None
    assert supply.records is not None
    assert demand.records is not None

    transport2 = Model(
        m,
        name="transport2",
        equations=[cost2, supply, demand],
        problem="LP",
        sense="min",
        objective=z2,
    )
    transport2.solve()
    second_z2_value = z2.toValue()
    assert math.isclose(second_z2_value, 768.375, rel_tol=1e-3)


def test_solve(data):
    m, canning_plants, markets, capacities, demands, distances = data
    i = Set(m, name="i", records=canning_plants)
    j = Set(m, name="j", records=markets)

    a = Parameter(m, name="a", domain=[i], records=capacities)
    b = Parameter(m, name="b", domain=[j], records=demands)
    d = Parameter(m, name="d", domain=[i, j], records=distances)
    c = Parameter(m, name="c", domain=[i, j])
    c[i, j] = 90 * d[i, j] / 1000

    x = Variable(m, name="x", domain=[i, j], type="Positive")
    z = Variable(m, name="z")

    cost = Equation(m, name="cost")
    supply = Equation(m, name="supply", domain=[i])
    demand = Equation(m, name="demand", domain=[j])

    cost[...] = Sum((i, j), c[i, j] * x[i, j]) == z
    supply[i] = Sum(j, x[i, j]) <= a[i]
    demand[j] = Sum(i, x[i, j]) >= b[j]

    transport = Model(
        m,
        name="transport",
        equations=[cost, supply, demand],
        problem="LP",
        sense="min",
        objective=z,
    )

    freeLinks = Set(
        m, "freeLinks", domain=[i, j], records=[("seattle", "chicago")]
    )
    # Test limited variables
    transport2 = Model(
        m,
        name="transport2",
        equations=[cost, supply, demand],
        problem="LP",
        sense="min",
        objective=z,
        limited_variables=[x[freeLinks]],
    )

    assert (
        transport2.getDeclaration()
        == "Model transport2 / cost,supply,demand,x(freeLinks) /;"
    )

    # Test output redirection
    redirection_path = os.path.join(os.getcwd(), "tmp", "bla.gms")
    with open(redirection_path, "w") as file:
        _ = transport.solve(
            options=Options(time_limit=100),
            output=file,
        )

    # Redirect to an invalid stream
    class Dummy:
        def write(self, data): ...

    with pytest.raises(ValidationError):
        transport.solve(output=Dummy())

    # Redirect output to logger
    logger = logging.getLogger("TEST_LOGGER")
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(name)s - %(levelname)s] %(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    class CustomOutput:
        def write(self, data):
            logger.info(data.strip())

        def flush(self): ...

    custom_output = CustomOutput()

    _ = transport.solve(
        options=Options(time_limit=100),
        output=custom_output,
    )

    assert os.path.exists(redirection_path)
    assert transport.status == ModelStatus.OptimalGlobal
    assert transport.solve_status == SolveStatus.NormalCompletion

    pytest.raises(
        ValidationError,
        transport.solve,
        None,
        None,
        None,
        None,
        None,
        "bla",
    )

    from gamspy._model import ATTRIBUTE_MAP

    for attr_name in ATTRIBUTE_MAP.values():
        assert hasattr(transport, attr_name)

        # Make sure model attributes are not in the container
        assert attr_name not in m.data

    # Make sure dummy variable and equation is not in the container
    assert not any("dummy_" in name for name in m.data)

    # Test invalid problem
    pytest.raises(ValueError, Model, m, "dummy", "bla", [cost])

    # Test invalid sense
    pytest.raises(ValueError, Model, m, "dummy", "LP", [cost], "bla")

    # Test invalid objective variable
    pytest.raises(TypeError, Model, m, "dummy", "LP", [cost], "min", a)

    # Test invalid commandline options
    pytest.raises(
        TypeError,
        transport.solve,
        None,
        {"bla": 100},
    )

    pytest.raises(TypeError, transport.solve, None, 5)

    # Try to solve invalid model
    m = Container()
    cost = Equation(m, "cost")
    model = Model(m, "dummy", equations=[cost], problem="LP", sense="min")
    pytest.raises(Exception, model.solve)


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="It doesn't work in Docker Windows Server container.",
)
def test_interrupt():
    m = Container()

    f = Set(
        m,
        name="f",
        description="faces on a dice",
        records=[f"face{idx}" for idx in range(1, 7)],
    )
    dice = Set(
        m,
        name="dice",
        description="number of dice",
        records=[f"dice{idx}" for idx in range(1, 7)],
    )

    flo = Parameter(m, name="flo", description="lowest face value", records=1)
    fup = Parameter(
        m, "fup", description="highest face value", records=len(dice) * len(f)
    )

    fp = Alias(m, name="fp", alias_with=f)

    wnx = Variable(m, name="wnx", description="number of wins")
    fval = Variable(
        m,
        name="fval",
        domain=[dice, f],
        description="face value on dice - may be fractional",
    )
    comp = Variable(
        m,
        name="comp",
        domain=[dice, f, fp],
        description="one implies f beats fp",
        type=VariableType.BINARY,
    )

    fval.lo[dice, f] = flo
    fval.up[dice, f] = fup
    fval.fx["dice1", "face1"] = flo

    eq1 = Equation(m, "eq1", domain=dice, description="count the wins")
    eq3 = Equation(
        m,
        "eq3",
        domain=[dice, f, fp],
        description="definition of non-transitive relation",
    )
    eq4 = Equation(
        m,
        "eq4",
        domain=[dice, f],
        description="different face values for a single dice",
    )

    eq1[dice] = Sum((f, fp), comp[dice, f, fp]) == wnx
    eq3[dice, f, fp] = (
        fval[dice, f] + (fup - flo + 1) * (1 - comp[dice, f, fp])
        >= fval[dice.lead(1, type="circular"), fp] + 1
    )
    eq4[dice, f.lag(1)] = fval[dice, f.lag(1)] + 1 <= fval[dice, f]

    xdice = Model(
        m,
        "xdice",
        equations=m.getEquations(),
        problem=Problem.MIP,
        sense=Sense.MAX,
        objective=wnx,
    )

    def interrupt_gams(model):
        time.sleep(4)
        model.interrupt()

    import threading

    thread = threading.Thread(target=interrupt_gams, args=(xdice,))
    thread.start()

    xdice.solve(output=sys.stdout, options=Options(time_limit=60))
    assert xdice.objective_value is not None
    assert xdice.solve_status == SolveStatus.UserInterrupt
    thread.join()

    after_interrupt = Set(m, records=range(3))
    assert after_interrupt.toList() == ["0", "1", "2"]

    summary = xdice.solve(output=sys.stdout, options=Options(time_limit=2))
    assert summary is not None
    assert xdice.solve_status == SolveStatus.ResourceInterrupt


def test_solver_options(data):
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

    # Test solver change
    transport.solve(solver="conopt", solver_options={"rtmaxv": "1.e12"})

    assert os.path.exists(f"{m.working_directory}{os.sep}conopt.opt")


def test_ellipsis(data):
    m, canning_plants, markets, capacities, demands, distances = data
    i = Set(m, name="i", records=canning_plants)
    j = Set(m, name="j", records=markets)

    a = Parameter(m, name="a", domain=[i], records=capacities)
    b = Parameter(m, name="b", domain=[j], records=demands)
    d = Parameter(m, name="d", domain=[i, j], records=distances)
    c = Parameter(m, name="c", domain=[i, j])
    c[...] = 90 * d[...] / 1000
    f = Parameter(m)

    with pytest.raises(ValidationError):
        f[i, j] = 5

    with pytest.raises(ValidationError):
        f[i] = 5

    x = Variable(m, name="x", domain=[i, j], type="Positive")

    supply = Equation(m, name="supply", domain=[i])
    demand = Equation(m, name="demand", domain=[j])

    supply[...] = Sum(j, x[...]) <= a[...]
    demand[...] = Sum(i, x[...]) >= b[...]

    with pytest.raises(ValueError):
        transport = Model(m, name="")

    transport = Model(
        m,
        name="transport",
        equations=m.getEquations(),
        problem="LP",
        sense=Sense.MIN,
        objective=Sum((i, j), c[i, j] * x[i, j]),
    )
    transport.solve()
    assert transport.objective_value == 153.675

    supply.l[...] = 5
    assert supply.records.level.to_list() == [5.0, 5.0]

    domain = validation._transform_given_indices(["a", "b", "c"], ["a", ...])
    assert domain == ["a", "b", "c"]

    domain = validation._transform_given_indices(
        ["a", "b", "c"], ["a", ..., "c"]
    )
    assert domain == ["a", "b", "c"]

    domain = validation._transform_given_indices(["a", "b", "c"], [..., "c"])
    assert domain == ["a", "b", "c"]

    with pytest.raises(ValidationError):
        c[..., ...] = 5


def test_slice(data):
    m, canning_plants, markets, capacities, _, distances = data
    i = Set(m, name="i", records=canning_plants)
    i2 = Set(m, name="i2", records=canning_plants)
    i3 = Set(m, name="i3", records=canning_plants)
    i4 = Set(m, name="i4", records=canning_plants)
    j = Set(m, name="j", records=markets)
    k = Set(m, "k", domain=[i, i2, i3, i4])
    assert k[..., i4].gamsRepr() == "k(i,i2,i3,i4)"
    assert k[i, ..., i4].gamsRepr() == "k(i,i2,i3,i4)"
    assert k[i, ...].gamsRepr() == "k(i,i2,i3,i4)"
    assert k[..., i3, :].gamsRepr() == "k(i,i2,i3,i4)"

    a = Parameter(m, name="a", domain=[i], records=capacities)
    assert a[:].gamsRepr() == "a(i)"
    d = Parameter(m, name="d", domain=[i, j], records=distances)
    c = Parameter(m, name="c", domain=[i, j])
    assert c[:, :].gamsRepr() == "c(i,j)"
    c[:, :] = 90 * d[:, :] / 1000

    x = Variable(m, name="x", domain=[i, j], type="Positive")
    assert x[:, :].gamsRepr() == "x(i,j)"

    supply = Equation(m, name="supply", domain=[i])
    assert supply[:].gamsRepr() == "supply(i)"

    supply.l[:] = 5
    assert supply.l[:].gamsRepr() == "supply.l(i)"

    date = Set(m, "date", description="trading date")
    ntd = Set(m, "ntd", domain=[date], description="none-training days")
    error = Parameter(m, "error", domain=[date], description="Absolute error")
    error_test = Parameter(
        m,
        "error_test",
        description="Absolute error in entire testing phase",
        is_miro_output=True,
    )
    error_test[:] = Sum(ntd, error[ntd])


def test_max_line_length(data):
    m, canning_plants, markets, capacities, demands, distances = data
    i = Set(m, name="i", records=canning_plants)
    j = Set(m, name="j", records=markets)

    a = Parameter(m, name="a", domain=[i], records=capacities)
    b = Parameter(m, name="b", domain=[j], records=demands)
    d = Parameter(m, name="d", domain=[i, j], records=distances)
    c = Parameter(m, name="c", domain=[i, j])
    f = Parameter(
        m,
        "fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        records=1,
    )
    c[i, j] = 90 * d[i, j] / 1000

    x = Variable(m, name="x", domain=[i, j], type="Positive")

    supply = Equation(m, name="supply", domain=[i])
    demand = Equation(m, name="demand", domain=[j])

    # This generates an equation with length > 80000
    long_expr = f
    for _ in range(1200):
        long_expr *= f
    supply[i] = Sum(j, x[i, j]) * long_expr <= a[i]
    demand[j] = Sum(i, x[i, j]) >= b[j]

    transport = Model(
        m,
        name="transport",
        equations=m.getEquations(),
        problem="LP",
        sense=Sense.MIN,
        objective=Sum((i, j), c[i, j] * x[i, j]),
    )
    transport.solve()


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
    assert summary["Solver Status"].tolist()[0] == "Normal"


def test_validation(data):
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

    with pytest.raises(ValidationError):
        supply[j] = Sum(j, x[i, j]) <= a[i]

    with pytest.raises(ValidationError):
        demand[i, j] = Sum(i, x[i, j]) >= b[j]

    with pytest.raises(TypeError):
        c[b[j]] = 90 * d[i, j] / 1000


def test_validation_2():
    m = Container()
    c = Set(m, "c")
    i = Set(m, "i")
    s = Set(m, "s")
    key = Set(m, "key", domain=[i, s, c])
    cost = Parameter(m, "cost", domain=i)
    a1 = Parameter(m, "a1", domain=i)
    a2 = Parameter(m, "a2", domain=[i, s])
    a3 = Parameter(m, "a3", domain=[i, c])
    cobj = Variable(m, "cobj")
    y = Variable(m, "y", domain=[i])
    fin = Variable(m, "fin", domain=[i])
    xin = Variable(m, "xin", domain=[i, c])
    rec = Variable(m, "rec", domain=[i, s, c])
    obj = Equation(m, "obj")

    obj[...] = cobj == Sum(
        i,
        (
            (cost[i] * y[i])
            + (
                (
                    (a1[i] + Sum(key[i, s, c], (a2[i, s] * rec[key])))
                    + Sum(c, (a3[i, c] * xin[i, c]))
                )
                * fin[i]
            )
        ),
    )


def test_validation_3():
    m = Container()

    i = Set(m, "i")
    j = Set(m, "j")
    v = Set(m, "v")
    k = Set(m, "k")
    z = Variable(m, "z", domain=[v, k])
    vk = Set(m, "vk", domain=[v, k])
    n = Parameter(m, domain=k)
    a = Variable(m, "a", domain=i)

    z.up[vk[v, k]] = n[k]

    with pytest.raises(ValidationError):
        a.up[i] = Sum(vk, Sum(j, n[k]))

    a.up[i] = Sum(vk[v, k], Sum(j, n[k]))


def test_after_exception(data):
    m, *_ = data
    x = Variable(m, "x", type="positive")
    e = Equation(m, "e", definition=x <= x + 1)
    with pytest.raises(ValidationError):
        t = Model(
            m,
            name="t",
            equations=[e],
            problem="LP",
            sense=Sense.MIN,
            objective=x,
        )
    x.type = "free"
    t = Model(
        m,
        name="t",
        equations=[e],
        problem="LP",
        sense=Sense.MIN,
        objective=x,
    )

    x.type = "positive"
    x.lo[...] = 0
    try:
        # This must fail because `Objective variable is not a free variable`
        t.solve()
    except GamspyException:
        pass

    f = Parameter(m, "f")
    f[...] = 5

    assert f.getAssignment() == "f = 5;"


def test_invalid_arguments(data):
    m, canning_plants, markets, capacities, demands, distances = data
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
        m, name="demand", domain=j, description="satisfy demand at market j"
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

    with pytest.raises(TypeError):
        transport.solve(solver=sys.stdout)

    with pytest.raises(ValidationError):
        transport.solve(solver="sadwqeq")

    # solver is not installed
    with pytest.raises(ValidationError):
        transport.solve(solver="SNOPT")

    # solver is not capable of solving this problem type
    with pytest.raises(ValidationError):
        transport.solve(solver="PATH")

    # we do not accept dict anymore
    with pytest.raises(TypeError):
        transport.solve(options={"bla": "bla"})


def test_marking_updated_symbols(data):
    m, *_ = data
    height_data = [
        1.47,
        1.50,
        1.52,
        1.55,
        1.57,
        1.60,
        1.63,
        1.65,
        1.68,
        1.70,
        1.73,
        1.75,
        1.78,
        1.80,
        1.83,
    ]

    weight_data = [
        52.21,
        53.12,
        54.48,
        55.84,
        57.20,
        58.57,
        59.93,
        61.29,
        63.11,
        64.47,
        66.28,
        68.10,
        69.92,
        72.19,
        74.46,
    ]

    num_recs = len(height_data)

    x_recs = np.ones((num_recs, 2))
    x_recs[:, 0] = height_data

    set_15 = Set(m, name="set15", records=range(15))
    set_2 = Set(m, name="set2", records=range(2))

    X = Parameter(
        m, name="X", domain=[set_15, set_2], records=x_recs, uels_on_axes=True
    )

    y_recs = np.array(weight_data)

    y = Parameter(
        m, name="y", domain=[set_15], records=y_recs, uels_on_axes=True
    )

    w = Variable(m, name="w", domain=[set_2])

    loss = Variable(m, name="loss", domain=[])

    loss_eq = Equation(m, name="set_loss", domain=[])

    loss_eq[...] = loss == Sum(
        set_15, (y[set_15] - Sum(set_2, X[set_15, set_2] * w[set_2])) ** 2
    )

    model = Model(
        m,
        name="OLS",
        problem="QCP",
        equations=m.getEquations(),
        sense="MIN",
        objective=loss,
    )

    model.solve()
    assert w.records is not None
    assert loss.records is not None


def test_multiprocessing():
    f_values = [90, 120, 150, 180]
    expected_values = [153.675, 204.89999999999998, 256.125, 307.35]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for expected, objective in zip(
            expected_values, executor.map(transport, f_values)
        ):
            assert math.isclose(expected, objective)


def test_selective_loading(data):
    m, canning_plants, markets, capacities, demands, distances = data
    i = Set(m, name="i", records=canning_plants)
    j = Set(m, name="j", records=markets)
    a = Parameter(m, name="a", domain=[i], records=capacities)
    b = Parameter(m, name="b", domain=[j], records=demands)
    d = Parameter(m, name="d", domain=[i, j], records=distances)
    c = Parameter(m, name="c", domain=[i, j])
    c[i, j] = 90 * d[i, j] / 1000

    x = Variable(m, name="x", domain=[i, j], type="Positive")
    z = Variable(m, name="z")

    cost = Equation(m, name="cost")
    supply = Equation(m, name="supply", domain=[i])
    demand = Equation(m, name="demand", domain=[j])

    cost[...] = Sum((i, j), c[i, j] * x[i, j]) == z
    supply[i] = Sum(j, x[i, j]) <= a[i]
    demand[j] = Sum(i, x[i, j]) >= b[j]

    transport = Model(
        m,
        name="transport",
        equations=[cost, supply, demand],
        problem="LP",
        sense="min",
        objective=z,
    )
    with pytest.raises(ValidationError):
        transport.solve(load_symbols=["x"])

    with pytest.raises(ValidationError):
        transport.solve(load_symbols=x)

    transport.solve(load_symbols=[])

    assert x.records is None
    assert supply.records is None
    assert transport.objective_value == 153.675

    transport.solve(load_symbols=[x])

    assert x.records is not None
    assert supply.records is None
    assert transport.objective_value == 153.675


def test_execution_error(data):
    m, *_ = data
    m = Container()
    x = Variable(m)
    y = Variable(m)
    obj = gamspy_math.sqr(math.pi - x / y)
    model = Model(m, objective=obj, problem="nlp", sense="min")
    try:
        model.solve()  # this will trigger a division by 0 execution error
    except GamspyException:
        y.l = 1

    summary = model.solve()  # this should work
    assert summary is not None
