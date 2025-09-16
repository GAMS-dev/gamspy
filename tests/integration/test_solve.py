from __future__ import annotations

import concurrent.futures
import logging
import math
import multiprocessing
import os
import platform
import shutil
import sys
import tempfile
import time

import numpy as np
import pytest

import gamspy as gp
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

if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method("spawn")


def ReSHOPAnnotation(m, s):
    return m.addGamsCode("EmbeddedCode ReSHOP:\n" + s + "\nendEmbeddedCode")


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


def transport_with_ctx(f_value):
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

    m = Container()
    with m:
        i = Set(
            name="i",
            records=["seattle", "san-diego"],
            description="canning plants",
        )
        j = Set(
            name="j",
            records=["new-york", "chicago", "topeka"],
            description="markets",
        )

        # Data
        a = Parameter(
            name="a",
            domain=i,
            records=capacities,
            description="capacity of plant i in cases",
        )
        b = Parameter(
            name="b",
            domain=j,
            records=demands,
            description="demand at market j in cases",
        )
        d = Parameter(
            name="d",
            domain=[i, j],
            records=distances,
            description="distance in thousands of miles",
        )
        c = Parameter(
            name="c",
            domain=[i, j],
            description="transport cost in thousands of dollars per case",
        )
        f = Parameter(name="f", records=f_value)
        c[i, j] = f * d[i, j] / 1000

        # Variable
        x = Variable(
            name="x",
            domain=[i, j],
            type="Positive",
            description="shipment quantities in cases",
        )

        # Equation
        supply = Equation(
            name="supply",
            domain=i,
            description="observe supply limit at plant i",
        )
        demand = Equation(
            name="demand", domain=j, description="satisfy demand at market j"
        )

        supply[i] = Sum(j, x[i, j]) <= a[i]
        demand[j] = Sum(i, x[i, j]) >= b[j]

        transport = Model(
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
    assert c["san-diego", "new-york"].records["value"].squeeze() == 0.225
    e[...] = 5
    assert e.records.values.tolist() == [[5.0]]

    with pytest.raises(TypeError):
        e.records = 5

    x = Variable(m, name="x", domain=[i, j], type="Positive")
    assert x.records is None
    assert x[i, "new-york"].records is None
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

    # Test the columns of the records
    assert x[i, j].records.columns.tolist() == [
        "i",
        "j",
        "level",
    ]
    assert x.l[i, j].records.columns.tolist() == ["i", "j", "level"]
    assert x.m[i, j].records.columns.tolist() == ["i", "j", "marginal"]
    assert x.up[i, j].records.columns.tolist() == ["i", "j", "upper"]
    assert x.scale[i, j].records.columns.tolist() == ["i", "j", "scale"]

    # Test the records of the filtered records
    assert x[i, "new-york"].records.values.tolist() == [
        ["seattle", "new-york", 50.0],
        ["san-diego", "new-york", 275.0],
    ]
    assert x.l.records.values.tolist() == [
        ["seattle", "new-york", 50.0],
        ["seattle", "chicago", 300.0],
        ["san-diego", "new-york", 275.0],
        ["san-diego", "topeka", 275.0],
    ]
    assert x.l[i, j].records.values.tolist() == [
        ["seattle", "new-york", 50.0],
        ["seattle", "chicago", 300.0],
        ["san-diego", "new-york", 275.0],
        ["san-diego", "topeka", 275.0],
    ]
    assert x.l[...].records.values.tolist() == [
        ["seattle", "new-york", 50.0],
        ["seattle", "chicago", 300.0],
        ["san-diego", "new-york", 275.0],
        ["san-diego", "topeka", 275.0],
    ]
    assert x.l[i, "new-york"].records.values.tolist() == [
        ["seattle", "new-york", 50.0],
        ["san-diego", "new-york", 275.0],
    ]
    assert x.l["san-diego", j].records.values.tolist() == [
        ["san-diego", "new-york", 275.0],
        ["san-diego", "topeka", 275.0],
    ]
    assert x.l["san-diego", "new-york"].records["level"].squeeze() == 275.0
    assert math.isclose(
        x.m["san-diego", "chicago"].records["marginal"].squeeze(), 0.009
    )
    assert z.l.records["level"].squeeze() == 153.675

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

    # Test the columns of the records
    assert supply["seattle"].records.columns.tolist() == [
        "i",
        "level",
    ]
    assert supply.l[i].records.columns.tolist() == ["i", "level"]
    assert supply.up[i].records.columns.tolist() == ["i", "upper"]
    assert supply.lo[i].records.columns.tolist() == ["i", "lower"]
    assert supply.scale[i].records.columns.tolist() == ["i", "scale"]
    assert supply.range[i].records.columns.tolist() == ["i", "range"]
    assert supply.slacklo[i].records.columns.tolist() == ["i", "slacklo"]

    # Test the records of the filtered records
    assert supply.l[i].records.values.tolist() == [
        ["seattle", 350.0],
        ["san-diego", 550.0],
    ]
    assert supply[i].records.values.tolist() == [
        ["seattle", 350.0],
        ["san-diego", 550.0],
    ]
    assert supply["seattle"].records.values.tolist() == [
        ["seattle", 350.0],
    ]
    assert supply.l["seattle"].records["level"].squeeze() == 350.0
    assert supply.m["seattle"].records["marginal"].squeeze() == -0.0
    assert supply.lo["san-diego"].records["lower"].squeeze() == float("-inf")
    assert supply.up["san-diego"].records["upper"].squeeze() == 600.0
    assert supply.scale["san-diego"].records["scale"].squeeze() == 1.0
    assert supply.range["san-diego"].records["range"].squeeze() == float("inf")
    assert supply.slacklo["san-diego"].records["slacklo"].squeeze() == float(
        "inf"
    )
    assert supply.slackup["san-diego"].records["slackup"].squeeze() == 50.0
    assert supply.slack["san-diego"].records["slack"].squeeze() == 50.0
    assert supply.infeas["san-diego"].records is None

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

    e1 = Equation(m, "e1", domain=[i1, i2, i3, i4])
    e1.generateRecords(seed=1)
    e1.lo = 5
    e1.up = 10
    assert e1.range[i1, i2, i3, i4].records.values.tolist() == [
        ["0", "0", "0", "0", 5.0],
        ["0", "0", "0", "1", 5.0],
        ["0", "0", "1", "0", 5.0],
        ["0", "0", "1", "1", 5.0],
        ["0", "1", "0", "0", 5.0],
        ["0", "1", "0", "1", 5.0],
        ["0", "1", "1", "0", 5.0],
        ["0", "1", "1", "1", 5.0],
        ["1", "0", "0", "0", 5.0],
        ["1", "0", "0", "1", 5.0],
        ["1", "0", "1", "0", 5.0],
        ["1", "0", "1", "1", 5.0],
        ["1", "1", "0", "0", 5.0],
        ["1", "1", "0", "1", 5.0],
        ["1", "1", "1", "0", 5.0],
        ["1", "1", "1", "1", 5.0],
    ]
    e1.lo = 0.5
    e1.up = 0.6
    assert e1.slacklo[i1, i2, i3, i4].records.values.tolist() == [
        ["0", "0", "0", "0", 0.011821624700256717],
        ["0", "0", "0", "1", 0.4504636963259353],
        ["0", "0", "1", "1", 0.44864944713724386],
        ["0", "1", "1", "0", 0.32770259382044176],
        ["1", "0", "0", "0", 0.049593687673059494],
        ["1", "0", "1", "0", 0.2535131086748066],
        ["1", "0", "1", "1", 0.03814331321927822],
        ["1", "1", "0", "1", 0.2884287034284043],
    ]
    assert e1.slackup[i1, i2, i3, i4].records.values.tolist() == [
        ["0", "0", "0", "0", 0.08817837529974326],
        ["0", "0", "1", "0", 0.45584038728036624],
        ["0", "1", "0", "0", 0.2881685479895145],
        ["0", "1", "0", "1", 0.17667355102742432],
        ["0", "1", "1", "1", 0.1908008636308387],
        ["1", "0", "0", "0", 0.05040631232694048],
        ["1", "0", "0", "1", 0.5724408867569316],
        ["1", "0", "1", "1", 0.06185668678072176],
        ["1", "1", "0", "0", 0.2702682835009078],
        ["1", "1", "1", "0", 0.296805170708355],
        ["1", "1", "1", "1", 0.14650211051934847],
    ]
    assert e1.slack[i1, i2, i3, i4].records.values.tolist() == [
        ["0", "0", "0", "0", 0.011821624700256717],
        ["1", "0", "0", "0", 0.049593687673059494],
        ["1", "0", "1", "1", 0.03814331321927822],
    ]
    assert e1.infeas[i1, i2, i3, i4].records.values.tolist() == [
        ["0", "0", "0", "1", 0.3504636963259353],
        ["0", "0", "1", "0", 0.35584038728036627],
        ["0", "0", "1", "1", 0.3486494471372439],
        ["0", "1", "0", "0", 0.18816854798951455],
        ["0", "1", "0", "1", 0.07667355102742435],
        ["0", "1", "1", "0", 0.22770259382044178],
        ["0", "1", "1", "1", 0.09080086363083872],
        ["1", "0", "0", "1", 0.47244088675693163],
        ["1", "0", "1", "0", 0.1535131086748066],
        ["1", "1", "0", "0", 0.17026828350090784],
        ["1", "1", "0", "1", 0.18842870342840434],
        ["1", "1", "1", "0", 0.19680517070835502],
        ["1", "1", "1", "1", 0.046502110519348494],
    ]
    assert e1.infeas["0", "1", "0", i4].records.values.tolist() == [
        ["0", "1", "0", "0", 0.18816854798951455],
        ["0", "1", "0", "1", 0.07667355102742435],
    ]
    assert (
        e1.infeas["0", "1", "0", "1"].records["infeas"].squeeze()
        == 0.07667355102742435
    )

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

    g = gp.Parameter(m, "g", domain=["*"], records=[("a", 1), ("b", 2)])
    assert g.records.values.tolist() == [["a", 1.0], ["b", 2.0]]

    m = gp.Container()

    i = gp.Set(m, "i", records=range(5))
    j = gp.Set(m, "j", domain=i, records=range(3))
    k = gp.Set(m, "k", records=range(6, 10))
    l = gp.Set(m, "l", domain=[i, k])
    al = gp.Alias(m, "al", alias_with=j)
    l.generateRecords()
    assert i[k].records is None

    assert i[j].records["uni"].values.tolist() == ["0", "1", "2"]
    assert i[al].records["uni"].values.tolist() == ["0", "1", "2"]

    with pytest.raises(ValidationError):
        _ = al.lead(2).records

    a = gp.Parameter(m, "a", domain=i, records=np.array([1, 2, 3, 4, 5]))
    assert a[j].records.values.tolist() == [["0", 1.0], ["1", 2.0], ["2", 3.0]]

    m = gp.Container()
    i = gp.Set(m, records=range(5))
    j = gp.Set(m, records=range(3))
    ij = gp.Set(m, domain=[i, j])
    ij.generateRecords()
    assert ij[i, j].where[
        (gp.Ord(i) > 2) & (gp.Ord(j) > 2)
    ].records.values.tolist() == [
        ["2", "2", ""],
        ["3", "2", ""],
        ["4", "2", ""],
    ]
    assert gp.Domain(i, j).where[
        (gp.Ord(i) > 2) & (gp.Ord(j) > 2)
    ].records.values.tolist() == [
        ["2", "2", ""],
        ["3", "2", ""],
        ["4", "2", ""],
    ]
    m.close()


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

    with pytest.raises(ValidationError):
        _ = Model(
            m,
            name="transport2",
            equations=[cost, supply, demand],
            problem="LP",
            sense="min",
            objective=z,
            limited_variables=x[freeLinks],
        )

    # Test limited variables
    transport3 = Model(
        m,
        name="transport3",
        equations=[cost, supply, demand],
        problem="LP",
        sense="min",
        objective=z,
        limited_variables=[x[freeLinks]],
    )

    assert (
        transport3.getDeclaration()
        == "Model transport3 / cost,supply,demand,x(freeLinks) /;"
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
    pytest.raises(ValueError, Model, m, "dummy", "", "bla", [cost])

    # Test invalid sense
    pytest.raises(ValueError, Model, m, "dummy", "", "LP", [cost], "bla")

    # Test invalid objective variable
    pytest.raises(TypeError, Model, m, "dummy", "", "LP", [cost], "min", a)

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
    platform.system() != "Linux",
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
    assert os.path.exists(os.path.join(m.working_directory, "conopt4.opt"))

    # Test solver option validation

    ## Baron
    with pytest.raises(ValidationError):
        transport.solve(solver="baron", solver_options={"blabla": "1.e12"})

    ## Cbc
    with pytest.raises(ValidationError):
        transport.solve(solver="cbc", solver_options={"blabla": "1.e12"})

    ## Conopt
    with pytest.raises(ValidationError):
        transport.solve(solver="conopt", solver_options={"blabla": "1.e12"})

    with pytest.raises(ValidationError):
        transport.solve(solver="conopt4", solver_options={"blabla": "1.e12"})

    ## Convert
    with pytest.raises(ValidationError):
        transport.solve(solver="convert", solver_options={"blabla": "1.e12"})

    ## Copt
    with pytest.raises(ValidationError):
        transport.solve(solver="copt", solver_options={"blabla": "1.e12"})

    ## Cplex
    with pytest.raises(ValidationError):
        transport.solve(solver="cplex", solver_options={"blabla": "1.e12"})

    ## Examiner
    with pytest.raises(ValidationError):
        transport.solve(solver="examiner", solver_options={"blabla": "1.e12"})

    ## Examiner2
    with pytest.raises(ValidationError):
        transport.solve(solver="examiner2", solver_options={"blabla": "1.e12"})

    ## gurobi
    with pytest.raises(ValidationError):
        transport.solve(solver="gurobi", solver_options={"blabla": "1.e12"})

    ## Highs will not care whether there is a wrong solver option
    transport.solve(solver="highs", solver_options={"blabla": "1.e12"})

    ## Ipopt
    with pytest.raises(ValidationError):
        transport.solve(solver="ipopt", solver_options={"blabla": "1.e12"})

    ## Kestrel will expect kestrel_solver option but will not find it
    with pytest.raises(GamspyException):
        transport.solve(solver="kestrel", solver_options={"blabla": "1.e12"})

    ## Knitro
    with pytest.raises(ValidationError):
        transport.solve(solver="knitro", solver_options={"blabla": "1.e12"})

    ## Minos
    with pytest.raises(ValidationError):
        transport.solve(solver="minos", solver_options={"blabla": "1.e12"})

    ## Mosek
    with pytest.raises(ValidationError):
        transport.solve(solver="mosek", solver_options={"blabla": "1.e12"})

    ## Snopt
    with pytest.raises(ValidationError):
        transport.solve(solver="snopt", solver_options={"blabla": "1.e12"})

    ## Soplex will not care
    transport.solve(solver="soplex", solver_options={"blabla": "1.e12"})

    ## Xpress
    with pytest.raises(ValidationError):
        transport.solve(solver="xpress", solver_options={"blabla": "1.e12"})

    # Test disabled solver option validation
    gp.set_options({"SOLVER_OPTION_VALIDATION": 0})
    transport.solve(solver="conopt", solver_options={"blabla": "1.e12"})
    gp.set_options({"SOLVER_OPTION_VALIDATION": 1})


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

    domain = validation._expand_ellipsis_slice(["a", "b", "c"], ["a", ...])
    assert domain == ["a", "b", "c"]

    domain = validation._expand_ellipsis_slice(
        ["a", "b", "c"], ["a", ..., "c"]
    )
    assert domain == ["a", "b", "c"]

    domain = validation._expand_ellipsis_slice(["a", "b", "c"], [..., "c"])
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
    for _ in range(1300):
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

    v = Set(m, "v")
    k = Set(m, "k")
    z = Variable(m, "z", domain=[v, k])
    vk = Set(m, "vk", domain=[v, k])
    n = Parameter(m, domain=k)

    z.up[vk[v, k]] = n[k]


def test_context_manager(data):
    with Container():
        i = Set()
        a = Alias(alias_with=i)
        _ = Parameter()
        _ = Variable()
        _ = Equation()

    m, canning_plants, markets, capacities, demands, distances = data
    with m:
        i = Set(records=canning_plants)
        j = Set(name="j", records=markets)

        a = Parameter(name="a", domain=i, records=capacities)
        b = Parameter(name="b", domain=j, records=demands)
        d = Parameter(name="d", domain=[i, j], records=distances)
        c = Parameter(
            name="c",
            domain=[i, j],
            description="transport cost in thousands of dollars per case",
        )
        c[i, j] = 90 * d[i, j] / 1000

        # Variable
        x = Variable(
            name="x",
            domain=[i, j],
            type="Positive",
            description="shipment quantities in cases",
        )

        # Equation
        supply = Equation(
            name="supply",
            domain=i,
            description="observe supply limit at plant i",
        )
        demand = Equation(
            name="demand", domain=j, description="satisfy demand at market j"
        )

        supply[i] = Sum(j, x[i, j]) <= a[i]
        demand[j] = Sum(i, x[i, j]) >= b[j]

        transport = Model(
            name="transport",
            equations=m.getEquations(),
            problem="LP",
            sense=Sense.MIN,
            objective=Sum((i, j), c[i, j] * x[i, j]),
        )
        transport.solve()

    import math

    assert math.isclose(transport.objective_value, 153.675000, rel_tol=0.001)

    assert i.container.working_directory is m.working_directory

    # We should still be able to access the symbols of m
    c[i, j] = 90 * d[i, j] / 100
    transport.solve()
    assert math.isclose(transport.objective_value, 1536.75000, rel_tol=0.001)

    m2 = Container()
    i2 = Set(m2, "i2")
    a2 = Parameter(m2, "a2", domain=i2)
    assert i2.container.working_directory is m2.working_directory
    assert a2.container.working_directory is m2.working_directory

    with m2:
        i3 = Set(m2, "i3")
    assert i3.container.working_directory is m2.working_directory

    # Make sure that the symbols of m is not affected by the new context manager
    assert i.container.working_directory is m.working_directory


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
        transport.solve(solver="miles")

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


def test_multiprocessing_with_ctx():
    f_values = [90, 120, 150, 180]
    expected_values = [153.675, 204.89999999999998, 256.125, 307.35]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for expected, objective in zip(
            expected_values, executor.map(transport_with_ctx, f_values)
        ):
            assert math.isclose(expected, objective)

    assert gp._ctx_managers == {}


def test_threading_with_ctx():
    f_values = [90, 120, 150, 180]
    expected_values = [153.675, 204.89999999999998, 256.125, 307.35]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for expected, objective in zip(
            expected_values, executor.map(transport_with_ctx, f_values)
        ):
            assert math.isclose(expected, objective)

    assert gp._ctx_managers == {}


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


def test_emp():
    m = Container()
    t = Set(m, name="m", records=[0, 1])
    a = Set(m, name="a", records=["a0", "a1"])
    beta = 7
    alpha = 6

    x = Variable(m, domain=[a, t])
    obj = Variable(m, name="obj", domain=[a])

    oterms = a.toList()
    cterms = a.toList()

    # Agent 0
    oterms[0] = (
        beta / 2 * gamspy_math.sqr(x["a0", "0"]) - alpha * x["a0", "0"]
    ) + (
        1 / 2 * gamspy_math.sqr(x["a0", "1"])
        + 3 * x["a0", "1"] * x["a1", "1"]
        - 4 * x["a0", "1"]
    )
    cterms[0] = x["a0", "1"] - x["a0", "0"]

    # Agent 1
    oterms[1] = x["a1", "0"] + (
        1 / 2 * gamspy_math.sqr(x["a1", "1"])
        + x["a0", "1"] * x["a1", "1"]
        - 3 * x["a1", "1"]
    )
    cterms[1] = x["a1", "1"]

    defobj = Equation(m, name="defobj", domain=a)
    defobj[a] = obj[a] == sum(
        o.where[a.sameAs(f"a{i}")] for i, o in enumerate(oterms)
    )

    cons = Equation(m, name="cons", domain=a)
    cons[a] = (
        sum(c.where[a.sameAs(f"a{i}")] for i, c in enumerate(cterms)) >= 0
    )

    x.lo["a0", "0"] = 0
    x.fx["a1", "0"] = 0

    with pytest.raises(ValidationError):
        _ = Model(m, name="nash", equations=[defobj, cons], problem="emp")

    m.close()


def test_subsolver_options():
    m = Container()
    t = Set(m, name="m", records=[0, 1])
    a = Set(m, name="a", records=["a0", "a1"])
    beta = 7
    alpha = 6

    x = Variable(m, name="x", domain=[a, t])
    obj = Variable(m, name="obj", domain=[a])

    oterms = a.toList()
    cterms = a.toList()

    # Agent 0
    oterms[0] = (
        beta / 2 * gamspy_math.sqr(x["a0", "0"]) - alpha * x["a0", "0"]
    ) + (
        1 / 2 * gamspy_math.sqr(x["a0", "1"])
        + 3 * x["a0", "1"] * x["a1", "1"]
        - 4 * x["a0", "1"]
    )
    cterms[0] = x["a0", "1"] - x["a0", "0"]

    # Agent 1
    oterms[1] = x["a1", "0"] + (
        1 / 2 * gamspy_math.sqr(x["a1", "1"])
        + x["a0", "1"] * x["a1", "1"]
        - 3 * x["a1", "1"]
    )
    cterms[1] = x["a1", "1"]

    defobj = Equation(m, name="defobj", domain=a)
    defobj[a] = obj[a] == sum(
        o.where[a.sameAs(f"a{i}")] for i, o in enumerate(oterms)
    )

    cons = Equation(m, name="cons", domain=a)
    cons[a] = (
        sum(c.where[a.sameAs(f"a{i}")] for i, c in enumerate(cterms)) >= 0
    )

    x.lo["a0", "0"] = 0
    x.fx["a1", "0"] = 0

    nash = Model(m, name="nash", equations=[defobj, cons], problem="emp")

    ReSHOPAnnotation(
        m,
        """
    n(a): min obj(a) x(a,'*') defobj(a) cons(a)
    root: Nash(n(a))
    """,
    )

    m.writeSolverOptions("path", {"crash_method": "none", "prox_pert": 0})

    gp.set_options({"ALLOW_AMBIGUOUS_EQUATIONS": "yes"})

    with tempfile.NamedTemporaryFile("w", delete=False) as file:
        nash.solve(
            solver="reshop",
            options=Options(log_file="log.log"),
            output=file,
            solver_options={"subsolveropt": 1},
        )
        file.close()

        with open(file.name) as f:
            assert ">>  crash_method none" in f.read()

        os.remove(file.name)

    m.writeSolverOptions("nlpec", {"testTol": 1e-006})
    with tempfile.NamedTemporaryFile("w", delete=False) as file:
        nash.solve(
            solver="reshop",
            options=Options(log_file="log.log", mcp="nlpec"),
            output=file,
            solver_options={"subsolveropt": 1},
        )
        file.close()
        with open(file.name) as f:
            content = f.read()
            print(content)
            assert ">>  testTol 1e-06" in content

    assert np.isclose(
        x.records["level"].to_numpy(),
        np.array([0.8571428571428571, 2.5, 0.0, 0.5]),
    ).all()

    m.close()
    gp.set_options({"ALLOW_AMBIGUOUS_EQUATIONS": "auto"})


def test_ambiguity():
    m = gp.Container()

    c = gp.Parameter(m, name="c", domain=[], records=0.5)
    x = gp.Variable(m, name="x", type="negative")
    f = gp.Equation(m, name="f")
    f[...] = (x - c) >= 0
    assert f.getDefinition() == "f .. x - c =g= 0;"

    f2 = gp.Equation(m, name="f2", definition=(x - c) >= 0)
    assert f2.getDefinition() == "f2 .. x - c =g= 0;"
    assert f.latexRepr() == f2.latexRepr()

    mcp_model = gp.Model(m, "mcp_model", problem="MCP", matches={f: x})

    gp.set_options({"ALLOW_AMBIGUOUS_EQUATIONS": "auto"})

    with pytest.raises(ValidationError):
        mcp_model.solve()

    gp.set_options({"ALLOW_AMBIGUOUS_EQUATIONS": "yes"})
    with pytest.raises(GamspyException):
        mcp_model.solve()  # error from GAMS bad bound on variable x

    gp.set_options({"ALLOW_AMBIGUOUS_EQUATIONS": "no"})

    with pytest.raises(ValidationError):
        mcp_model.solve()  # ambiguous equations not allowed

    lp_model = gp.Model(
        m,
        "lp_model",
        problem="LP",
        equations=[f],
        objective=x + 2,
        sense="MIN",
    )
    gp.set_options({"ALLOW_AMBIGUOUS_EQUATIONS": "auto"})
    lp_model.solve()
    gp.set_options({"ALLOW_AMBIGUOUS_EQUATIONS": "yes"})
    lp_model.solve()
    gp.set_options({"ALLOW_AMBIGUOUS_EQUATIONS": "no"})
    with pytest.raises(ValidationError):
        lp_model.solve()
    gp.set_options({"ALLOW_AMBIGUOUS_EQUATIONS": "auto"})
