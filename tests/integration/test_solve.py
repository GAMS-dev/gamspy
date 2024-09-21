# flake8: noqa
# fmt: off

from __future__ import annotations

import concurrent.futures
import logging
import math
import os
import sys
import time
import pytest

import gamspy._validation as validation
import numpy as np
from gamspy import (
    Card,
    Container,
    Equation,
    Model,
    ModelStatus,
    Options,
    Ord,
    Parameter,
    Sense,
    Set,
    Smax,
    SolveStatus,
    Sum,
    Variable,
)
import math
import gamspy.math as gamspy_math
from gamspy.exceptions import GamspyException, ValidationError

import pytest
pytestmark = pytest.mark.integration

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
        description="canning plants"
    )
    j = Set(
        m,
        name="j",
        records=["new-york", "chicago", "topeka"],
        description="markets"
    )

    # Data
    a = Parameter(
        m,
        name="a",
        domain=i,
        records=capacities,
        description="capacity of plant i in cases"
    )
    b = Parameter(
        m,
        name="b",
        domain=j,
        records=demands,
        description="demand at market j in cases"
    )
    d = Parameter(
        m,
        name="d",
        domain=[i, j],
        records=distances,
        description="distance in thousands of miles"
    )
    c = Parameter(
        m,
        name="c",
        domain=[i, j],
        description="transport cost in thousands of dollars per case"
    )
    f = Parameter(m, name="f", records=f_value)
    c[i, j] = f * d[i, j] / 1000

    # Variable
    x = Variable(
        m,
        name="x",
        domain=[i, j],
        type="Positive",
        description="shipment quantities in cases"
    )

    # Equation
    supply = Equation(
        m,
        name="supply",
        domain=i,
        description="observe supply limit at plant i"
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
        description="canning plants"
    )
    j = Set(
        m,
        name="j",
        records=["new-york", "chicago", "topeka"],
        description="markets"
    )

    # Data
    a = Parameter(
        m,
        name="a",
        domain=i,
        records=capacities,
        description="capacity of plant i in cases"
    )
    b = Parameter(
        m,
        name="b",
        domain=j,
        records=demands,
        description="demand at market j in cases"
    )
    d = Parameter(
        m,
        name="d",
        domain=[i, j],
        records=distances,
        description="distance in thousands of miles"
    )
    c = Parameter(
        m,
        name="c",
        domain=[i, j],
        description="transport cost in thousands of dollars per case"
    )
    f = Parameter(m, name="f", records=f_value)
    c[i, j] = f * d[i, j] / 1000

    # Variable
    x = Variable(
        m,
        name="x",
        domain=[i, j],
        type="Positive",
        description="shipment quantities in cases"
    )

    # Equation
    supply = Equation(
        m,
        name="supply",
        domain=i,
        description="observe supply limit at plant i"
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
    i = Set(m,'i')
    p = m.addParameter('base',[i])
    d = Parameter(m,'d')
    i.setRecords(['i1','i2'])
    d[...] = 0
    i.setRecords(['i0','i1'])
    p[i] = i.ord
    assert(p.records.values.tolist() == [['i1', 1.0], ['i0', 2.0]])


def test_read_on_demand(data):
    m, canning_plants, markets, capacities, demands, distances = data
    i = Set(m, name="i", records=canning_plants)
    j = Set(m, name="j", records=markets)
    k = Set(
        m, name="k", records=["seattle", "san-diego", "california"]
    )
    k["seattle"] = False
    assert(
        k.records.loc[0, :].values.tolist() == ["san-diego", ""]
    )

    a = Parameter(m, name="a", domain=[i], records=capacities)
    b = Parameter(m, name="b", domain=[j], records=demands)
    d = Parameter(m, name="d", domain=[i, j], records=distances)
    c = Parameter(m, name="c", domain=[i, j])
    e = Parameter(m, name="e")

    c[i, j] = 90 * d[i, j] / 1000
    assert(
        c.records.values.tolist() ==
        [
            ["seattle", "new-york", 0.225],
            ["seattle", "chicago", 0.153],
            ["seattle", "topeka", 0.162],
            ["san-diego", "new-york", 0.225],
            ["san-diego", "chicago", 0.162],
            ["san-diego", "topeka", 0.126],
        ]
    )
    e[...] = 5
    assert(e.records.values.tolist() == [[5.0]])

    with pytest.raises(TypeError):
        e.records = 5

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
    transport.solve()

    # Test the columns of a set
    assert(i.records.columns.tolist() == ["uni", "element_text"])

    # Test the columns of a parameter
    assert(a.records.columns.tolist() == ["i", "value"])

    # Test the columns of scalar variable
    assert(
        z.records.columns.tolist()
        == ["level", "marginal", "lower", "upper", "scale"]
    )

    # Test the columns of indexed variable
    assert(
        x.records.columns.tolist()
        == ["i", "j", "level", "marginal", "lower", "upper", "scale"]
    )

    # Test the columns of equation
    assert(
        cost.records.columns.tolist()
        == ["level", "marginal", "lower", "upper", "scale"]
    )

    # Test the columns of indexed equation
    assert(
        supply.records.columns.tolist()
        == ["i", "level", "marginal", "lower", "upper", "scale"]
    )

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

    assert(z.records is not None)
    assert(x.records is not None)
    assert(cost.records is not None)
    assert(supply.records is not None)
    assert(demand.records is not None)

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
    
    
    freeLinks = Set(m, "freeLinks", domain=[i,j], records=[('seattle', 'chicago')])
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

    assert(
        transport2.getDeclaration() ==
        "Model transport2 / cost,supply,demand,x(freeLinks) /;"
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
        def write(self, data):
            ...

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

        def flush(self):
            ...

    custom_output = CustomOutput()

    _ = transport.solve(
        options=Options(time_limit=100),
        output=custom_output,
    )

    assert(os.path.exists(redirection_path))
    assert(transport.status == ModelStatus.OptimalGlobal)
    assert(transport.solve_status == SolveStatus.NormalCompletion)

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
        assert(hasattr(transport, attr_name))

        # Make sure model attributes are not in the container
        assert not attr_name in m.data.keys()

    # Make sure dummy variable and equation is not in the container
    assert not any("dummy_" in name for name in m.data.keys())

    # Test invalid problem
    pytest.raises(ValueError, Model, m, "dummy", "bla", [cost])

    # Test invalid sense
    pytest.raises(
        ValueError, Model, m, "dummy", "LP", [cost], "bla"
    )

    # Test invalid objective variable
    pytest.raises(
        TypeError, Model, m, "dummy", "LP", [cost], "min", a
    )

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

def test_interrupt(data):
    m, *_ = data
    cont = Container()

    power_forecast_recs = np.array(
        [
            287,
            275,
            262,
            250,
            255,
            260,
            265,
            270,
            267,
            265,
            262,
            260,
            262,
            265,
            267,
            270,
            277,
            285,
            292,
            300,
            310,
            320,
            330,
            340,
            357,
            375,
            392,
            410,
            405,
            400,
            395,
            390,
            400,
            410,
            420,
            430,
            428,
            427,
            426,
            425,
            432,
            440,
            447,
            455,
            458,
            462,
            466,
            470,
            466,
            462,
            458,
            455,
            446,
            437,
            428,
            420,
            416,
            412,
            408,
            405,
            396,
            387,
            378,
            370,
            375,
            380,
            385,
            390,
            383,
            377,
            371,
            365,
            368,
            372,
            376,
            380,
            386,
            392,
            398,
            405,
            408,
            412,
            416,
            420,
            413,
            407,
            401,
            395,
            386,
            377,
            368,
            360,
            345,
            330,
            315,
            300,
        ]
    )

    # Energy
    t = Set(
        cont,
        name="t",
        records=[f"t{i}" for i in range(1, 97)],
        description="time slices (quarter-hour)",
    )

    PowerForecast = Parameter(
        cont,
        name="PowerForecast",
        domain=[t],
        records=power_forecast_recs,
        description="electric power forecast",
    )

    # Power Plant (PP)
    cPPvar = Parameter(
        cont,
        name="cPPvar",
        records=25,
        description="variable cost of power plant [euro / MWh]",
    )
    pPPMax = Parameter(
        cont,
        name="pPPMax",
        records=300,
        description="maximal capacity of power plant      [MW]",
    )

    m = Set(
        cont,
        name="m",
        records=[f"m{i}" for i in range(1, 9)],
        description="'stage of the power plant",
    )
    iS = Set(
        cont,
        name="iS",
        records=[f"iS{i}" for i in range(9)],
        description="interval for constant PP operation",
    )
    iI = Set(
        cont,
        name="iI",
        records=[f"iI{i}" for i in range(17)],
        description="length of idle time period",
    )

    cBL = Parameter(
        cont,
        name="cBL",
        records=32,
        description="cost for one base load contract [euro / MWh]",
    )
    cPL = Parameter(
        cont,
        name="cPL",
        records=41,
        description="cost for one peak load contract [euro / MWh]",
    )

    IPL = Parameter(
        cont,
        name="IPL",
        domain=[t],
        description="indicator function for peak load contracts",
    )
    IPL[t] = (Ord(t) >= 33) & (Ord(t) <= 80)

    pLFCref = Parameter(
        cont,
        name="pLFCref",
        records=400,
        description="power reference level for the LFC",
    )

    b = Set(
        cont,
        name="b",
        records=[f"b{i}" for i in range(1, 4)],
        description="support points of the zone prices",
    )

    eLFCbY = Parameter(
        cont,
        name="eLFCbY",
        domain=[b],
        records=np.array([54750, 182500, 9000000]),
        description="amount of energy at support point b",
    )
    cLFCvar = Parameter(
        cont,
        name="cLFCvar",
        domain=[b],
        records=np.array([80.0, 65.0, 52.0]),
        description="specific energy price in segment b",
    )
    eLFCb = Parameter(
        cont,
        name="eLFCb",
        domain=[b],
        description="daily border of energy volumes for LFC",
    )
    cLFCs = Parameter(
        cont,
        name="cLFCs",
        domain=[b],
        description="accumulated cost for LFC up to segment b",
    )

    # calculate the daily borders of the energy volumes for the zones
    eLFCb[b] = eLFCbY[b] / 365

    # calculate the accumulated cost
    cLFCs["b1"] = 0
    cLFCs["b2"] = cLFCvar["b1"] * eLFCb["b1"]
    cLFCs[b].where[Ord(b) > 2] = cLFCs[b.lag(1)] + cLFCvar[b.lag(1)] * (
        eLFCb[b.lag(1)] - eLFCb[b.lag(2)]
    )

    c = Variable(cont, name="c", type="free", description="total cost")
    cPP = Variable(
        cont, name="cPP", type="positive", description="cost of PP usage"
    )
    pPP = Variable(
        cont,
        name="pPP",
        type="positive",
        domain=[t],
        description="power withdrawn from power plant",
    )
    delta = Variable(
        cont,
        name="delta",
        type="binary",
        domain=[m, t],
        description="indicate if the PP is in stage m at time t",
    )
    chiS = Variable(
        cont,
        name="chiS",
        type="positive",
        domain=[t],
        description="indicate if there is a PP stage change",
    )
    chiI = Variable(
        cont,
        name="chiI",
        type="positive",
        domain=[t],
        description="indicate if the PP left the idle stage",
    )
    cSM = Variable(
        cont,
        name="cSM",
        type="positive",
        description="cost of energy from SM",
    )
    pSM = Variable(
        cont,
        name="pSM",
        type="positive",
        domain=[t],
        description="power from the spot market",
    )
    alpha = Variable(
        cont,
        name="alpha",
        type="integer",
        description="quantity of base load contracts",
    )
    beta = Variable(
        cont,
        name="beta",
        type="integer",
        description="quantity of peak load contracts",
    )
    cLFC = Variable(
        cont,
        name="cLFC",
        type="positive",
        description="cost of LFC which is the enery rate",
    )
    eLFCtot = Variable(
        cont,
        name="eLFCtot",
        type="positive",
        description="total energy amount of LFC",
    )
    eLFCs = Variable(
        cont,
        name="eLFCs",
        type="positive",
        domain=[b],
        description="energy from LFC in segment b",
    )
    pLFC = Variable(
        cont,
        name="pLFC",
        type="positive",
        domain=[t],
        description="power from the LFC",
    )
    mu = Variable(
        cont,
        name="mu",
        type="binary",
        domain=[b],
        description="indicator for segment b (for zone prices)",
    )

    alpha.up[...] = Smax(t, PowerForecast[t])
    beta.up[...] = alpha.up
    pLFC.up[t] = pLFCref

    obj = Equation(cont, name="obj", description="objective function")
    demand = Equation(
        cont,
        name="demand",
        domain=[t],
        description="demand constraint for energy forcast",
    )
    PPcost = Equation(cont, name="PPcost", description="power plant cost")
    PPpower = Equation(
        cont,
        name="PPpower",
        domain=[t],
        description="power of power plant at time t",
    )
    PPstage = Equation(
        cont,
        name="PPstage",
        domain=[t],
        description="exactly one stage of power plant at any time",
    )
    PPchiS1 = Equation(
        cont,
        name="PPchiS1",
        domain=[t, m],
        description="relate chi and delta variables first constraint",
    )
    PPchiS2 = Equation(
        cont,
        name="PPchiS2",
        domain=[t, m],
        description="relate chi and delta variables second constraint",
    )
    PPstageChange = Equation(
        cont,
        name="PPstageChange",
        domain=[t],
        description="restrict the number of stage changes",
    )
    PPstarted = Equation(
        cont,
        name="PPstarted",
        domain=[t],
        description="connect chiZ and chi variables",
    )
    PPidleTime = Equation(
        cont,
        name="PPidleTime",
        domain=[t],
        description="control the idle time of the plant",
    )
    SMcost = Equation(
        cont,
        name="SMcost",
        description="cost associated with spot market",
    )
    SMpower = Equation(
        cont,
        name="SMpower",
        domain=[t],
        description="power from the spot market",
    )
    LFCcost = Equation(
        cont, name="LFCcost", description="cost for the LFC"
    )
    LFCenergy = Equation(
        cont,
        name="LFCenergy",
        description="total energy from the LFC",
    )
    LFCmu = Equation(
        cont,
        name="LFCmu",
        description="exactly one price segment b",
    )
    LFCenergyS = Equation(
        cont,
        name="LFCenergyS",
        description="connect the mu variables with the total energy",
    )
    LFCemuo = Equation(
        cont,
        name="LFCemuo",
        description="accumulated energy amount for segement b1",
    )
    LFCemug = Equation(
        cont,
        name="LFCemug",
        domain=[b],
        description="accumulated energy amount for all other segements",
    )

    # the objective function: total cost eq. (6)
    obj[...] = c == cPP + cSM + cLFC

    # meet the power demand for each time period exactly eq. (23)
    demand[t] = pPP[t] + pSM[t] + pLFC[t] == PowerForecast[t]

    # (fix cost +) variable cost * energy amount produced eq. (7) & (8)
    PPcost[...] = cPP == cPPvar * Sum(t, 0.25 * pPP[t])

    # power produced by the power plant eq. (26)
    PPpower[t] = pPP[t] == pPPMax * Sum(
        m.where[Ord(m) > 1], 0.1 * (Ord(m) + 2) * delta[m, t]
    )

    # the power plant is in exactly one stage at any time eq. (25)
    PPstage[t] = Sum(m, delta[m, t]) == 1

    # next constraints model the minimum time period a power plant is in
    # the same state and the constraint of the minimum idle time
    # we need variable 'chiS' to find out when a status change takes place
    # eq. (27)
    PPchiS1[t, m].where[Ord(t) > 1] = (
        chiS[t] >= delta[m, t] - delta[m, t.lag(1)]
    )

    # second constraint for 'chiS' variable eq. (28)
    PPchiS2[t, m].where[Ord(t) > 1] = (
        chiS[t] >= delta[m, t.lag(1)] - delta[m, t]
    )

    # control the minimum change time period eq. (29)
    PPstageChange[t].where[Ord(t) < Card(t) - Card(iS) + 2] = (
        Sum(iS, chiS[t.lead(Ord(iS))]) <= 1
    )

    # indicate if the plant left the idle state eq. (30)
    PPstarted[t] = chiI[t] >= delta["m1", t.lag(1)] - delta["m1", t]

    # control the minimum idle time period:
    # it has to be at least Nk2 time periods long eq. (31)
    PPidleTime[t].where[Ord(t) < Card(t) - Card(iI) + 2] = (
        Sum(iI, chiI[t.lead(Ord(iI))]) <= 1
    )

    # cost for the spot market eq. (12)
    # consistent of the base load (alpha) and peak load (beta) contracts
    SMcost[...] = cSM == 24 * cBL * alpha + 12 * cPL * beta

    # Spot Market power contribution eq. (9)
    SMpower[t] = pSM[t] == alpha + IPL[t] * beta

    # cost of the LFC is given by the energy rate eq. (14) & (21)
    LFCcost[...] = cLFC == Sum(b, cLFCs[b] * mu[b] + cLFCvar[b] * eLFCs[b])

    # total energy from the LFC eq. (16)
    # connect the eLFC[t] variables with eLFCtot
    LFCenergy[...] = eLFCtot == Sum(t, 0.25 * pLFC[t])

    # indicator variable 'mu':
    # we are in exactly one price segment b eq. (18)
    LFCmu[...] = Sum(b, mu[b]) == 1

    # connect the 'mu' variables with the total energy amount eq. (19)
    LFCenergyS[...] = eLFCtot == Sum(
        b.where[Ord(b) > 1], eLFCb[b.lag(1)] * mu[b]
    ) + Sum(b, eLFCs[b])

    # accumulated energy amount for segment "b1" eq. (20)
    LFCemuo[...] = eLFCs["b1"] <= eLFCb["b1"] * mu["b1"]

    # accumulated energy amount for all other segments (then "b1") eq. (20)
    LFCemug[b].where[Ord(b) > 1] = (
        eLFCs[b] <= (eLFCb[b] - eLFCb[b.lag(1)]) * mu[b]
    )

    energy = Model(
        cont,
        name="energy",
        equations=cont.getEquations(),
        problem="MIP",
        sense=Sense.MIN,
        objective=c,
    )
    
    def interrupt_gams(model):
        time.sleep(1)
        model.interrupt()

    import threading

    threading.Thread(target=interrupt_gams, args=(energy,)).start()
    
    try:
        energy.solve(options=Options(relative_optimality_gap=0.000001))
        assert(energy.objective_value)
    except GamspyException:
        pass

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

    assert(
        os.path.exists(
            f"{m.working_directory}{os.sep}conopt.opt"
        )
    )

    pytest.raises(
        ValidationError, transport.solve, None, None, {"rtmaxv": "1.e12"}
    )

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
        f[i,j] = 5

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
    assert(transport.objective_value== 153.675)

    supply.l[...] = 5
    assert(supply.records.level.to_list() == [5.0, 5.0])
    
    domain = validation._transform_given_indices(["a", "b", "c"], ["a", ...])
    assert(domain == ["a", "b", "c"])

    domain = validation._transform_given_indices(["a", "b", "c"], ["a", ..., "c"])
    assert(domain == ["a", "b", "c"])

    domain = validation._transform_given_indices(["a", "b", "c"], [..., "c"])
    assert(domain == ["a", "b", "c"])
    
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
    assert(k[..., i4].gamsRepr() == "k(i,i2,i3,i4)")
    assert(k[i, ..., i4].gamsRepr() == "k(i,i2,i3,i4)")
    assert(k[i, ...].gamsRepr() == "k(i,i2,i3,i4)")
    assert(k[..., i3, :].gamsRepr() == "k(i,i2,i3,i4)")

    a = Parameter(m, name="a", domain=[i], records=capacities)
    assert(a[:].gamsRepr() == "a(i)")
    d = Parameter(m, name="d", domain=[i, j], records=distances)
    c = Parameter(m, name="c", domain=[i, j])
    assert(c[:, :].gamsRepr() == "c(i,j)")
    c[:, :] = 90 * d[:, :] / 1000

    x = Variable(m, name="x", domain=[i, j], type="Positive")
    assert(x[:, :].gamsRepr() == "x(i,j)")

    supply = Equation(m, name="supply", domain=[i])
    assert(supply[:].gamsRepr() == "supply(i)")

    supply.l[:] = 5
    assert(supply.l[:].gamsRepr() == "supply.l(i)")
    
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
    f = Parameter(m, "fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", records=1)
    c[i, j] = 90 * d[i, j] / 1000

    x = Variable(m, name="x", domain=[i, j], type="Positive")

    supply = Equation(m, name="supply", domain=[i])
    demand = Equation(m, name="demand", domain=[j])
    
    # This generates an equation with length > 80000
    supply[i] = Sum(j, x[i, j])*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f <= a[i]
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
    assert(summary['Solver Status'].tolist()[0] == 'Normal')
    
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
    
    assert(f.getAssignment() == "f = 5;")
        
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
    assert(w.records is not None)
    assert(loss.records is not None)

def test_multiprocessing():
    f_values = [90, 120, 150, 180]
    expected_values = [153.675, 204.89999999999998, 256.125, 307.35]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for expected, objective in zip(expected_values, executor.map(transport, f_values)):
            assert(math.isclose(expected, objective))

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
        transport.solve(load_symbols=['x'])

    with pytest.raises(ValidationError):
        transport.solve(load_symbols=x)

    transport.solve(load_symbols=[])

    assert x.records is None
    assert supply.records is None
    assert(transport.objective_value == 153.675)

    transport.solve(load_symbols=[x])

    assert x.records is not None
    assert supply.records is None
    assert(transport.objective_value == 153.675)

def test_execution_error(data):
    m, *_ = data
    m = Container()
    x = Variable(m)
    y = Variable(m)
    obj = gamspy_math.sqr(math.pi - x / y)
    model = Model(m, objective=obj, problem="nlp", sense="min")
    try:
        model.solve()  # this will trigger a division by 0 execution error
    except:
        y.l = 1

    summary = model.solve()  # this should work
    assert(summary is not None)
