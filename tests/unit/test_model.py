from __future__ import annotations

import os
import tempfile
import uuid

import pandas as pd
import pytest

from gamspy import (
    Alias,
    Container,
    Equation,
    Model,
    Options,
    Parameter,
    Problem,
    Sense,
    Set,
    Sum,
    Variable,
    VariableType,
)
from gamspy._model import FileFormat
from gamspy._options import ConvertOptions
from gamspy.exceptions import ValidationError

pytestmark = pytest.mark.unit


@pytest.fixture
def data():
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

    yield m, canning_plants, markets, distances, capacities, demands
    m.close()


def test_model(data):
    m, canning_plants, markets, distances, capacities, demands = data
    i = Set(
        m,
        name="i",
        records=canning_plants,
        description="Canning Plants",
    )
    j = Set(m, name="j", records=markets, description="Markets")

    # Params
    a = Parameter(m, name="a", domain=[i], records=capacities)
    b = Parameter(m, name="b", domain=[j], records=demands)
    d = Parameter(m, name="d", domain=[i, j], records=distances)
    c = Parameter(m, name="c", domain=[i, j])
    c[i, j] = 90 * d[i, j] / 1000

    x = Variable(m, name="x", domain=[i, j], type="Positive")
    z = Variable(m, name="z")

    # Equation definition without an index
    cost = Equation(
        m,
        name="cost",
        description="define objective function",
    )
    cost[...] = Sum((i, j), c[i, j] * x[i, j]) == z

    # Equation definition with an index
    supply = Equation(
        m,
        name="supply",
        domain=[i],
        description="observe supply limit at plant i",
    )
    supply[i] = Sum(j, x[i, j]) <= a[i]

    demand = Equation(m, name="demand", domain=[j])
    demand[j] = Sum(i, x[i, j]) >= b[j]

    with pytest.raises(ValidationError):
        _ = Model(m, sense="min")

    # Model with implicit objective
    test_model = Model(
        m,
        name="test_model",
        equations=[supply, demand],
        problem="LP",
        sense="min",
        objective=Sum((i, j), c[i, j] * x[i, j]),
    )
    test_model.solve(solver="CPLEX")
    assert list(m.data.keys()) == [
        "i",
        "j",
        "a",
        "b",
        "d",
        "c",
        "x",
        "z",
        "cost",
        "supply",
        "demand",
        "test_model_objective_variable",
        "test_model_objective",
    ]
    assert test_model.objective_value == 153.675

    # Test convert
    with tempfile.TemporaryDirectory() as tmpdir:
        test_model.convert(
            tmpdir,
            file_format=FileFormat.GDXJacobian,
            options=ConvertOptions(GDXNames=0),
        )
        assert os.path.exists(os.path.join(tmpdir, "jacobian.gdx"))
        with open(os.path.join(m.working_directory, "convert.opt")) as f:
            assert "GDXNames 0" in f.read()

    # Test unknown file format
    with pytest.raises(ValidationError):
        test_model.convert(
            tmpdir,
            file_format="Unknown format",
            options=ConvertOptions(GDXNames=0),
        )

    # Test GAMSJacobian
    with tempfile.TemporaryDirectory() as tmpdir:
        test_model.convert(
            tmpdir,
            file_format=[
                FileFormat.GAMSJacobian,
                FileFormat.GDXJacobian,
                FileFormat.GAMSPyJacobian,
            ],
            options=ConvertOptions(GDXNames=0),
        )
        assert os.path.exists(os.path.join(tmpdir, "jacobian.gms"))
        assert os.path.exists(os.path.join(tmpdir, "jacobian.gdx"))
        assert os.path.exists(os.path.join(tmpdir, "jacobian.py"))

        with open(os.path.join(tmpdir, "jacobian.gms")) as file:
            assert (
                "$if not set jacfile $set jacfile jacobian.gdx" in file.read()
            )

    # Check if the name is reserved
    pytest.raises(ValidationError, Model, m, "set", "", "LP")

    # Equation definition with more than one index
    bla = Equation(
        m,
        name="bla",
        domain=[i, j],
        description="observe supply limit at plant i",
    )
    bla[i, j] = x[i, j] <= a[i]

    # Test model with specific equations
    test_model2 = Model(
        m,
        name="test_model2",
        equations=[cost, supply],
        problem="LP",
        sense="min",
        objective=z,
    )
    assert test_model2.getDeclaration() == "Model test_model2 / cost,supply /;"
    assert test_model2.equations == [cost, supply]

    test_model3 = Model(
        m,
        name="test_model3",
        equations=[cost],
        problem="LP",
        sense="min",
        objective=z,
    )
    test_model3.equations = [cost, supply]
    assert test_model3.equations == [cost, supply]

    test_model4 = m.addModel(
        name="test_model4",
        equations=[cost, supply],
        problem="LP",
        sense="min",
        objective=z,
    )

    assert test_model4.equations == test_model3.equations

    test_model5 = m.addModel(
        name="test_model5",
        equations=[cost, supply],
        problem="LP",
        sense="min",
        objective=z,
        matches={supply: x, cost: z},
    )
    assert (
        test_model5.getDeclaration()
        == "Model test_model5 / supply.x,cost.z /;"
    )

    # Equations provided as strings
    pytest.raises(
        ValueError, Model, m, "test_model5", "", "LP", ["cost", "supply"]
    )

    # Test matches
    test_model6 = Model(
        m,
        name="test_model6",
        equations=[supply],
        matches={demand: x},
        problem="LP",
        sense="min",
    )
    assert (
        test_model6.getDeclaration()
        == "Model test_model6 / supply,demand.x /;"
    )

    # Test no name
    _ = Model(
        m,
        equations=[supply, demand],
        problem="LP",
        sense="min",
        objective=Sum((i, j), c[i, j] * x[i, j]),
    )
    m.addModel(
        equations=[supply, demand],
        problem="LP",
        sense="min",
        objective=Sum((i, j), c[i, j] * x[i, j]),
    )

    # Test repr and str
    assert str(test_model6).startswith(
        f"Model {test_model6.name}:\n  Problem Type: LP\n  Sense: MIN\n  Equations:"
    )

    # empty model name
    pytest.raises(
        ValueError,
        Model,
        m,
        "",
        "test_model7",
        "",
        m.getEquations(),
        "min",
        Sum((i, j), c[i, j] * x[i, j]),
    )

    # model name too long
    pytest.raises(
        ValidationError,
        Model,
        m,
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "",
        "LP",
        m.getEquations(),
        "min",
        Sum((i, j), c[i, j] * x[i, j]),
    )

    # model name is not an str
    pytest.raises(
        TypeError,
        Model,
        m,
        5,
        "",
        "LP",
        m.getEquations(),
        "min",
        Sum((i, j), c[i, j] * x[i, j]),
    )

    # model name contains empty space
    pytest.raises(
        ValidationError,
        Model,
        m,
        "test_model 8",
        "",
        "LP",
        m.getEquations(),
        "min",
        Sum((i, j), c[i, j] * x[i, j]),
    )

    # model name begins with underscore
    pytest.raises(
        ValidationError,
        Model,
        m,
        "_test_model7",
        "",
        "LP",
        m.getEquations(),
        "min",
        Sum((i, j), c[i, j] * x[i, j]),
    )

    test_model8 = Model(
        m,
        name="test_model8",
        description="some description",
        equations=[supply, demand],
        problem="LP",
        sense="min",
        objective=Sum((i, j), c[i, j] * x[i, j]),
    )
    assert (
        test_model8.getDeclaration()
        == 'Model test_model8 "some description" / supply,demand,test_model8_objective /;'
    )


def test_feasibility(data):
    m, canning_plants, markets, distances, capacities, demands = data
    m = Container()

    i = Set(m, name="i", records=["seattle", "san-diego"])
    j = Set(m, name="j", records=["new-york", "chicago", "topeka"])

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
        sense="feasibility",
    )
    assert (
        transport._generate_solve_string()
        == "solve transport using LP MIN transport_objective_variable"
    )
    transport.solve()
    assert x.records is not None

    pytest.raises(
        ValidationError,
        Model,
        m,
        "transport2",
        "",
        "LP",
        m.getEquations(),
        "feasibility",
        Sum((i, j), c[i, j] * x[i, j]),
    )


def test_tuple_equations(data):
    m, canning_plants, markets, distances, capacities, demands = data
    i = Set(
        m,
        name="i",
        records=canning_plants,
        description="Canning Plants",
    )
    j = Set(m, name="j", records=markets, description="Markets")

    # Params
    a = Parameter(m, name="a", domain=[i], records=capacities)
    b = Parameter(m, name="b", domain=[j], records=demands)
    d = Parameter(m, name="d", domain=[i, j], records=distances)
    c = Parameter(m, name="c", domain=[i, j])
    c[i, j] = 90 * d[i, j] / 1000

    x = Variable(m, name="x", domain=[i, j], type="Positive")
    z = Variable(m, name="z")

    # Equation definition without an index
    cost = Equation(
        m,
        name="cost",
        description="define objective function",
    )
    cost[...] = Sum((i, j), c[i, j] * x[i, j]) == z

    # Equation definition with an index
    supply = Equation(
        m,
        name="supply",
        domain=[i],
        description="observe supply limit at plant i",
    )
    supply[i] = Sum(j, x[i, j]) <= a[i]

    demand = Equation(m, name="demand", domain=[j])
    demand[j] = Sum(i, x[i, j]) >= b[j]

    test_model = Model(
        m,
        name="test_model",
        equations=(supply, demand),
        problem="LP",
        sense="min",
        objective=Sum((i, j), c[i, j] * x[i, j]),
    )
    test_model.solve()

    test_model2 = Model(
        m,
        name="test_model2",
        equations=set(m.getEquations()) - {cost},
        problem="LP",
        sense="min",
        objective=Sum((i, j), c[i, j] * x[i, j]),
    )
    test_model2.solve()


def test_computeInfeasibilities(data):
    m, canning_plants, markets, distances, capacities, demands = data
    m = Container()

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

    b[j] = 1.5 * b[j]
    transport.solve()

    infeasibilities = transport.computeInfeasibilities()
    columns = [
        "i",
        "level",
        "marginal",
        "lower",
        "upper",
        "scale",
        "infeasibility",
    ]
    assert list(infeasibilities.keys()) == [
        "supply",
        "x",
        "demand",
        "transport_objective",
        "transport_objective_variable",
    ]
    assert list(infeasibilities["supply"].columns) == columns
    assert infeasibilities["supply"].values.tolist() == [
        ["san-diego", 1000.0, 0.0, float("-inf"), 600.0, 1.0, 400.0]
    ]

    assert x.computeInfeasibilities().values.tolist() == [
        [
            "seattle",
            "new-york",
            -100.0,
            0.0,
            0.0,
            float("inf"),
            1.0,
            100.0,
        ]
    ]

    assert supply.computeInfeasibilities().values.tolist() == [
        ["san-diego", 1000.0, 0.0, float("-inf"), 600.0, 1.0, 400.0]
    ]

    all_infeasibilities = transport.computeInfeasibilities()
    assert list(all_infeasibilities.keys()) == [
        "supply",
        "x",
        "demand",
        "transport_objective",
        "transport_objective_variable",
    ]

    assert [
        elem.values.tolist() for elem in list(infeasibilities.values())
    ] == [
        [["san-diego", 1000.0, 0.0, -float("inf"), 600.0, 1.0, 400.0]],
        [
            [
                "seattle",
                "new-york",
                -100.0,
                0.0,
                0.0,
                float("inf"),
                1.0,
                100.0,
            ]
        ],
        [],
        [],
        [],
    ]


def test_equations(data):
    m, canning_plants, markets, distances, capacities, demands = data
    e = Equation(m, "e")
    e.l[...] = -10
    e.lo[...] = 5
    model = Model(
        m,
        "my",
        problem=Problem.LP,
        equations=[e],
        sense=Sense.FEASIBILITY,
    )

    with pytest.raises(ValidationError):
        model.solve()


def test_equation_listing(data):
    m, canning_plants, markets, distances, capacities, demands = data
    cont = Container()

    # Prepare data
    steel_plants = ["ahmsa", "fundidora", "sicartsa", "hylsa", "hylsap"]
    markets = ["mexico-df", "monterrey", "guadalaja"]
    commodities = [
        "pellets",
        "coke",
        "nat-gas",
        "electric",
        "scrap",
        "pig-iron",
        "sponge",
        "steel",
    ]
    final_products = ["steel"]
    intermediate_products = ["sponge", "pig-iron"]
    raw_materials = ["pellets", "coke", "nat-gas", "electric", "scrap"]
    processes = ["pig-iron", "sponge", "steel-oh", "steel-el", "steel-bof"]
    productive_units = [
        "blast-furn",
        "openhearth",
        "bof",
        "direct-red",
        "elec-arc",
    ]

    io_coefficients = pd.DataFrame(
        [
            ["pellets", "pig-iron", -1.58],
            ["pellets", "sponge", -1.38],
            ["coke", "pig-iron", -0.63],
            ["nat-gas", "sponge", -0.57],
            ["electric", "steel-el", -0.58],
            ["scrap", "steel-oh", -0.33],
            ["scrap", "steel-bof", -0.12],
            ["pig-iron", "pig-iron", 1.00],
            ["pig-iron", "steel-oh", -0.77],
            ["pig-iron", "steel-bof", -0.95],
            ["sponge", "sponge", 1.00],
            ["sponge", "steel-el", -1.09],
            ["steel", "steel-oh", 1.00],
            ["steel", "steel-el", 1.00],
            ["steel", "steel-bof", 1.00],
        ]
    )

    capacity_utilization = pd.DataFrame(
        [
            ["blast-furn", "pig-iron", 1.0],
            ["openhearth", "steel-oh", 1.0],
            ["bof", "steel-bof", 1.0],
            ["direct-red", "sponge", 1.0],
            ["elec-arc", "steel-el", 1.0],
        ]
    )

    capacities_of_units = pd.DataFrame(
        [
            ["blast-furn", "ahmsa", 3.25],
            ["blast-furn", "fundidora", 1.40],
            ["blast-furn", "sicartsa", 1.10],
            ["openhearth", "ahmsa", 1.50],
            ["openhearth", "fundidora", 0.85],
            ["bof", "ahmsa", 2.07],
            ["bof", "fundidora", 1.50],
            ["bof", "sicartsa", 1.30],
            ["direct-red", "hylsa", 0.98],
            ["direct-red", "hylsap", 1.00],
            ["elec-arc", "hylsa", 1.13],
            ["elec-arc", "hylsap", 0.56],
        ]
    )

    rail_distances = pd.DataFrame(
        [
            ["ahmsa", "mexico-df", 1204],
            ["ahmsa", "monterrey", 218],
            ["ahmsa", "guadalaja", 1125],
            ["ahmsa", "export", 739],
            ["fundidora", "mexico-df", 1017],
            ["fundidora", "guadalaja", 1030],
            ["fundidora", "export", 521],
            ["sicartsa", "mexico-df", 819],
            ["sicartsa", "monterrey", 1305],
            ["sicartsa", "guadalaja", 704],
            ["hylsa", "mexico-df", 1017],
            ["hylsa", "guadalaja", 1030],
            ["hylsa", "export", 521],
            ["hylsap", "mexico-df", 185],
            ["hylsap", "monterrey", 1085],
            ["hylsap", "guadalaja", 760],
            ["hylsap", "export", 315],
            ["import", "mexico-df", 428],
            ["import", "monterrey", 521],
            ["import", "guadalaja", 300],
        ]
    )

    product_prices = pd.DataFrame(
        [
            ["pellets", "domestic", 18.7],
            ["coke", "domestic", 52.17],
            ["nat-gas", "domestic", 14.0],
            ["electric", "domestic", 24.0],
            ["scrap", "domestic", 105.0],
            ["steel", "import", 150],
            ["steel", "export", 140],
        ]
    )

    demand_distribution = pd.DataFrame(
        [["mexico-df", 55], ["monterrey", 30], ["guadalaja", 15]]
    )

    dt = 5.209  # total demand for final goods in 1979
    rse = 40  # raw steel equivalence
    eb = 1.0  # export bound

    # Set
    i = Set(
        cont,
        name="i",
        records=pd.DataFrame(steel_plants),
        description="steel plants",
    )
    j = Set(
        cont,
        name="j",
        records=pd.DataFrame(markets),
        description="markets",
    )
    c = Set(
        cont,
        name="c",
        records=pd.DataFrame(commodities),
        description="commidities",
    )
    cf = Set(
        cont,
        name="cf",
        records=pd.DataFrame(final_products),
        domain=c,
        description="final products",
    )
    ci = Set(
        cont,
        name="ci",
        records=pd.DataFrame(intermediate_products),
        domain=c,
        description="intermediate products",
    )
    cr = Set(
        cont,
        name="cr",
        records=pd.DataFrame(raw_materials),
        domain=c,
        description="raw materials",
    )
    p = Set(
        cont,
        name="p",
        records=pd.DataFrame(processes),
        description="processes",
    )
    m = Set(
        cont,
        name="m",
        records=pd.DataFrame(productive_units),
        description="productive units",
    )

    # Data
    a = Parameter(
        cont,
        name="a",
        domain=[c, p],
        records=io_coefficients,
        description="input-output coefficients",
    )
    b = Parameter(
        cont,
        name="b",
        domain=[m, p],
        records=capacity_utilization,
        description="capacity utilization",
    )
    k = Parameter(
        cont,
        name="k",
        domain=[m, i],
        records=capacities_of_units,
        description="capacities of productive units",
    )
    dd = Parameter(
        cont,
        name="dd",
        domain=j,
        records=demand_distribution,
        description="distribution of demand",
    )
    d = Parameter(
        cont,
        name="d",
        domain=[c, j],
        description="demand for steel in 1979",
    )

    d["steel", j] = dt * (1 + rse / 100) * dd[j] / 100

    rd = Parameter(
        cont,
        name="rd",
        domain=["*", "*"],
        records=rail_distances,
        description="rail distances from plants to markets",
    )

    muf = Parameter(
        cont,
        name="muf",
        domain=[i, j],
        description="transport rate: final products",
    )
    muv = Parameter(
        cont, name="muv", domain=j, description="transport rate: imports"
    )
    mue = Parameter(
        cont, name="mue", domain=i, description="transport rate: exports"
    )

    muf[i, j] = (2.48 + 0.0084 * rd[i, j]).where[rd[i, j]]
    muv[j] = (2.48 + 0.0084 * rd["import", j]).where[rd["import", j]]
    mue[i] = (2.48 + 0.0084 * rd[i, "export"]).where[rd[i, "export"]]

    prices = Parameter(
        cont,
        name="prices",
        domain=[c, "*"],
        records=product_prices,
        description="product prices (us$ per unit)",
    )

    pdp = Parameter(cont, name="pd", domain=c, description="domestic prices")
    pv = Parameter(cont, name="pv", domain=c, description="import prices")
    pe = Parameter(cont, name="pe", domain=c, description="export prices")

    pdp[c] = prices[c, "domestic"]
    pv[c] = prices[c, "import"]
    pe[c] = prices[c, "export"]

    # Variable
    z = Variable(
        cont,
        name="z",
        domain=[p, i],
        type="Positive",
        description="process level",
    )
    x = Variable(
        cont,
        name="x",
        domain=[c, i, j],
        type="Positive",
        description="shipment of final products",
    )
    u = Variable(
        cont,
        name="u",
        domain=[c, i],
        type="Positive",
        description="purchase of domestic materials",
    )
    v = Variable(
        cont,
        name="v",
        domain=[c, j],
        type="Positive",
        description="imports",
    )
    e = Variable(
        cont,
        name="e",
        domain=[c, i],
        type="Positive",
        description="exports",
    )
    phipsi = Variable(cont, name="phipsi", description="raw material cost")
    philam = Variable(cont, name="philam", description="transport cost")
    phipi = Variable(cont, name="phipi", description="import cost")
    phieps = Variable(cont, name="phieps", description="export revenue")

    # Equation declaration
    mbf = Equation(
        cont,
        name="mbf",
        domain=[c, i],
        description="material balances: final products",
    )
    mbi = Equation(
        cont,
        name="mbi",
        domain=[c, i],
        description="material balances: intermediates",
    )
    mbr = Equation(
        cont,
        name="mbr",
        domain=[c, i],
        description="material balances: raw materials",
    )
    cc = Equation(
        cont,
        name="cc",
        domain=[m, i],
        description="capacity constraint",
    )
    mr = Equation(
        cont,
        name="mr",
        domain=[c, j],
        description="market requirements",
    )
    me = Equation(
        cont,
        name="me",
        domain=c,
        description="maximum export",
    )
    apsi = Equation(
        cont,
        name="apsi",
        description="accounting: raw material cost",
    )
    alam = Equation(
        cont,
        name="alam",
        description="accounting: transport cost",
    )
    api = Equation(cont, name="api", description="accounting: import cost")
    aeps = Equation(
        cont,
        name="aeps",
        description="accounting: export cost",
    )

    # Equation definition
    obj = phipsi + philam + phipi - phieps  # Total Cost

    mbf[cf, i] = Sum(p, a[cf, p] * z[p, i]) >= Sum(j, x[cf, i, j]) + e[cf, i]
    mbi[ci, i] = Sum(p, a[ci, p] * z[p, i]) >= 0
    mbr[cr, i] = Sum(p, a[cr, p] * z[p, i]) + u[cr, i] >= 0
    cc[m, i] = Sum(p, b[m, p] * z[p, i]) <= k[m, i]
    mr[cf, j] = Sum(i, x[cf, i, j]) + v[cf, j] >= d[cf, j]
    me[cf] = Sum(i, e[cf, i]) <= eb
    apsi[...] = phipsi == Sum((cr, i), pdp[cr] * u[cr, i])
    alam[...] = philam == Sum((cf, i, j), muf[i, j] * x[cf, i, j]) + Sum(
        (cf, j), muv[j] * v[cf, j]
    ) + Sum((cf, i), mue[i] * e[cf, i])
    api[...] = phipi == Sum((cf, j), pv[cf] * v[cf, j])
    aeps[...] = phieps == Sum((cf, i), pe[cf] * e[cf, i])

    mexss = Model(
        cont,
        name="mexss",
        equations=cont.getEquations(),
        problem="LP",
        sense=Sense.MIN,
        objective=obj,
    )

    mexss.solve(options=Options(equation_listing_limit=100))
    assert len(mexss.getEquationListing().split("\n")) == 74
    assert (
        len(mexss.getEquationListing(infeasibility_threshold=2.5).split("\n"))
        == 2
    )


def test_jupyter_behaviour(data):
    m, canning_plants, markets, distances, capacities, demands = data
    i = Set(m, name="i", records=canning_plants)
    i = Set(m, name="i", records=canning_plants)
    j = Set(m, name="j", records=markets)

    a = Parameter(m, name="a", domain=[i], records=capacities)
    a = Parameter(m, name="a", domain=[i], records=capacities)
    b = Parameter(m, name="b", domain=[j], records=demands)
    d = Parameter(m, name="d", domain=[i, j], records=distances)
    c = Parameter(m, name="c", domain=[i, j])
    c[i, j] = 90 * d[i, j] / 1000

    x = Variable(m, name="x", domain=[i, j], type="Positive")
    x = Variable(m, name="x", domain=[i, j], type="Positive")

    supply = Equation(m, name="supply", domain=[i])
    supply = Equation(m, name="supply", domain=[i])
    demand = Equation(m, name="demand", domain=[j])
    demand = Equation(m, name="demand", domain=[j])

    supply[i] = Sum(j, x[i, j]) <= a[i]
    supply[i] = Sum(j, x[i, j]) <= a[i]
    demand[j] = Sum(i, x[i, j]) >= b[j]
    demand[j] = Sum(i, x[i, j]) >= b[j]

    transport = Model(
        m,
        name="transport",
        equations=[supply, demand],
        problem="LP",
        sense="min",
        objective=Sum((i, j), c[i, j] * x[i, j]),
    )

    transport = Model(
        m,
        name="transport",
        equations=[supply, demand],
        problem="LP",
        sense="min",
        objective=Sum((i, j), c[i, j] * x[i, j]),
    )
    transport.solve()
    transport.solve()


def test_solve_string_lp(data):
    m, canning_plants, markets, distances, capacities, demands = data
    i = Set(m, name="i")
    j = Set(m, name="j")

    # Params
    a = Parameter(m, name="a", domain=[i])
    b = Parameter(m, name="b", domain=[j])
    c = Parameter(m, name="c", domain=[i, j])

    x = Variable(m, name="x", domain=[i, j], type="Positive")
    z = Variable(m, name="z")

    # Equation definition without an index
    cost = Equation(
        m,
        name="cost",
        description="define objective function",
    )
    cost[...] = Sum((i, j), c[i, j] * x[i, j]) == z

    # Equation definition with an index
    supply = Equation(
        m,
        name="supply",
        domain=[i],
        description="observe supply limit at plant i",
    )
    supply[i] = Sum(j, x[i, j]) <= a[i]

    demand = Equation(m, name="demand", domain=[j])
    demand[j] = Sum(i, x[i, j]) >= b[j]

    test_model = Model(
        m,
        name="test_model",
        equations=[supply, demand],
        problem="LP",
        sense="min",
        objective=z,
    )
    assert (
        test_model._generate_solve_string()
        == "solve test_model using LP MIN z"
    )


def test_solve_string_mcp(data):
    m, canning_plants, markets, distances, capacities, demands = data
    c = Set(m, "c")
    h = Set(m, "h")
    s = Set(m, "s")

    cc = Alias(m, "cc", c)

    e = Parameter(m, "e", domain=[c, h])
    esub = Parameter(m, "esub", domain=h)

    alpha = Parameter(m, "alpha", domain=[c, h])
    a = Parameter(m, "a", domain=[c, s])

    p = Variable(m, "p", type=VariableType.POSITIVE, domain=c)
    y = Variable(m, "y", type=VariableType.POSITIVE, domain=s)
    i = Variable(m, "i", type=VariableType.POSITIVE, domain=h)

    mkt = Equation(m, "mkt", domain=c, description="commodity market")
    profit = Equation(m, "profit", domain=s, description="zero profit")
    income = Equation(m, "income", domain=h, description="income index")

    mkt[c] = Sum(s, a[c, s] * y[s]) + Sum(h, e[c, h]) >= Sum(
        h.where[esub[h] != 1],
        (i[h] / Sum(cc, alpha[cc, h] * p[cc] ** (1 - esub[h])))
        * alpha[c, h]
        * (1 / p[c]) ** esub[h],
    ) + Sum(h.where[esub[h] == 1], i[h] * alpha[c, h] / p[c])

    profit[s] = -Sum(c, a[c, s] * p[c]) >= 0
    income[h] = i[h] >= Sum(c, p[c] * e[c, h])

    hansen = Model(
        m,
        "hansen",
        problem=Problem.MCP,
        matches={mkt: p, profit: y, income: i},
    )
    assert hansen._generate_solve_string() == "solve hansen using MCP"

    # Test new mcp matching syntax

    # One equation to many variables
    hansen = Model(
        m,
        "hansen2",
        problem=Problem.MCP,
        matches={mkt: (p, y, i)},
    )

    assert hansen.getDeclaration() == "Model hansen2 / mkt:(p|y|i) /;"

    # Many equations to one variable
    hansen = Model(
        m,
        "hansen2",
        problem=Problem.MCP,
        matches={(mkt, profit, income): i},
    )

    assert (
        hansen.getDeclaration() == "Model hansen2 / (mkt|profit|income):i /;"
    )

    # Many to many should fail for now
    with pytest.raises(TypeError):
        hansen = Model(
            m,
            "hansen4",
            problem=Problem.MCP,
            matches={(mkt, profit, income): (p, y, i)},
        )

    # Non-dict matches should fail
    with pytest.raises(TypeError):
        hansen = Model(
            m,
            "hansen4",
            problem=Problem.MCP,
            matches=(i, income),
        )


def test_solve_string_cns(data):
    m, canning_plants, markets, distances, capacities, demands = data
    x = Variable(m, "x")
    f = Equation(m, "f")
    f[...] = x * x == 4
    x.l = 1
    m = Model(m, name="m", equations=[f], problem="CNS", sense="FEASIBILITY")
    assert m._generate_solve_string() == "solve m using CNS"


def test_models_with_same_name():
    import gamspy as gp

    m = gp.Container()

    slot = gp.Set(
        m, "slot", records=list("abcdefghijklmnopqrs"), description="position"
    )
    t = gp.Set(m, "t", records=[f"t{i}" for i in range(1, 20)])

    tiles = gp.Parameter(
        m, "tiles", domain=t, description="numerical value of tile"
    )

    tiles[t] = gp.Ord(t)

    axis = gp.Set(
        m, "axis", records=[f"ax{i}" for i in range(1, 16)], description="axis"
    )

    data = [
        ("ax1", list("abc")),
        ("ax2", list("defg")),
        ("ax3", list("hijkl")),
        ("ax4", list("mnop")),
        ("ax5", list("qrs")),
        ("ax6", list("cgl")),
        ("ax7", list("bfkp")),
        ("ax8", list("aejos")),
        ("ax9", list("dinr")),
        ("ax10", list("hmq")),
        ("ax11", list("lps")),
        ("ax12", list("gkor")),
        ("ax13", list("cfjnq")),
        ("ax14", list("beim")),
        ("ax15", list("adh")),
    ]
    axis_sum = gp.Set(
        m,
        "axis_sum",
        domain=[axis, slot],
        records=[(ax, s) for ax, slt in data for s in slt],
        description="axes to sum",
    )

    corner = gp.Set(
        m,
        "corner",
        domain=[slot],
        records=list("achlqs"),
        description="corners",
    )

    z = gp.Variable(m, "z", "free")
    x = gp.Variable(m, "x", "binary", domain=[slot, t])
    v = gp.Variable(m, "v", "free", domain=[slot])

    obj = gp.Equation(m, "obj")
    obj[...] = z == 1

    axessum = gp.Equation(m, "axessum", domain=[axis])
    axessum[axis] = gp.Sum(axis_sum[axis, slot], v[slot]) == 38

    onetileperslot = gp.Equation(m, "onetileperslot", domain=[slot])
    onetileperslot[slot] = gp.Sum(t, x[slot, t]) == 1

    useall = gp.Equation(m, "useall", domain=[t])
    useall[t] = gp.Sum(slot, x[slot, t]) == 1

    vdef = gp.Equation(m, "vdef", domain=[slot])
    vdef[slot] = gp.Sum(t, x[slot, t] * tiles[t]) == v[slot]

    rotations = gp.Equation(m, "rotations", domain=[slot])
    rotations[corner[slot]].where[~gp.math.same_as(slot, "a")] = (
        v[slot] >= v["a"] + 1
    )

    reflect = gp.Equation(m, "reflect")
    reflect[...] = v["h"] == v["c"] + 1

    magic = gp.Model(
        m,
        name="hexagon",
        equations=[obj, axessum, onetileperslot, useall, vdef],
        problem="MIP",
        sense="min",
        objective=z,
    )

    df_magic = magic.solve()

    magic_unique = gp.Model(
        m,
        name="hexagon",
        equations=m.getEquations(),
        problem="MIP",
        sense="min",
        objective=z,
    )

    df_magic_unique = magic_unique.solve()

    assert df_magic["Model Status"].tolist()[0] == "OptimalGlobal"
    assert df_magic_unique["Model Status"].tolist()[0] == "IntegerInfeasible"


def test_models_with_many_equations():
    # Number of equations must be too many that the model declaration exceeds the 80000 line length limit of GAMS.
    m = Container()
    for _ in range(2300):
        name = "e" + str(uuid.uuid4()).replace("-", "_")[:35]
        _ = Equation(m, name=name)

    # This should not fail.
    _ = Model(m, "my_model", equations=m.getEquations())
    m.close()
