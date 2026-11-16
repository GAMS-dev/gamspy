from __future__ import annotations

import math
import os
import tempfile
import uuid
import warnings
from pathlib import Path

import pandas as pd
import pytest

import gamspy as gp
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
from gamspy._backend.local import LocalSolveHandle
from gamspy._model import FileFormat
from gamspy._options import ConvertOptions
from gamspy.exceptions import GamspyException, ValidationError

pytestmark = pytest.mark.unit


def test_model(transport):
    m, canning_plants, markets, distances, capacities, demands = transport
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

    with pytest.raises(
        TypeError, match=r"Container must of type `Container` but found <class 'str'>"
    ):
        Model(container="invalidtype")

    with pytest.raises(
        ValidationError, match=r"Model is missing required argument `container`."
    ):
        Model(name="bla")

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

    # Test convert with rename
    with tempfile.TemporaryDirectory() as tmpdir:
        test_model.convert(
            tmpdir,
            file_format=FileFormat.GDXJacobian,
            options=ConvertOptions(GDXNames=0, GDXJacobian="my_jacobian.gdx"),
        )
        assert os.path.exists(os.path.join(tmpdir, "my_jacobian.gdx"))

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
            assert "$if not set jacfile $set jacfile jacobian.gdx" in file.read()

    # Test Path
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir) / "folder with blank"
        test_model.convert(
            tmpdir_path,
            file_format=[
                FileFormat.GAMSJacobian,
                FileFormat.GDXJacobian,
                FileFormat.GAMSPyJacobian,
                FileFormat.GAMSDict,
                FileFormat.GAMSDictMap,
            ],
            options=ConvertOptions(GDXNames=0),
        )
        assert os.path.exists(tmpdir_path / "jacobian.gms")
        assert os.path.exists(tmpdir_path / "jacobian.gdx")
        assert os.path.exists(tmpdir_path / "jacobian.py")
        assert os.path.exists(tmpdir_path / "dict.txt")
        assert os.path.exists(tmpdir_path / "dictmap.gdx")

        with open(tmpdir_path / "jacobian.gms") as file:
            assert "$if not set jacfile $set jacfile jacobian.gdx" in file.read()

    # Test multiple rename
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        test_model.convert(
            tmpdir_path,
            file_format=[
                FileFormat.GAMSJacobian,
                FileFormat.GDXJacobian,
                FileFormat.GAMSPyJacobian,
                FileFormat.GAMSDict,
                FileFormat.GAMSDictMap,
            ],
            options=ConvertOptions(
                GDXNames=0,
                GAMSDict="my_dict.txt",
                GAMSPyJacobian="my_jacobian.py",
                GAMSJacobian="my_jacobian.gms",
            ),
        )
        assert os.path.exists(tmpdir_path / "my_jacobian.gms")
        assert os.path.exists(tmpdir_path / "jacobian.gdx")
        assert os.path.exists(tmpdir_path / "my_jacobian.py")
        assert os.path.exists(tmpdir_path / "my_dict.txt")
        assert os.path.exists(tmpdir_path / "dictmap.gdx")

    # Check if the name is reserved
    with pytest.raises(ValidationError):
        Model(m, "set", "", "LP")

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
    assert test_model5.getDeclaration() == "Model test_model5 / supply.x,cost.z /;"

    # Equations provided as strings
    with pytest.raises(ValueError):
        Model(m, "test_model5", "", "LP", ["cost", "supply"])

    # Test matches
    test_model6 = Model(
        m,
        name="test_model6",
        equations=[supply],
        matches={demand: x},
        problem="LP",
        sense="min",
    )
    assert test_model6.getDeclaration() == "Model test_model6 / supply,demand.x /;"

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
    with pytest.raises(ValueError):
        Model(
            m,
            "",
            "test_model7",
            "",
            m.getEquations(),
            "min",
            Sum((i, j), c[i, j] * x[i, j]),
        )

    # model name too long
    with pytest.raises(ValidationError):
        Model(
            m,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "",
            "LP",
            m.getEquations(),
            "min",
            Sum((i, j), c[i, j] * x[i, j]),
        )

    # model name is not an str
    with pytest.raises(TypeError):
        Model(m, 5, "", "LP", m.getEquations(), "min", Sum((i, j), c[i, j] * x[i, j]))

    # model name contains empty space
    with pytest.raises(ValidationError):
        Model(
            m,
            "test_model 8",
            "",
            "LP",
            m.getEquations(),
            "min",
            Sum((i, j), c[i, j] * x[i, j]),
        )

    # model name begins with underscore
    with pytest.raises(ValidationError):
        Model(
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


def test_feasibility(transport):
    m, distances, capacities, demands = (
        transport.container,
        transport.distances,
        transport.capacities,
        transport.demands,
    )
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

    with pytest.raises(ValidationError):
        Model(
            m,
            "transport2",
            "",
            "LP",
            m.getEquations(),
            "feasibility",
            Sum((i, j), c[i, j] * x[i, j]),
        )


def test_tuple_equations(transport):
    m, canning_plants, markets, distances, capacities, demands = transport
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


def test_computeInfeasibilities(transport):
    m, canning_plants, markets, distances, capacities, demands = transport
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

    assert [elem.values.tolist() for elem in list(infeasibilities.values())] == [
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


def test_equations(transport):
    m = transport.container
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


def test_equation_listing(transport):
    m, markets = transport.container, transport.markets
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
    muv = Parameter(cont, name="muv", domain=j, description="transport rate: imports")
    mue = Parameter(cont, name="mue", domain=i, description="transport rate: exports")

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
    assert len(mexss.getEquationListing(infeasibility_threshold=2.5).split("\n")) == 2


def test_jupyter_behaviour(transport):
    m, canning_plants, markets, distances, capacities, demands = transport
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


def test_solve_string_lp(transport):
    m = transport.container
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
    assert test_model._generate_solve_string() == "solve test_model using LP MIN z"


def test_solve_string_mcp(transport):
    m = transport.container
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

    assert hansen.getDeclaration() == "Model hansen2 / (mkt|profit|income):i /;"

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


def test_solve_string_cns(transport):
    m = transport.container
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

    tiles = gp.Parameter(m, "tiles", domain=t, description="numerical value of tile")

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
    rotations[corner[slot]].where[~gp.math.same_as(slot, "a")] = v[slot] >= v["a"] + 1

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

    magic_unique = gp.Model(
        m,
        name="hexagon",
        equations=m.getEquations(),
        problem="MIP",
        sense="min",
        objective=z,
    )

    df_magic = magic.solve()
    df_magic_unique = magic_unique.solve()

    assert df_magic["Model Status"].tolist()[0] == "IntegerInfeasible"
    assert df_magic_unique["Model Status"].tolist()[0] == "IntegerInfeasible"


def test_models_with_many_equations():
    # Number of equations must be too many that the model declaration exceeds the 80000 line length limit of GAMS.
    m = Container()
    for _ in range(2300):
        name = "e" + str(uuid.uuid4()).replace("-", "_")[:35]
        _ = Equation(m, name=name)

    # This should not fail.
    _ = Model(m, "my_model", equations=m.getEquations())


def test_mpsge_equation_definition():
    m = gp.Container()

    N = 5
    lb = 0.5
    ub = float("inf")
    assert lb <= ub
    # ub = 1.5
    i = gp.Set(m, records=range(N))

    x = gp.Variable(m, domain=i)
    x.lo[...] = lb
    x.up[...] = ub

    f = gp.Equation(m, domain=i)
    g = gp.Equation(m, domain=i, type=gp.EquationType.NONBINDING)
    h = gp.Equation(
        m, domain=i, type="nonbinding", definition=gp.math.exp(x[i]) - gp.Ord(i)
    )

    f[i] = gp.math.exp(x[i]) >= gp.Ord(i)
    g[i] = gp.math.exp(x[i]) != gp.Ord(i)

    mcpf = gp.Model(m, problem=gp.Problem.MCP, matches={f: x})
    mcph = gp.Model(m, problem=gp.Problem.MCP, matches={h: x})

    x.l[i] = 1

    mcpf.solve()

    # check correctness
    correct_result = [min(max(lb, math.log(j + 1)), ub) for j in range(N)]

    xx = x.records.level.to_list()
    print("solution x:", xx)
    delta = [(x - y) for x, y in zip(correct_result, xx, strict=True)]
    print("delta:", delta)

    for expected, found in zip(correct_result, xx, strict=True):
        assert math.isclose(expected, found, rel_tol=1e-6, abs_tol=1e-8)

    mcph.solve()


def test_length():
    m = gp.Container()
    i = gp.Set(m, records=["i1", "i2"])
    assert len(i) == 2

    j = gp.Alias(m, alias_with=i)
    assert len(j) == 2

    a = gp.Parameter(m, domain=i)
    assert len(a) == 0
    a.generateRecords()
    assert len(a) == 2
    assert len(a[i].where[gp.Ord(i) > 1]) == 1
    assert len(gp.Sum(i, a[i]).records) == 1

    b = gp.Variable(m, domain=i)
    assert len(b) == 0
    b.generateRecords()
    assert len(b) == 2
    assert len(b.l) == 2
    assert len(b.l["i1"]) == 1
    assert len(b["i1"].l) == 1

    c = gp.Equation(m, domain=i)
    assert len(c) == 0
    c.generateRecords()
    assert len(c) == 2
    assert len(c.l) == 2
    assert len(c.l["i1"]) == 1
    assert len(c["i1"].l) == 1


def test_reassigned_equation_exception():
    m = gp.Container()
    x = gp.Variable(m, "x")
    e = gp.Equation(m, "e")
    e = x == 25
    with pytest.raises(
        TypeError,
        match=r"Each given equation in `equations` argument must be a type of Equation but found an expression:",
    ):
        gp.Model(m, equations=[e])


def test_grid_solve_validations():
    m = gp.Container()
    x = gp.Variable(m, "x")
    e = gp.Equation(m, "e", definition=x >= 5)
    model = gp.Model(
        m, "grid_model", equations=[e], problem="LP", sense="MIN", objective=x
    )

    handle = LocalSolveHandle(model, 123, "cplex", gp.Options())
    assert isinstance(handle, gp.SolveHandle)
    assert handle.backend == "local"
    assert "job=123" in repr(handle)

    with pytest.raises(ValidationError, match=r"`timeout` must be nonnegative"):
        handle.wait(timeout=-1)

    # A collected handle cannot be waited for or polled again.
    handle._is_collected = True
    with pytest.raises(ValidationError, match=r"already been collected"):
        handle.wait()

    with pytest.raises(ValidationError, match=r"already been collected"):
        handle.isReady()

    m.close()


def test_async_client_deprecation():
    with pytest.warns(DeprecationWarning, match=r"`is_blocking=False` is deprecated"):
        gp.EngineClient(host="https://engine.gams.com/api/", is_blocking=False)

    with pytest.warns(DeprecationWarning, match=r"`is_blocking=False` is deprecated"):
        gp.NeosClient(email="test@test.com", is_blocking=False)

    # The blocking default stays silent.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        gp.EngineClient(host="https://engine.gams.com/api/")
        gp.NeosClient(email="test@test.com")


def test_model_attributes():
    m = gp.Container()
    x = gp.Variable(m, "x")
    e = gp.Equation(m, "e", definition=x == 25)
    model = gp.Model(m, "dummy", equations=[e])
    with pytest.raises(ValidationError, match=r"The model is not solved yet"):
        _ = model.num_domain_violations

    with pytest.raises(ValidationError, match=r"The model is not solved yet"):
        _ = model.algorithm_time

    with pytest.raises(ValidationError, match=r"The model is not solved yet"):
        _ = model.total_solve_time

    with pytest.raises(ValidationError, match=r"The model is not solved yet"):
        _ = model.total_solver_time

    with pytest.raises(ValidationError, match=r"The model is not solved yet"):
        _ = model.num_iterations

    with pytest.raises(ValidationError, match=r"The model is not solved yet"):
        _ = model.marginals

    with pytest.raises(ValidationError, match=r"The model is not solved yet"):
        _ = model.max_infeasibility

    with pytest.raises(ValidationError, match=r"The model is not solved yet"):
        _ = model.mean_infeasibility

    with pytest.raises(ValidationError, match=r"The model is not solved yet"):
        _ = model.num_equations

    with pytest.raises(ValidationError, match=r"The model is not solved yet"):
        _ = model.num_nonzeros

    with pytest.raises(ValidationError, match=r"The model is not solved yet"):
        _ = model.num_variables

    with pytest.raises(ValidationError, match=r"The model is not solved yet"):
        _ = model.objective_value

    with pytest.raises(ValidationError, match=r"The model is not solved yet"):
        _ = model.model_generation_time

    with pytest.raises(ValidationError, match=r"The model is not solved yet"):
        _ = model.solve_model_time

    with pytest.raises(ValidationError, match=r"The model is not solved yet"):
        _ = model.solve_status

    with pytest.raises(ValidationError, match=r"The model is not solved yet"):
        _ = model.solver_version


def test_model_instance():
    m = gp.Container()
    i = gp.Set(m, "i", records=["i1", "i2"])
    p = gp.Parameter(m, "p", domain=i, records=[("i1", 2), ("i2", 3)])
    x = gp.Variable(m, "x", domain=i, type="positive")
    y = gp.Variable(m, "y", type="integer")
    z = gp.Variable(m, "z")
    e = gp.Equation(m, "e", domain=i)
    e[i] = p[i] * x[i] + gp.math.sqr(z) >= 5
    e2 = gp.Equation(m, "e2", definition=y <= 4)
    model = gp.Model(
        m,
        "mod",
        equations=[e, e2],
        problem="MINLP",
        sense="min",
        objective=Sum(i, x[i]) + z + y,
    )
    z.l = 1.5

    instance = model.getModelInstance()
    assert instance.objective_variable == "mod_objective_variable"
    assert instance.equations.to_dict("list") == {
        "name": ["e(i1)", "e(i2)", "e2", "mod_objective"],
        "type": ["=G=", "=G=", "=L=", "=E="],
        "rhs": [5.0, 5.0, 4.0, 0.0],
        "nonlinear": [True, True, False, False],
    }
    assert instance.variables.to_dict("list") == {
        "name": ["x(i1)", "x(i2)", "y", "z", "mod_objective_variable"],
        "type": ["continuous", "continuous", "integer", "continuous", "continuous"],
        "lower": [0.0, 0.0, 0.0, -math.inf, -math.inf],
        "upper": [math.inf] * 5,
        "level": [0.0, 0.0, 0.0, 1.5, 0.0],
    }
    jacobian = instance.jacobian
    assert len(jacobian) == 10
    assert jacobian[jacobian.equation == "e(i1)"].to_dict("list") == {
        "equation": ["e(i1)", "e(i1)"],
        "variable": ["x(i1)", "z"],
        "coefficient": [2.0, 3.0],
        "nonlinear": [False, True],
    }

    with pytest.raises(ValidationError, match=r"The model is not solved yet"):
        _ = model.objective_value

    p["i1"] = 10
    jacobian = model.getModelInstance().jacobian
    assert jacobian.loc[0, "coefficient"] == 10.0

    with pytest.raises(TypeError, match="`options` must be of type Option"):
        model.getModelInstance(options={"time_limit": 5})

    with pytest.raises(TypeError, match="`options` must be of type Option"):
        model.getModelInstance(options=gp.FreezeOptions())


def test_model_instance_mcp():
    m = gp.Container()
    v = gp.Variable(m, "v", type="positive")
    f = gp.Equation(m, "f", definition=v >= gp.Number(1))
    model = gp.Model(m, "mcp", problem="MCP", matches={f: v})

    instance = model.getModelInstance()
    assert instance.objective_variable is None
    assert instance.indicators.empty
    assert instance.jacobian.to_dict("list") == {
        "equation": ["f"],
        "variable": ["v"],
        "coefficient": [1.0],
        "nonlinear": [False],
    }


def test_model_instance_indicators():
    m = gp.Container()
    i = gp.Set(m, "i", records=["i1", "i2"])
    j = gp.Set(m, "j", records=["j1", "j2"])
    x = gp.Variable(m, "x", domain=[i, j], type="positive")
    b = gp.Variable(m, "b", domain=i, type="binary")
    off = gp.formulations.indicator(b[i], 0, x[i, j] <= 0, native=True)
    on = gp.formulations.indicator(b[i], 1, x[i, j] >= 2, native=True)
    model = gp.Model(
        m,
        "mod",
        equations=m.getEquations(),
        problem="MIP",
        sense="min",
        objective=Sum([i, j], x[i, j]),
    )

    indicators = model.getModelInstance().indicators
    off_name = off.equations_created["indicator"].name
    on_name = on.equations_created["indicator"].name
    assert indicators.to_dict("list") == {
        "equation": [
            f"{off_name}(i1,j1)",
            f"{off_name}(i1,j2)",
            f"{off_name}(i2,j1)",
            f"{off_name}(i2,j2)",
            f"{on_name}(i1,j1)",
            f"{on_name}(i1,j2)",
            f"{on_name}(i2,j1)",
            f"{on_name}(i2,j2)",
        ],
        "variable": ["b(i1)", "b(i1)", "b(i2)", "b(i2)"] * 2,
        "value": [0] * 4 + [1] * 4,
    }

    b.fx["i1"] = 1
    with pytest.raises(ValidationError, match="hold_fixed_variables"):
        model.getModelInstance(gp.Options(hold_fixed_variables=True))


def test_model_instance_frozen():
    m = gp.Container()
    i = gp.Set(m, "i", records=["i1", "i2"])
    q = gp.Parameter(m, "q", domain=i, records=[("i1", 1), ("i2", 2)])
    w = gp.Variable(m, "w", domain=i)
    g = gp.Equation(m, "g", domain=i)
    g[i] = w[i] >= q[i]
    model = gp.Model(
        m, "lp", equations=[g], problem="LP", sense="min", objective=Sum(i, w[i])
    )
    model.freeze(modifiables=[q])

    assert model.getModelInstance().equations["rhs"].tolist() == [1.0, 2.0, 0.0]

    q.setRecords([("i1", 7), ("i2", 7)])
    model.solve()
    assert model.objective_value == 14.0
    model.unfreeze()


def test_feasibility_relaxation(transport):
    m = transport.container
    i = Set(m, "i", records=transport.canning_plants)
    j = Set(m, "j", records=transport.markets)
    a = Parameter(m, "a", domain=i, records=transport.capacities)
    b = Parameter(m, "b", domain=j, records=transport.demands)
    d = Parameter(m, "d", domain=[i, j], records=transport.distances)
    b["new-york"] = 400
    x = Variable(m, "x", domain=[i, j], type="positive")
    supply = Equation(m, "supply", domain=i)
    supply[i] = Sum(j, x[i, j]) <= a[i]
    demand = Equation(m, "demand", domain=j)
    demand[j] = Sum(i, x[i, j]) >= b[j]
    model = Model(
        m,
        "transport",
        equations=[supply, demand],
        problem="LP",
        sense="min",
        objective=Sum((i, j), d[i, j] * x[i, j]),
    )
    definition = demand.getDefinition()
    equations = [equation.name for equation in model.equations]

    relax = model.feasibilityRelaxation(equations=[demand])
    assert relax.slacks[demand].positive.name == "demand_slack"
    assert relax.model.name == "transport_feasrelax"
    assert (
        model.feasibilityRelaxation().slacks[demand].positive.name.startswith("slack_")
    )
    assert [equation.name for equation in model.equations] == equations
    assert demand.getDefinition() == definition
    assert list(relax.slacks) == [demand]
    assert relax.model.equations[0] is supply
    assert len(relax.model.equations) == 3
    with pytest.raises(ValidationError, match="must be solved"):
        relax.violations()

    relax.model.solve()
    assert relax.violation.toValue() == 25

    relax.solve()
    assert relax.violation.records.upper.item() == math.inf
    assert relax.slacks[demand].negative is None
    assert relax.slacks[demand].positive.toList() == [
        ("new-york", 25.0),
        ("chicago", 0.0),
        ("topeka", 0.0),
    ]
    assert relax.violations()["demand"].values.tolist() == [["new-york", -25.0]]
    assert model.status == gp.ModelStatus.NotSolved


def test_feasibility_relaxation_objective_variable(transport):
    m = transport.container
    i = Set(m, "i", records=transport.canning_plants)
    j = Set(m, "j", records=transport.markets)
    a = Parameter(m, "a", domain=i, records=transport.capacities)
    b = Parameter(m, "b", domain=j, records=transport.demands)
    d = Parameter(m, "d", domain=[i, j], records=transport.distances)
    b["new-york"] = 400
    x = Variable(m, "x", domain=[i, j], type="positive")
    z = Variable(m, "z")
    supply = Equation(m, "supply", domain=i)
    supply[i] = Sum(j, x[i, j]) <= a[i]
    demand = Equation(m, "demand", domain=j)
    demand[j] = Sum(i, x[i, j]) >= b[j]
    cost = Equation(m, "cost", definition=z == Sum((i, j), d[i, j] * x[i, j]))
    model = Model(
        m,
        "transport",
        equations=[supply, demand, cost],
        problem="LP",
        sense="min",
        objective=z,
    )

    relax = model.feasibilityRelaxation()
    assert list(relax.slacks) == [supply, demand, cost]

    relax.solve()
    assert relax.slacks[cost].positive.toValue() == 0
    assert relax.slacks[cost].negative.toValue() == 0
    assert z.toValue() == pytest.approx(relax.optimality_model.objective_value)
    violations = relax.violations()
    assert list(violations) == ["demand"]
    assert violations["demand"].values.tolist() == [["new-york", -25.0]]


def test_feasibility_relaxation_problems():
    m = Container()
    x = Variable(m, "x")
    x.lo = 0
    x.up = 1
    e = Equation(m, "e", definition=gp.math.sqr(x) == 4)
    cns = Model(m, equations=[e], problem="CNS")
    relax = cns.feasibilityRelaxation(weights={e: Parameter(m, "w", records=2)})
    assert relax.model.problem == Problem.NLP
    assert relax.optimality_model is None

    relax.solve()
    assert relax.violation.toValue() == pytest.approx(6)
    assert relax.violations()["e"].violation.item() == pytest.approx(-3)

    i = Set(m, "i", records=["a", "b"])
    j = Alias(m, "j", i)
    y = Variable(m, "y", domain=i)
    g = Equation(m, "g", domain=i)
    g[j] = y[j] >= 1
    k = Set(m, "k", domain=i, records=["a"])
    limited = Model(m, equations=[g], problem="LP", limited_variables=[y[k]])
    relax = limited.feasibilityRelaxation()
    relax.solve()
    assert relax.violations()["g"].values.tolist() == [["b", -1.0]]

    hard = Equation(m, "hard", definition=x >= 2)
    soft = Equation(m, "soft", definition=x <= 0)
    model = Model(m, equations=[hard, soft], problem="LP", sense="min", objective=x)
    relax = model.feasibilityRelaxation(equations=[soft])
    relax.solve()
    assert relax.optimality_model.status == gp.ModelStatus.NotSolved
    assert relax.violation.records.upper.item() == math.inf


def test_feasibility_relaxation_conditions():
    m = Container()
    i = Set(m, "i", records=["i1", "i2", "i3"])
    k = Set(m, "k", records=["k1", "k2"])
    p = Parameter(m, "p", domain=i, records=[("i1", 5), ("i2", 1)])
    w = Parameter(m, "w", domain=[i, k], records=[("i1", "k1", 1), ("i2", "k1", 10)])
    x = Variable(m, "x", domain=i)
    x.lo[i] = 2
    x.up[i] = 4
    e = Equation(m, "e", domain=[i, k])
    e[i, "k1"].where[p[i] > 0] = x[i] == p[i]
    model = Model(m, equations=[e], problem="LP")

    relax = model.feasibilityRelaxation(weights={e: w})
    assert relax.optimality_model is None

    relax.model.solve()
    assert relax.violation.toValue() == 11
    assert relax.slacks[e].positive.toList() == [("i1", "k1", 0.0), ("i2", "k1", 1.0)]
    assert relax.slacks[e].negative.toList() == [("i1", "k1", 1.0), ("i2", "k1", 0.0)]
    assert relax.violations()["e"].values.tolist() == [
        ["i1", "k1", -1.0],
        ["i2", "k1", 1.0],
    ]
    assert relax.model.num_variables == 7
    assert relax.model.num_equations == 3


def test_feasibility_relaxation_scalar():
    m = Container()
    c = Parameter(m, "c", records=1)
    x = Variable(m, "x")
    lo = Equation(m, "lo")
    lo.where[c > 0] = x >= 3
    up = Equation(m, "up", definition=x <= 1)
    nb = Equation(m, "nb", type="nonbinding", definition=x == 100)
    model = Model(m, equations=[lo, up, nb], problem="LP", sense="min", objective=x)

    relax = model.feasibilityRelaxation(weights={up: 2})
    assert list(relax.slacks) == [lo, up]
    assert nb in relax.model.equations

    relax.model.solve()
    assert relax.violation.toValue() == 2

    relax.violation.up = relax.violation.l
    relax.optimality_model.solve()
    assert x.toValue() == 1
    assert relax.slacks[lo].positive.toValue() == 2


def test_feasibility_relaxation_measures():
    m = Container()
    x = Variable(m, "x")
    lo = Equation(m, "lo", definition=x >= 3)
    up = Equation(m, "up", definition=x <= 1)
    up2 = Equation(m, "up2", definition=x <= 2.5)
    model = Model(m, equations=[lo, up, up2], problem="LP")

    squares = model.feasibilityRelaxation(equations=[lo, up], measure="squares")
    assert squares.model.problem == Problem.QCP
    squares.model.solve()
    assert squares.slacks[lo].positive.toValue() == pytest.approx(1)
    assert squares.slacks[up].positive.toValue() == pytest.approx(1)

    count = model.feasibilityRelaxation(measure="count")
    assert count.model.problem == Problem.MIP
    count.model.solve()
    assert count.violation.toValue() == 1
    assert count.slacks[up].positive.toValue() == 0
    assert count.slacks[up2].positive.toValue() == 0
    assert list(count.violations()) == ["lo"]

    # big-M links each binary with one equation instead of two
    big = model.feasibilityRelaxation(measure="count", big_m=10)
    assert len(big.model.equations) == len(count.model.equations) - 3
    big.model.solve()
    assert big.violation.toValue() == 1
    assert list(big.violations()) == ["lo"]

    # lo needs a violation of 2, so a big-M of 1 forces a second violated row
    small = model.feasibilityRelaxation(measure="count", big_m={lo: 1, up: 10})
    small.model.solve()
    assert small.violation.toValue() == 2

    i = Set(m, "i", records=["i1", "i2", "i3"])
    k = Set(m, "k", records=["k1", "k2"])
    p = Parameter(m, "p", domain=i, records=[("i1", 5), ("i2", 1)])
    w = Parameter(m, "w", domain=[i, k], records=[("i1", "k1", 1), ("i2", "k1", 10)])
    y = Variable(m, "y", domain=i)
    y.lo[i] = 2
    y.up[i] = 4
    e = Equation(m, "e", domain=[i, k])
    e[i, "k1"].where[p[i] > 0] = y[i] == p[i]
    indexed = Model(m, equations=[e], problem="LP")
    count = indexed.feasibilityRelaxation(weights={e: w}, measure="count")
    count.model.solve()
    assert count.violation.toValue() == 11
    assert len(count.violations()["e"]) == 2

    bound = Parameter(
        m, "bound", domain=[i, k], records=[("i1", "k1", 2), ("i2", "k1", 2)]
    )
    big = indexed.feasibilityRelaxation(
        weights={e: w}, measure="count", big_m={e: bound}
    )
    big.model.solve()
    assert big.violation.toValue() == 11
    assert len(big.violations()["e"]) == 2

    with pytest.raises(ValidationError, match="measure"):
        model.feasibilityRelaxation(measure="max")

    with pytest.raises(ValidationError, match="big_m"):
        model.feasibilityRelaxation(big_m=10)

    with pytest.raises(ValidationError, match="Big-M given for `up2`"):
        model.feasibilityRelaxation(equations=[lo], measure="count", big_m={up2: 1})

    relaxed = Model(m, equations=[lo, up], problem="RMIP")
    with pytest.raises(ValidationError, match="relaxed problem type"):
        relaxed.feasibilityRelaxation(measure="count")


def _exp_sqr(y):
    return gp.math.exp(gp.math.sqr(y)) - 1


@pytest.mark.parametrize(
    ("problem", "relaxed", "term"),
    [
        ("LP", ("LP", "QCP", "MIP"), lambda y: 0),
        ("MIP", ("MIP", "MIQCP", "MIP"), lambda y: 0),
        ("RMIP", ("RMIP", "RMIQCP", None), lambda y: 0),
        ("QCP", ("QCP", "QCP", "MIQCP"), gp.math.sqr),
        ("MIQCP", ("MIQCP", "MIQCP", "MIQCP"), gp.math.sqr),
        ("RMIQCP", ("RMIQCP", "RMIQCP", None), gp.math.sqr),
        ("NLP", ("NLP", "NLP", "MINLP"), _exp_sqr),
        ("MINLP", ("MINLP", "MINLP", "MINLP"), _exp_sqr),
        ("RMINLP", ("RMINLP", "RMINLP", None), _exp_sqr),
        ("DNLP", ("DNLP", "DNLP", "MINLP"), lambda y: gp.math.Max(y, 0)),
        ("CNS", ("NLP", "NLP", "MINLP"), gp.math.sqr),
    ],
)
@pytest.mark.parametrize("measure", ["sum", "squares", "count"])
def test_feasibility_relaxation_problem_types(problem, relaxed, term, measure):
    m = Container()
    x = Variable(m, "x", type="integer" if problem.startswith("MI") else "free")
    y = Variable(m, "y")
    x.lo, x.up, y.lo, y.up, y.l = 0, 10, -1, 1, -0.5
    if problem == "CNS":
        lo = Equation(m, "lo", definition=x == 3)
        up = Equation(m, "up", definition=x + term(y) == 1)
        model = Model(m, equations=[lo, up], problem=problem)
    else:
        lo = Equation(m, "lo", definition=x >= 3)
        up = Equation(m, "up", definition=x + term(y) <= 1)
        model = Model(
            m, equations=[lo, up], problem=problem, sense="min", objective=2 * x
        )

    expected = relaxed[("sum", "squares", "count").index(measure)]
    if expected is None:
        with pytest.raises(ValidationError, match="relaxed problem type"):
            model.feasibilityRelaxation(measure=measure)
        return

    relax = model.feasibilityRelaxation(measure=measure)
    assert relax.model.problem == Problem(expected)
    assert (relax.optimality_model is None) == (problem == "CNS")

    relax.solve()
    solved = (
        gp.ModelStatus.OptimalGlobal,
        gp.ModelStatus.OptimalLocal,
        gp.ModelStatus.Integer,
    )
    assert relax.model.status in solved
    assert relax.violation.toValue() == pytest.approx(
        1 if measure == "count" else 2, rel=1e-6
    )
    violated = list(relax.violations())
    if measure == "squares":
        assert violated == ["lo", "up"]
    else:
        assert len(violated) == 1

    if relax.optimality_model is not None:
        assert relax.optimality_model.status in solved
        best = {"sum": 1, "squares": 2, "count": 0}[measure]
        assert x.toValue() == pytest.approx(best, rel=1e-4, abs=1e-6)
        if measure != "squares":
            assert violated == ["lo"]


@pytest.mark.requires_solvers("baron", "knitro")
@pytest.mark.parametrize("solver", ["baron", "knitro"])
def test_feasibility_relaxation_count_big_m(solver):
    m = Container()
    x = Variable(m, "x")
    y = Variable(m, "y")
    x.lo, x.up, y.lo, y.up = 0, 3, 0, 3
    disk = Equation(m, "disk", definition=gp.math.sqr(x) + gp.math.sqr(y) <= 1)
    product = Equation(m, "product", definition=x * y >= 2)
    growth = Equation(m, "growth", definition=gp.math.exp(x) >= 3)
    model = Model(
        m,
        equations=[disk, product, growth],
        problem="NLP",
        sense="min",
        objective=x + y,
    )

    # the default SOS1 link is not supported by BARON and KNITRO
    sos1 = model.feasibilityRelaxation(measure="count")
    assert sos1.model.problem == Problem.MINLP
    with pytest.raises(GamspyException, match="CapabilityError"):
        sos1.solve(solver=solver)

    big = model.feasibilityRelaxation(measure="count", big_m=20)
    big.solve(solver=solver)
    assert big.violation.toValue() == pytest.approx(1)
    assert list(big.violations()) == ["disk"]
    assert big.optimality_model.status in (
        gp.ModelStatus.OptimalGlobal,
        gp.ModelStatus.OptimalLocal,
        gp.ModelStatus.Integer,
    )


def test_feasibility_relaxation_validation():
    m = Container()
    x = Variable(m, "x")
    e = Equation(m, "e", definition=x >= 3)
    other = Equation(m, "other", definition=x <= 1)
    undefined = Equation(m, "undefined")
    model = Model(m, equations=[e, undefined], problem="LP")

    with pytest.raises(ValidationError, match="not an equation of model"):
        model.feasibilityRelaxation(equations=[other])

    with pytest.raises(ValidationError, match="not relaxed"):
        model.feasibilityRelaxation(equations=[e], weights={other: 2})

    with pytest.raises(ValidationError, match="cannot be relaxed"):
        model.feasibilityRelaxation()

    mcp = Model(m, problem="MCP", matches={e: x})
    with pytest.raises(ValidationError, match="matches"):
        mcp.feasibilityRelaxation()
