from __future__ import annotations

import gc
import glob
import math
import os
import shutil
import subprocess
import sys
import uuid

import pandas as pd
import pytest
import urllib3

import gamspy.utils as utils
from gamspy import (
    Alias,
    Container,
    Equation,
    Model,
    Parameter,
    Problem,
    Sense,
    Set,
    Sum,
    UniverseAlias,
    Variable,
    deserialize,
    serialize,
)
from gamspy.exceptions import GamspyException, ValidationError

pytestmark = pytest.mark.unit


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
    shutil.rmtree("tmp")
    m.close()
    mpsge_file_path = os.path.join(os.getcwd(), "HANSEN.GEN")
    if os.path.exists(mpsge_file_path):
        os.remove(mpsge_file_path)


def test_container(data):
    m, *_ = data
    import gams.transfer as gt

    with pytest.raises(ValidationError):
        _ = Container(working_directory="")

    with pytest.raises(ValidationError):
        _ = Container(working_directory="a" * 205)

    shutil.rmtree("a" * 205)

    with pytest.raises(TypeError):
        m = Container(options={"bla": "bla"})

    i = gt.Set(m, "i")
    m._cast_symbols()
    assert isinstance(m["i"], Set)

    j = gt.Alias(m, "j", i)
    m._cast_symbols()
    assert isinstance(m["j"], Alias)

    a = gt.Parameter(m, "a")
    m._cast_symbols()
    assert isinstance(m["a"], Parameter)

    v = gt.Variable(m, "v")
    m._cast_symbols()
    assert isinstance(m["v"], Variable)

    e = gt.Equation(m, "e", type="eq")
    m._cast_symbols()
    assert isinstance(m["e"], Equation)

    # Test load_from
    with pytest.raises(ValidationError):
        _ = Container(load_from=1)

    with pytest.raises(ValidationError):
        _ = Container(load_from="bla.gdp")

    new_cont = Container(m)
    new_cont["a"][...] = 5
    assert new_cont.data.keys() == m.data.keys()

    # Test getters
    m = Container()

    i = Set(m, "i")
    assert isinstance(m["i"], Set)

    j = Alias(m, "j", i)
    assert isinstance(m["j"], Alias)

    a = Parameter(m, "a")
    assert isinstance(m["a"], Parameter)

    v = Variable(m, "v")
    assert isinstance(m["v"], Variable)

    e = Equation(m, "e")
    assert isinstance(m["e"], Equation)

    assert m.getSets() == [i]
    assert m.getAliases() == [j]
    assert m.getParameters() == [a]
    assert m.getVariables() == [v]
    assert m.getEquations() == [e]

    # test addX syntax
    m = Container()
    i1 = m.addSet("i")
    pytest.raises(ValueError, m.addSet, "i", i1)
    assert isinstance(i1, Set)
    i2 = m.addSet("i")
    assert id(i1) == id(i2)
    i3 = m.addSet("i", records=["new_record"], description="new desc")
    assert id(i1) == id(i3)
    pytest.raises(ValueError, m.addSet, "i", [j])
    pytest.raises(ValueError, m.addSet, "i", None, 5)

    j1 = m.addAlias("j", i1)
    assert isinstance(j1, Alias)
    j2 = m.addAlias("j", i1)
    assert id(j1) == id(j2)
    j3 = m.addAlias("j", j2)
    assert id(j3) == id(j2)

    a1 = m.addParameter("a")
    pytest.raises(ValueError, m.addParameter, "a", i1)
    assert isinstance(a1, Parameter)
    a2 = m.addParameter("a")
    assert id(a1) == id(a2)
    pytest.raises(ValueError, m.addParameter, "a", ["*"])
    pytest.raises(ValueError, m.addParameter, "a", None, None, 5)

    v1 = m.addVariable("v")
    pytest.raises(ValueError, m.addVariable, "v", "free", domain=i1)
    assert isinstance(v1, Variable)
    v2 = m.addVariable("v", description="blabla", records=pd.DataFrame())
    assert id(v1) == id(v2)
    pytest.raises(ValueError, m.addVariable, "v", "free", ["*"])
    pytest.raises(TypeError, m.addVariable, "v", "dayum")

    e1 = m.addEquation("e")
    pytest.raises(ValueError, m.addEquation, "e", "regular", i1)
    assert isinstance(e1, Equation)
    e2 = m.addEquation("e")
    assert id(e1) == id(e2)
    pytest.raises(ValueError, m.addEquation, "e", "bla")
    pytest.raises(TypeError, m.addEquation, "e", "leq")
    e3 = m.addEquation("e", records=pd.DataFrame())
    assert id(e3) == id(e1)


def test_str(data):
    m, *_ = data
    assert str(m) == f"<Empty Container ({hex(id(m))})>"

    _ = Set(m, "i")
    assert (
        str(m)
        == f"<Container ({hex(id(m))}) with {len(m)} symbols: {m.data.keys()}>"
    )


def test_read_write(data):
    m, *_ = data
    gdx_path = os.path.join("tmp", "test.gdx")

    m2 = Container()
    _ = Set(m2, "i", records=["i1", "i2"])
    _ = Set(m2, "j", records=["j1", "j2"])
    m2.write(gdx_path)

    _ = Set(m, name="k", records=["k1", "k2"])
    m.read(gdx_path, ["i"])
    assert list(m.data.keys()) == ["k", "i"]


def test_read_synch():
    m = Container()

    j = Set(
        m,
        name="j",
        records=["new-york", "chicago", "topeka"],
        description="markets",
    )
    _ = Set(
        m,
        name="j_sub",
        records=["new-york", "chicago"],
        domain=j,
        description="markets",
    )

    gdx_file = "test.gdx"
    m.write(gdx_file)
    m = Container()
    m.read(gdx_file)

    assert m["j"].toList() == ["new-york", "chicago", "topeka"]
    assert m["j_sub"].toList() == ["new-york", "chicago"]
    m["j_sub"][m["j"]] = False

    m = Container(gdx_file)

    assert m["j"].toList() == ["new-york", "chicago", "topeka"]
    assert m["j_sub"].toList() == ["new-york", "chicago"]
    m["j_sub"][m["j"]] = False

    os.remove("test.gdx")


def test_loadRecordsFromGdx(data):
    m, *_ = data
    gdx_path = os.path.join("tmp", "test.gdx")

    i = Set(m, name="i", records=["i1", "i2"])
    a = Parameter(m, name="a", domain=[i], records=[("i1", 1), ("i2", 2)])
    m.write(gdx_path)

    # Load all
    new_container = Container()
    i = Set(new_container, name="i")
    a = Parameter(new_container, name="a", domain=[i])
    new_container.loadRecordsFromGdx(gdx_path)

    assert i.records.values.tolist() == [["i1", ""], ["i2", ""]]

    assert a.records.values.tolist() == [["i1", 1.0], ["i2", 2.0]]

    # Load specific symbols
    new_container2 = Container()
    i = Set(new_container2, name="i")
    a = Parameter(new_container2, name="a", domain=[i])
    new_container2.loadRecordsFromGdx(gdx_path, ["i"])

    assert i.records.values.tolist() == [["i1", ""], ["i2", ""]]
    assert a.records is None

    m = Container()
    i = Set(m, "i", records=["i1", "i2", "i3"])
    j = Set(m, "j", i, records=["i1", "i2"])
    m.write(gdx_path)

    m = Container()
    i = Set(m, "i")
    j = Set(m, "j", i, domain_forwarding=True)
    m.loadRecordsFromGdx(gdx_path, ["j"])
    assert i.toList() == ["i1", "i2"]
    assert j.toList() == ["i1", "i2"]


def test_enums():
    assert str(Problem.LP) == "LP"
    assert str(Sense.MAX) == "MAX"

    assert Problem.values() == [
        "LP",
        "NLP",
        "QCP",
        "DNLP",
        "MIP",
        "RMIP",
        "MINLP",
        "RMINLP",
        "MIQCP",
        "RMIQCP",
        "MCP",
        "CNS",
        "MPEC",
        "RMPEC",
        "EMP",
        "MPSGE",
    ]

    assert Sense.values() == ["MIN", "MAX", "FEASIBILITY"]


def test_add_gams_code_domain_recovery(data):
    m, *_ = data
    i = Set(m, name="i", records=range(10))
    m.addGamsCode("positive variable x(i); x.l(i)=1;")
    x = m["x"]
    x.fx[i].where[i.ord == 1] = 1
    assert x.domain == [i]
    assert x.fx.domain == [i]


def test_arbitrary_gams_code():
    m = Container()
    i = Set(m, "i", records=["i1", "i2"])
    i["i1"] = False
    m.addGamsCode("scalar piHalf / [pi/2] /;")
    assert "piHalf" in m.data
    assert m["piHalf"].records.values[0][0] == 1.5707963267948966

    m = Container()
    codestr = """set T /month0*month4/;
    parameter demand(T) /month1 40, month2 60, month3 75, month4 25/;"""
    m.addGamsCode(codestr)
    m.addGamsCode("display demand;")

    T = m["T"]
    demand = m["demand"]
    assert T.toList() == [
        "month0",
        "month1",
        "month2",
        "month3",
        "month4",
    ]
    assert demand.domain == [T]


def test_add_gams_code_on_actual_models():
    links = {
        "LP": "https://gams.com/latest/gamslib_ml/trnsport.1",
        "MIP": "https://gams.com/latest/gamslib_ml/prodsch.9",
        "NLP": "https://gams.com/latest/gamslib_ml/weapons.18",
        "MCP": "https://gams.com/latest/gamslib_ml/wallmcp.127",
        "CNS": "https://gams.com/latest/gamslib_ml/camcns.209",
        "DNLP": "https://gams.com/latest/gamslib_ml/linear.23",
        "MINLP": "https://gams.com/latest/gamslib_ml/meanvarx.113",
        "QCP": "https://gams.com/latest/gamslib_ml/himmel11.95",
        "MIQCP": "https://gams.com/latest/gamslib_ml/qalan.282",
        "MPSGE": "https://gams.com/latest/gamslib_ml/hansmge.147",
    }

    for link in links.values():
        data = urllib3.request("GET", link).data.decode("utf-8")
        with Container() as m:
            m.addGamsCode(data)


def test_system_directory():
    import gamspy_base

    expected_path = gamspy_base.__path__[0]

    m = Container()

    if os.getenv("GAMSPY_GAMS_SYSDIR", None) is None:
        assert m.system_directory.lower() == expected_path.lower()

        assert (
            utils._get_gamspy_base_directory().lower() == expected_path.lower()
        )
    else:
        assert m.system_directory == os.environ["GAMSPY_GAMS_SYSDIR"]


def test_write_load_on_demand(data):
    m, *_ = data
    i = Set(m, name="i", records=["i1"])
    p1 = Parameter(m, name="p1", domain=[i], records=[["i1", 1]])
    p2 = Parameter(m, name="p2", domain=[i])
    p2[i] = p1[i]
    m.write(f"tmp{os.sep}data.gdx")
    m = Container(
        load_from=f"tmp{os.sep}data.gdx",
    )
    assert m["p2"].toList() == [("i1", 1.0)]


def test_copy(data):
    _, canning_plants, markets, capacities, demands, distances = data
    m = Container(
        working_directory=f"tmp{os.sep}copy",
    )

    i = Set(m, name="i", records=canning_plants)
    j = Set(m, name="j", records=markets)
    _ = Alias(m, "k", alias_with=j)
    _ = UniverseAlias(m)

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

    pytest.raises(ValidationError, m.copy, f"tmp{os.sep}copy")
    new_cont = m.copy(working_directory=f"tmp{os.sep}test")
    assert m.data.keys() == new_cont.data.keys()
    assert supply.getDefinition() == new_cont["supply"].getDefinition()
    assert demand.getDefinition() == new_cont["demand"].getDefinition()

    transport = Model(
        new_cont,
        name="transport",
        equations=m.getEquations(),
        problem="LP",
        sense=Sense.MIN,
        objective=Sum((i, j), c[i, j] * x[i, j]),
    )

    transport.solve()

    assert new_cont.gamsJobName()
    assert math.isclose(transport.objective_value, 153.675, rel_tol=1e-6)


def test_generate_gams_string():
    m = Container(debugging_level="keep")
    i = Set(m, "i")
    _ = Alias(m, "a", i)
    _ = Parameter(m, "p")
    _ = Variable(m, "v")
    _ = Equation(m, "e")

    assert (
        m.generateGamsString()
        == f"$onMultiR\n$onUNDF\nSet i(*);\n$gdxIn {m._gdx_in}\n$loadDC i\n$gdxIn\n$offUNDF\n$onMultiR\n$onUNDF\nAlias(i,a);\n$gdxIn {m._gdx_in}\n$loadDC i\n$gdxIn\n$offUNDF\n$onMultiR\n$onUNDF\nParameter p;\n$gdxIn {m._gdx_in}\n$loadDC p\n$gdxIn\n$offUNDF\n$onMultiR\n$onUNDF\nfree Variable v;\n$gdxIn {m._gdx_in}\n$loadDC v\n$gdxIn\n$offUNDF\n$onMultiR\n$onUNDF\nEquation e;\n$gdxIn {m._gdx_in}\n$loadDC e\n$gdxIn\n$offUNDF\n"
    )

    assert (
        m.generateGamsString(show_raw=True)
        == """Set i(*);
Alias(i,a);
Parameter p;
free Variable v;
Equation e;
"""
    )
    m.close()

    m2 = Container()
    with pytest.raises(ValidationError):
        m2.generateGamsString()

    m2.close()


def test_removal_of_autogenerated_symbols(data):
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
    transport.solve()
    assert list(m.data.keys()) == [
        "i",
        "j",
        "a",
        "b",
        "d",
        "c",
        "x",
        "supply",
        "demand",
        "transport_objective_variable",
        "transport_objective",
    ]


def test_write(data):
    m, *_ = data
    gdx_path = os.path.join("tmp", "test.gdx")

    from gamspy import SpecialValues

    _ = Parameter(m, "a", records=SpecialValues.EPS)
    m.write(gdx_path, eps_to_zero=True)

    m = Container(
        load_from=gdx_path,
    )
    assert int(m["a"].toValue()) == 0

    m.close()


def test_read(data):
    m, *_ = data
    gdx_path = os.path.join("tmp", "test.gdx")

    _ = Parameter(m, "a", records=5)
    m.write(gdx_path)

    m = Container()
    m.read(gdx_path, load_records=False)
    assert m["a"].records is None

    m.close()


def test_debugging_level():
    from gamspy.math import sqrt

    with pytest.raises(ValidationError):
        _ = Container(debugging_level="wrong_level")

    global working_directory

    def test_delete_success():
        global working_directory
        m = Container(debugging_level="delete")
        working_directory = m.working_directory
        _ = Equation(m, "e")

    test_delete_success()
    gc.collect()
    assert not os.path.exists(working_directory)

    def test_delete_err():
        global working_directory
        m = Container(debugging_level="delete")
        working_directory = m.working_directory
        e = Equation(m, "e")
        with pytest.raises(GamspyException):
            e[:] = sqrt(e) == 5

    test_delete_err()
    gc.collect()
    assert not os.path.exists(working_directory)

    def test_keep_success():
        m = Container(debugging_level="keep")
        global working_directory
        working_directory = m.working_directory
        _ = Equation(m, "e")
        _ = Equation(m, "e2")

    test_keep_success()
    gc.collect()
    assert os.path.exists(working_directory)
    assert len(glob.glob(os.path.join(working_directory, "*.gms"))) == 2

    def test_keep_err():
        m = Container(debugging_level="keep")
        global working_directory
        working_directory = m.working_directory
        e = Equation(m, "e")
        with pytest.raises(GamspyException):
            e[:] = sqrt(e) == 5

    test_keep_err()
    gc.collect()
    assert os.path.exists(working_directory)
    assert len(glob.glob(os.path.join(working_directory, "*.gms"))) == 2

    def test_keep_on_error_success():
        m = Container(debugging_level="keep_on_error")
        global working_directory
        working_directory = m.working_directory
        _ = Equation(m, "e")

    test_keep_on_error_success()
    gc.collect()
    assert not os.path.exists(working_directory)

    def test_keep_on_error_err():
        m = Container(debugging_level="keep_on_error")
        global working_directory
        working_directory = m.working_directory
        e = Equation(m, "e")
        with pytest.raises(GamspyException):
            e[:] = sqrt(e) == 5

    test_keep_on_error_err()
    gc.collect()
    assert os.path.exists(working_directory)
    assert len(glob.glob(os.path.join(working_directory, "*.gms"))) == 1


def test_read_from_gdx(data):
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
    _ = Alias(m, "k", alias_with=i)

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
    transport.solve()
    gdx_path = os.path.join("tmp", "out.gdx")
    m.write(gdx_path)
    m.close()

    m = Container(load_from=gdx_path)
    assert m["supply"].toList() == [
        ("seattle", 350.0),
        ("san-diego", 550.0),
    ]
    assert m["x"].toList() == [
        ("seattle", "new-york", 50.0),
        ("seattle", "chicago", 300.0),
        ("seattle", "topeka", 0.0),
        ("san-diego", "new-york", 275.0),
        ("san-diego", "chicago", 0.0),
        ("san-diego", "topeka", 275.0),
    ]
    assert m["c"].toList() == [
        ("seattle", "new-york", 0.225),
        ("seattle", "chicago", 0.153),
        ("seattle", "topeka", 0.162),
        ("san-diego", "new-york", 0.225),
        ("san-diego", "chicago", 0.162),
        ("san-diego", "topeka", 0.126),
    ]

    assert m["i"].toList() == ["seattle", "san-diego"]
    assert m["k"].toList() == ["seattle", "san-diego"]
    m.close()


def test_output():
    path = os.path.join("tmp", str(uuid.uuid4()) + ".py")
    os.makedirs("tmp", exist_ok=True)
    with open(path, "w") as file:
        file.write(
            "import sys\nfrom gamspy import Container, Set\nm = Container(output=sys.stdout)\ni = Set(m)\nj = Set(m)"
        )

    process = subprocess.run(
        [sys.executable, path], capture_output=True, check=True, text=True
    )
    assert process.stdout

    os.remove(path)


def test_restart():
    m = Container()
    save_path = os.path.join(m.working_directory, "save.g00")
    m._options._set_debug_options({"save": save_path})
    _ = Set(m, "i", records=["i1", "i2"])
    assert os.path.exists(save_path)
    m.close()

    m = Container(load_from=save_path)
    assert "i" in m.data
    assert m["i"].toList() == ["i1", "i2"]
    m.close()


def test_serialization(data):
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
    transport.solve()
    serialization_path = os.path.join("tmp", "serialized")
    serialize(m, serialization_path)

    m2 = deserialize(serialization_path + ".zip")
    assert m.data.keys() == m2.data.keys()

    # Test model
    transport2 = m2.models["transport"]
    assert transport.name == transport2.name
    assert transport.objective_value == transport2.objective_value

    # Test symbols
    c2 = m2["c"]
    assert c.records.equals(c2.records)
    x2 = m2["x"]
    assert x.records.equals(x2.records)
    assert x.domain_names == x2.domain_names

    # Test solve
    transport2.solve()
