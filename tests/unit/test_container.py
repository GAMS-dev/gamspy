from __future__ import annotations

import gc
import glob
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import timeit
import uuid
from pathlib import Path

import pandas as pd
import pytest
import requests

import gamspy.utils as utils
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
    UniverseAlias,
    Variable,
    VariableType,
    deserialize,
    serialize,
)
from gamspy.exceptions import GamspyException, ValidationError


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


@pytest.mark.unit
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

    m = Container(options=Options.fromGams({"reslim": 5}))

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

    ## Invalid
    with pytest.raises(ValidationError):
        _ = Container(load_from=1)

    with pytest.raises(ValidationError):
        _ = Container(load_from="bla.gdp")

    ## Read from another container
    new_cont = Container(m)
    new_cont["a"][...] = 5
    assert new_cont.data.keys() == m.data.keys()

    ## Read from a pathlike load_from
    gdx_path = os.path.join("tmp", "_" + str(uuid.uuid4()) + ".gdx")
    writer_cont = Container()
    i = Set(writer_cont, "i", records=range(3))
    writer_cont.write(gdx_path)
    reader_cont = Container(load_from=Path(gdx_path))
    assert "i" in reader_cont.data
    assert reader_cont["i"].toList() == ["0", "1", "2"]

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


@pytest.mark.unit
def test_str(data):
    m, *_ = data
    assert str(m) == f"<Empty Container ({hex(id(m))})>"

    _ = Set(m, "i")
    assert (
        str(m) == f"<Container ({hex(id(m))}) with {len(m)} symbols: {m.data.keys()}>"
    )


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
def test_loadRecordsFromGdx(data):
    m, *_ = data
    with tempfile.TemporaryDirectory() as tmp_path:
        gdx_path = os.path.join(tmp_path, "test.gdx")

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

        # Test renaming
        m = Container()
        i = Set(m, "i", records=range(5))
        m.write(gdx_path)

        m = Container()
        j = Set(m, "j")
        m.loadRecordsFromGdx(gdx_path, symbol_names={"i": "j"})
        assert j.toList() == ["0", "1", "2", "3", "4"]

        with pytest.raises(ValidationError):
            m.loadRecordsFromGdx(gdx_path, symbol_names={"i": "k"})


@pytest.mark.unit
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


@pytest.mark.unit
def test_add_gams_code_domain_recovery(data):
    m, *_ = data
    i = Set(m, name="i", records=range(10))
    m.addGamsCode("positive variable x(i); x.l(i)=1;")
    x = m["x"]
    x.fx[i].where[i.ord == 1] = 1
    assert x.domain == [i]
    assert x.fx.domain == [i]


@pytest.mark.unit
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


@pytest.mark.requires_license
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
        data = requests.get(link).content.decode("utf-8")
        with Container() as m:
            m.addGamsCode(data)


@pytest.mark.unit
def test_system_directory():
    import gamspy_base

    expected_path = gamspy_base.__path__[0]

    m = Container()

    if os.getenv("GAMSPY_GAMS_SYSDIR", None) is None:
        assert (
            Path(m.system_directory.lower()).resolve()
            == Path(expected_path.lower()).resolve()
        )

        assert (
            Path(utils._get_gamspy_base_directory().lower()).resolve()
            == Path(expected_path.lower()).resolve()
        )
    else:
        assert (
            Path(m.system_directory).resolve()
            == Path(os.environ["GAMSPY_GAMS_SYSDIR"]).resolve()
        )


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
def test_generate_gams_string():
    m = Container(debugging_level="keep")
    i = Set(m, "i")
    _ = Alias(m, "a", i)
    _ = Parameter(m, "p")
    _ = Variable(m, "v")
    _ = Equation(m, "e")

    generated = m.generateGamsString()
    expected = "$onMultiR\n$onUNDF\n$onDotL\nSet i(*) / /;\n$offDotL\n$offUNDF\n$offMulti\n$onMultiR\n$onUNDF\n$onDotL\nAlias(i,a);\n$offDotL\n$offUNDF\n$offMulti\n$onMultiR\n$onUNDF\n$onDotL\nParameter p / /;\n$offDotL\n$offUNDF\n$offMulti\n$onMultiR\n$onUNDF\n$onDotL\nfree Variable v / /;\n$offDotL\n$offUNDF\n$offMulti\n$onMultiR\n$onUNDF\n$onDotL\nEquation e / /;\n$offDotL\n$offUNDF\n$offMulti\n"
    assert generated == expected

    assert (
        m.generateGamsString(show_raw=True)
        == """Set i(*) / /;
Alias(i,a);
Parameter p / /;
free Variable v / /;
Equation e / /;
"""
    )
    m.close()

    m2 = Container()
    with pytest.raises(ValidationError):
        m2.generateGamsString()

    m2.close()


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
def test_read(data):
    m, *_ = data
    gdx_path = os.path.join("tmp", "test.gdx")

    _ = Parameter(m, "a", records=5)
    m.write(gdx_path)

    m = Container()
    m.read(gdx_path, load_records=False)
    assert m["a"].records is None

    m.close()


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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
    _ = Set(m, "j", records=range(6))
    assert list(m.data.keys()) == ["i", "j"]
    m.close()


@pytest.mark.unit
def test_serialization(data) -> None:
    m, canning_plants, markets, capacities, demands, distances = data
    i = Set(m, name="i", records=canning_plants)
    j = Set(m, name="j", records=markets)
    k = Alias(m, name="k", alias_with=i)

    a = Parameter(m, name="a", domain=[i], records=capacities)
    b = Parameter(m, name="b", domain=[j], records=demands)
    d = Parameter(m, name="d", domain=[i, j], records=distances)
    c = Parameter(m, name="c", domain=[i, j])
    c[i, j] = 90 * d[i, j] / 1000

    x = Variable(m, name="x", domain=[i, j], type="Positive")
    x._metadata = {"some_metadata": "some_value"}

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

    with tempfile.TemporaryDirectory() as directory:
        serialization_path = os.path.join(directory, "serialized.zip")
        serialization_path2 = os.path.join(directory, "serialized2.zip")
        # Incorrect zip file name
        with pytest.raises(ValidationError):
            serialize(m, "bla")

        # Incorrect container type
        with pytest.raises(ValidationError):
            serialize(i, serialization_path)

        serialize(m, serialization_path)  # try gp.serialize syntax
        m.serialize(serialization_path2)  # try container.serialize syntax

        m2 = deserialize(serialization_path)

    assert id(m) != id(m2)
    assert m.data.keys() == m2.data.keys()

    # Test model
    transport2 = m2.models["transport"]
    assert id(transport) != transport2
    assert transport.name == transport2.name
    assert transport.objective_value == transport2.objective_value

    # Test symbols
    i2: Set = m2["i"]
    j2: Set = m2["j"]
    k2: Alias = m2["k"]
    c2: Parameter = m2["c"]
    d2: Parameter = m2["d"]

    assert i.records.equals(i2.records)
    assert k.records.equals(k2.records)
    assert c.records.equals(c2.records)
    assert all(not isinstance(elem, str) for elem in c2.domain)
    assert c.domain_names == c2.domain_names
    x2: Variable = m2["x"]
    assert x.records.equals(x2.records)
    assert all(not isinstance(elem, str) for elem in c2.domain)
    assert x.domain_names == x2.domain_names
    assert x._metadata == x2._metadata
    supply2: Equation = m2["supply"]
    assert supply.records.equals(supply2.records)
    assert all(not isinstance(elem, str) for elem in supply2.domain)
    assert supply.domain_names == supply2.domain_names

    # Test assignment and definitions
    assert c.getAssignment() == c2.getAssignment()
    assert supply.getDefinition() == supply2.getDefinition()

    # Test solve
    c2[i2, j2] = 90 * d2[i2, j2] / 100
    transport2.solve()
    assert transport.objective_value != transport2.objective_value


@pytest.mark.unit
def test_mcp_serialization(data) -> None:
    m, _, _, _, _, _ = data
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

    with tempfile.TemporaryDirectory() as directory:
        serialization_path = os.path.join(directory, "mcp_serialized.zip")
        serialize(m, serialization_path)  # try gp.serialize syntax
        m2 = deserialize(serialization_path)

    hansen2 = m2.models["hansen"]

    for unserialized, serialized in zip(
        hansen._matches.items(), hansen2._matches.items(), strict=False
    ):
        orig_equation, orig_variable = unserialized
        serialized_equation, serialized_variable = serialized

        assert isinstance(serialized_equation, Equation)
        assert isinstance(serialized_variable, Variable)
        assert orig_equation.name == serialized_equation.name
        assert orig_variable.name == serialized_variable.name


@pytest.mark.unit
def test_auto_python_name_retrieval():
    import gamspy as gp

    with gp.Container():
        gp.set_options({"USE_PY_VAR_NAME": "no"})

        i = gp.Set()
        assert i.name != "i"  # autogen name, will be different every time

        gp.set_options({"USE_PY_VAR_NAME": "yes"})

        # Reserved names are not allowed,
        with pytest.raises(ValidationError):
            binary = gp.Set()  # noqa: F841

        i = gp.Set()
        assert i.name == "i"

        gp.set_options({"USE_PY_VAR_NAME": "no"})

        j = gp.Alias(alias_with=i)
        assert j.name != "j"  # autogen name, will be different every time

        gp.set_options({"USE_PY_VAR_NAME": "yes"})

        j = gp.Alias(alias_with=i)
        assert j.name == "j"

        gp.set_options({"USE_PY_VAR_NAME": "no"})

        k = gp.Parameter()
        assert k.name != "k"  # autogen name, will be different every time

        gp.set_options({"USE_PY_VAR_NAME": "yes"})

        k = gp.Parameter()
        assert k.name == "k"

        gp.set_options({"USE_PY_VAR_NAME": "no"})

        l = gp.Variable()
        assert l.name != "l"  # autogen name, will be different every time

        gp.set_options({"USE_PY_VAR_NAME": "yes"})

        l = gp.Variable()
        assert l.name == "l"

        gp.set_options({"USE_PY_VAR_NAME": "no"})

        n = gp.Equation()
        assert n.name != "n"  # autogen name, will be different every time

        gp.set_options({"USE_PY_VAR_NAME": "yes"})

        n = gp.Equation()
        assert n.name == "n"

    m = gp.Container()
    gp.set_options({"USE_PY_VAR_NAME": "no"})

    i = gp.Set(m)
    assert i.name != "i"  # autogen name, will be different every time

    gp.set_options({"USE_PY_VAR_NAME": "yes"})

    i = gp.Set(m)
    assert i.name == "i"

    # GAMS symbol names cannot begin with a '_' character
    with pytest.raises(ValidationError):
        _bla = gp.Set(m)

    gp.set_options({"USE_PY_VAR_NAME": "yes-or-autogenerate"})
    _ = gp.Set(m)  # autogen a name
    _ = gp.Alias(m, alias_with=i)
    _ = gp.Parameter(m)  # autogen a name
    _ = gp.Variable(m)  # autogen a name
    _ = gp.Equation(m)  # autogen a name

    test_set = gp.Set(m)
    assert test_set.name == "test_set"
    test_set = gp.Set(m)
    assert test_set.name != "test_set"

    test_parameter = gp.Parameter(m)
    assert test_parameter.name == "test_parameter"
    test_parameter = gp.Parameter(m)
    assert test_parameter.name != "test_parameter"

    test_variable = gp.Variable(m)
    assert test_variable.name == "test_variable"
    test_variable = gp.Variable(m)
    assert test_variable.name != "test_variable"

    test_equation = gp.Equation(m)
    assert test_equation.name == "test_equation"
    test_equation = gp.Equation(m)
    assert test_equation.name != "test_equation"

    gp.set_options({"USE_PY_VAR_NAME": "no"})

    j = gp.Alias(m, alias_with=i)
    assert j.name != "j"  # autogen name, will be different every time

    gp.set_options({"USE_PY_VAR_NAME": "yes"})

    j = gp.Alias(m, alias_with=i)
    assert j.name == "j"

    gp.set_options({"USE_PY_VAR_NAME": "no"})

    k = gp.Parameter(m)
    assert k.name != "k"  # autogen name, will be different every time

    gp.set_options({"USE_PY_VAR_NAME": "yes"})

    k = gp.Parameter(m)
    assert k.name == "k"

    gp.set_options({"USE_PY_VAR_NAME": "no"})

    l = gp.Variable(m)
    assert l.name != "l"  # autogen name, will be different every time

    gp.set_options({"USE_PY_VAR_NAME": "yes"})

    l = gp.Variable(m)
    assert l.name == "l"

    gp.set_options({"USE_PY_VAR_NAME": "no"})

    n = gp.Equation(m)
    assert n.name != "n"  # autogen name, will be different every time

    gp.set_options({"USE_PY_VAR_NAME": "yes"})

    n = gp.Equation(m)
    assert n.name == "n"

    p = gp.Model(m)
    assert p.name == "p"

    gp.set_options({"USE_PY_VAR_NAME": "no"})


@pytest.mark.unit
def test_explicit_license_path():
    import gamspy_base

    import gamspy as gp

    demo_license_path = os.path.join(gamspy_base.directory, "gamslice.txt")
    m = gp.Container(options=gp.Options(license=demo_license_path))
    assert m._license_path == demo_license_path

    f = gp.Set(
        m,
        name="f",
        description="faces on a dice",
        records=[f"face{idx}" for idx in range(1, 20)],
    )
    dice = gp.Set(
        m,
        name="dice",
        description="number of dice",
        records=[f"dice{idx}" for idx in range(1, 20)],
    )

    flo = gp.Parameter(m, name="flo", description="lowest face value", records=1)
    fup = gp.Parameter(
        m, "fup", description="highest face value", records=len(dice) * len(f)
    )

    fp = gp.Alias(m, name="fp", alias_with=f)

    wnx = gp.Variable(m, name="wnx", description="number of wins")
    fval = gp.Variable(
        m,
        name="fval",
        domain=[dice, f],
        description="face value on dice - may be fractional",
    )
    comp = gp.Variable(
        m,
        name="comp",
        domain=[dice, f, fp],
        description="one implies f beats fp",
        type=gp.VariableType.BINARY,
    )

    fval.lo[dice, f] = flo
    fval.up[dice, f] = fup
    fval.fx["dice1", "face1"] = flo

    eq1 = gp.Equation(m, "eq1", domain=dice, description="count the wins")
    eq3 = gp.Equation(
        m,
        "eq3",
        domain=[dice, f, fp],
        description="definition of non-transitive relation",
    )
    eq4 = gp.Equation(
        m,
        "eq4",
        domain=[dice, f],
        description="different face values for a single dice",
    )

    eq1[dice] = gp.Sum((f, fp), comp[dice, f, fp]) == wnx
    eq3[dice, f, fp] = (
        fval[dice, f] + (fup - flo + 1) * (1 - comp[dice, f, fp])
        >= fval[dice.lead(1, type="circular"), fp] + 1
    )
    eq4[dice, f - 1] = fval[dice, f - 1] + 1 <= fval[dice, f]

    xdice = gp.Model(
        m,
        "xdice",
        equations=m.getEquations(),
        problem=gp.Problem.MIP,
        sense=gp.Sense.MAX,
        objective=wnx,
    )

    # Should throw license error since we are using the demo license.
    with pytest.raises(GamspyException):
        xdice.solve()


def test_writeSolverOptions():
    m = Container()
    m.writeSolverOptions(
        "conopt",
        solver_options={"rtmaxv": "1.e12"},
    )
    solver_options_path = os.path.join(m.working_directory, "conopt.opt")
    assert os.path.exists(solver_options_path)
    with open(solver_options_path) as file:
        assert "rtmaxv" in file.read()

    m.writeSolverOptions("conopt", solver_options={"rtmaxv": "1.e12"}, file_number=2)
    solver_options_path = os.path.join(m.working_directory, "conopt.op2")
    assert os.path.exists(solver_options_path)

    m.writeSolverOptions("conopt", solver_options={"rtmaxv": "1.e12"}, file_number=9)
    solver_options_path = os.path.join(m.working_directory, "conopt.op9")
    assert os.path.exists(solver_options_path)

    m.writeSolverOptions("conopt", solver_options={"rtmaxv": "1.e12"}, file_number=10)
    solver_options_path = os.path.join(m.working_directory, "conopt.o10")
    assert os.path.exists(solver_options_path)

    m.writeSolverOptions("conopt", solver_options={"rtmaxv": "1.e12"}, file_number=99)
    solver_options_path = os.path.join(m.working_directory, "conopt.o99")
    assert os.path.exists(solver_options_path)

    m.writeSolverOptions("conopt", solver_options={"rtmaxv": "1.e12"}, file_number=100)
    solver_options_path = os.path.join(m.working_directory, "conopt.100")
    assert os.path.exists(solver_options_path)

    m.writeSolverOptions("conopt", solver_options={"rtmaxv": "1.e12"}, file_number=999)
    solver_options_path = os.path.join(m.working_directory, "conopt.999")
    assert os.path.exists(solver_options_path)

    m.writeSolverOptions("conopt", solver_options={"rtmaxv": "1.e12"}, file_number=1234)
    solver_options_path = os.path.join(m.working_directory, "conopt.1234")
    assert os.path.exists(solver_options_path)

    m.close()


@pytest.mark.unit
def test_domain_violations():
    import gamspy as gp

    gp.set_options({"DROP_DOMAIN_VIOLATIONS": 1})

    c = gp.Container()
    i = gp.Set(c, "i", records=["i1"])

    j = gp.Set(c, "j", domain=i, records=["i1", "i2"])
    assert j.toList() == ["i1"]
    assert j._domain_violations[0].violations == ["i2"]

    p = gp.Parameter(c, "p", domain=[i], records=[("i1", 10), ("i2", 20)])
    assert p.toList() == [("i1", 10.0)]
    assert p._domain_violations[0].violations == ["i2"]

    v = gp.Variable(
        c,
        "v",
        domain=[i],
        records=pd.DataFrame(
            data=[("i1", 5), ("i2", 10)],
            columns=["domain", "level"],
        ),
    )
    assert v.toList() == [("i1", 5.0)]
    assert v._domain_violations[0].violations == ["i2"]

    e = gp.Equation(
        c,
        "e",
        domain=[i],
        records=pd.DataFrame(
            data=[("i1", 10), ("i2", 10)],
            columns=["domain", "level"],
        ),
    )
    assert e.toList() == [("i1", 10.0)]
    assert e._domain_violations[0].violations == ["i2"]

    gp.set_options({"DROP_DOMAIN_VIOLATIONS": 0})


@pytest.mark.unit
def test_expert_sync():
    import gamspy as gp

    m = gp.Container()
    i = gp.Set(m, "i", records=["i1", "i2", "i3"])
    a = gp.Parameter(m, "a", domain=[i, i])
    a.synchronize = False
    for s in range(5):
        a[i, i].where[gp.Ord(i) == 1] = 0.1 * s
        assert a.records is None
    a.synchronize = True
    assert a.toList() == [("i1", "i1", 0.4)]


def one_by_one(n: int, m: Container):
    sets = [Set(m) for _ in range(10)]
    for set in sets:
        set.setRecords(range(n))

    params = [Parameter(m) for _ in range(10)]
    for param in params:
        param.setRecords(n)


def batched(n: int, m: Container):
    sets = [Set(m) for _ in range(10)]
    values = [range(n)] * 10
    m.setRecords(dict(zip(sets, values, strict=False)))

    params = [Parameter(m) for _ in range(10)]
    values = [n] * 10
    m.setRecords(dict(zip(params, values, strict=False)))


@pytest.mark.skipif(
    platform.system() != "Linux",
    reason="Test only for linux because other build machines are slow enough and there is no platform dependent behavior.",
)
@pytest.mark.unit
def test_batch_setRecords():
    n = 10
    m = Container()
    one_by_one_result = timeit.repeat(
        "one_by_one(n, m)",
        globals={"n": n, "one_by_one": one_by_one, "m": m},
        repeat=30,
        number=1,
    )

    m = Container()
    batched_result = timeit.repeat(
        "batched(n, m)",
        globals={"n": n, "batched": batched, "m": m},
        repeat=30,
        number=1,
    )
    assert min(one_by_one_result) > min(batched_result)
    m.close()

    m = Container()
    i = Set(m, "i")
    k = Set(m, "k")

    with pytest.raises(ValidationError):
        m.setRecords({i: range(10), k: range(5)}, uels_on_axes=[True, False, True])
