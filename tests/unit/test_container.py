from __future__ import annotations

import copy
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

import gams.transfer as gt
import gamspy_base
import numpy as np
import pandas as pd
import pytest

import gamspy as gp
import gamspy.utils as utils
from gamspy import (
    Alias,
    Container,
    Equation,
    Model,
    Options,
    Ord,
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
from gamspy.exceptions import GamspyException, GdxException, ValidationError


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
    mpsge_file_path = os.path.join(os.getcwd(), "HANSEN.GEN")
    if os.path.exists(mpsge_file_path):
        os.remove(mpsge_file_path)


@pytest.mark.unit
def test_container(data, tmp_path):
    m, *_ = data

    with pytest.raises(ValidationError):
        _ = Container(system_directory="invaliddir")

    _ = Container(system_directory=Path(gamspy_base.directory))

    with pytest.raises(ValidationError):
        _ = Container(working_directory="")

    long_dir = tmp_path / ("a" * 205)
    with pytest.raises(ValidationError):
        _ = Container(working_directory=str(long_dir))

    if long_dir.exists():
        shutil.rmtree(long_dir)

    with pytest.raises(TypeError):
        m = Container(options={"bla": "bla"})

    m = Container(options=Options.fromGams({"reslim": 5}))
    Parameter(m, "a")

    # Test load_from
    with pytest.raises(ValidationError):
        _ = Container(load_from=1)

    with pytest.raises(ValidationError):
        _ = Container(load_from="bla.gdp")

    ## Read from another container
    new_cont = Container(m)
    new_cont["a"][...] = 5
    assert new_cont.data.keys() == m.data.keys()

    ## Read from a pathlike load_from using tmp_path
    gdx_path = tmp_path / ("_" + str(uuid.uuid4()) + ".gdx")
    writer_cont = Container()
    i = Set(writer_cont, "i", records=range(3))
    writer_cont.write(str(gdx_path))
    reader_cont = Container(load_from=gdx_path)
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
    with pytest.raises(ValueError):
        m.addSet("i", i1)
    assert isinstance(i1, Set)
    i2 = m.addSet("i")
    assert id(i1) == id(i2)
    i3 = m.addSet("i", records=["new_record"], description="new desc")
    assert id(i1) == id(i3)
    with pytest.raises(ValueError):
        m.addSet("i", [j])
    with pytest.raises(ValueError):
        m.addSet("i", None, 5)

    j1 = m.addAlias("j", i1)
    assert isinstance(j1, Alias)
    j2 = m.addAlias("j", i1)
    assert id(j1) == id(j2)

    with pytest.raises(ValueError):
        m.addAlias("j", j2)

    a1 = m.addParameter("a")
    with pytest.raises(ValueError):
        m.addParameter("a", i1)
    assert isinstance(a1, Parameter)
    a2 = m.addParameter("a")
    assert id(a1) == id(a2)
    with pytest.raises(ValueError):
        m.addParameter("a", ["*"])
    with pytest.raises(ValueError):
        m.addParameter("a", None, None, 5)

    v1 = m.addVariable("v")
    with pytest.raises(ValueError):
        m.addVariable("v", "free", domain=i1)
    assert isinstance(v1, Variable)
    v2 = m.addVariable("v", description="blabla", records=pd.DataFrame())
    assert id(v1) == id(v2)
    with pytest.raises(ValueError):
        m.addVariable("v", "free", ["*"])
    with pytest.raises(ValueError):
        m.addVariable("v", "dayum")

    e1 = m.addEquation("e")
    with pytest.raises(ValueError):
        m.addEquation("e", "regular", i1)
    assert isinstance(e1, Equation)
    e2 = m.addEquation("e")
    assert id(e1) == id(e2)
    with pytest.raises(ValueError):
        m.addEquation("e", "bla")
    with pytest.raises(TypeError):
        m.addEquation("e", "leq")
    e3 = m.addEquation("e", records=pd.DataFrame())
    assert id(e3) == id(e1)

    # Test __iter__
    m = Container()
    i = Set(m, "i")
    a = Alias(m, "a", i)
    names = []
    symbols = []
    for name, symbol in m:
        names.append(name)
        symbols.append(symbol)

    assert names == ["i", "a"]
    assert symbols == [i, a]


@pytest.mark.unit
def test_str(data):
    m, *_ = data
    assert str(m) == f"<Empty Container ({hex(id(m))})>"

    _ = Set(m, "i")
    assert (
        str(m) == f"<Container ({hex(id(m))}) with {len(m)} symbols: {m.data.keys()}>"
    )


@pytest.mark.unit
def test_read_write(data, tmp_path):
    m, *_ = data
    gdx_path = str(tmp_path / "test.gdx")

    m2 = Container()
    _ = Set(m2, "i", records=["i1", "i2"])
    _ = Set(m2, "j", records=["j1", "j2"])
    m2.write(gdx_path)

    _ = Set(m, name="k", records=["k1", "k2"])
    m.read(gdx_path, ["i"])
    assert list(m.data.keys()) == ["k", "i"]


@pytest.mark.unit
def test_read_synch(tmp_path):
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

    gdx_file = str(tmp_path / "test.gdx")
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


@pytest.mark.unit
def test_loadRecordsFromGdx(data, tmp_path):
    m, *_ = data
    gdx_path = str(tmp_path / "test.gdx")

    i = Set(m, name="i", records=["i1", "i2"])
    a = Parameter(m, name="a", domain=[i], records=[("i1", 1), ("i2", 2)])
    m.write(gdx_path)

    # Load all
    new_container = Container()
    i = Set(new_container, name="i")
    a = Parameter(new_container, name="a", domain=[i])
    new_container.loadRecordsFromGdx(gdx_path)

    assert i._records is None  # records are not loaded yet.
    assert i.toList() == ["i1", "i2"]  # lazy load

    assert a._records is None  # records are not loaded yet.
    assert a.toList() == [("i1", 1.0), ("i2", 2.0)]  # lazy load

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

    # Test variable attribute domain after loadRecordsFromGdx
    m = Container()
    i = Set(m, "i", records=range(5))
    j = Set(m, "j", records=range(5))
    _ = Variable(m, "v", domain=[i, j])
    m.write(gdx_path)

    m = Container()
    m.loadRecordsFromGdx(gdx_path)
    i, j, v = m["i"], m["j"], m["v"]
    assert v.domain == [i, j]
    v.fx[i, j] = 5


def test_loadRecordsFromGdx_with_missing_symbols(tmp_path):
    m = gp.Container()
    i = gp.Set(m, "i", records=range(3))
    a = gp.Parameter(m, "a", domain=i, records=np.array([1, 2, 3]))
    m.write(tmp_path / "test.gdx")

    m2 = gp.Container()
    i = gp.Set(m2, "i")
    m2.loadRecordsFromGdx(tmp_path / "test.gdx")
    a = m2["a"]
    assert a.domain == [i]
    assert a.toList() == [("0", 1.0), ("1", 2.0), ("2", 3.0)], a.toList()

    m3 = gp.Container()
    i = gp.Set(m3, "i")
    m3.loadRecordsFromGdx(tmp_path / "test.gdx", symbol_names=["i", "a"])
    a = m3["a"]
    assert a.domain == [i]
    assert a.toList() == [("0", 1.0), ("1", 2.0), ("2", 3.0)], a.toList()


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
    directory = Path(__file__).parent / "gams_models"
    gams_models = {
        "LP": directory / "trnsport.gms",
        "MIP": directory / "prodsch.gms",
        "NLP": directory / "weapons.gms",
        "MCP": directory / "wallmcp.gms",
        "CNS": directory / "camcns.gms",
        "DNLP": directory / "linear.gms",
        "MINLP": directory / "meanvarx.gms",
        "QCP": directory / "himmel11.gms",
        "MIQCP": directory / "qalan.gms",
        "MPSGE": directory / "hansmge.gms",
    }

    for model in gams_models.values():
        with open(model) as file:
            content = file.read()

        with Container() as m:
            m.addGamsCode(content)


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
def test_write_load_on_demand(data, tmp_path):
    m, *_ = data
    i = Set(m, name="i", records=["i1"])
    p1 = Parameter(m, name="p1", domain=[i], records=[["i1", 1]])
    p2 = Parameter(m, name="p2", domain=[i])
    p2[i] = p1[i]

    gdx_path = str(tmp_path / "data.gdx")
    m.write(gdx_path)
    m = Container(
        load_from=gdx_path,
    )
    assert m["p2"].toList() == [("i1", 1.0)]


@pytest.mark.unit
def test_copy(data, tmp_path):
    _, canning_plants, markets, capacities, demands, distances = data

    copy_dir = str(tmp_path / "copy")
    m = Container(
        working_directory=copy_dir,
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

    with pytest.raises(ValidationError):
        m.copy(copy_dir)

    test_dir = str(tmp_path / "test")
    new_cont = m.copy(working_directory=test_dir)
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

    # Bare declarations (no records) are not synced eagerly; they are deferred
    # until the next real sync. Nothing has been sent to GAMS yet.
    assert m.generateGamsString() == ""

    # A real sync flushes all pending declarations together in a single block.
    m._synch_with_gams()
    generated = m.generateGamsString()
    expected = "$onMultiR\n$onUNDF\n$onDotL\nSet i(*) / /;\nAlias(i,a);\nParameter p / /;\nfree Variable v / /;\nEquation e / /;\n$offDotL\n$offUNDF\n$offMulti\n"
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
def test_write(data, tmp_path):
    m, *_ = data
    gdx_path = str(tmp_path / "test.gdx")

    from gamspy import SpecialValues

    _ = Parameter(m, "a", records=SpecialValues.EPS)
    m.write(gdx_path, eps_to_zero=True)

    m = Container(load_from=gdx_path)
    assert int(m["a"].toValue()) == 0

    m = gp.Container()
    _ = gp.Parameter(m, "a", records=gp.SpecialValues.EPS)
    m.write(gdx_path, eps_to_zero=False)

    m = gp.Container(load_from=gdx_path)
    assert str(m["a"].records.value.item()) != "0.0"

    with pytest.warns(DeprecationWarning):
        m.write(gdx_path, mode="string")

    m = gp.Container()
    _ = gp.Parameter(m, "a", records=5)
    _ = gp.Parameter(m, "b", records=10)
    m.write(gdx_path, symbol_names={"a": "c", "b": "d"})

    m = gp.Container(load_from=gdx_path)
    assert list(m.data.keys()) == ["c", "d"]

    m.close()


@pytest.mark.unit
def test_read(data, tmp_path):
    m, *_ = data
    gdx_path = str(tmp_path / "test.gdx")

    _ = Parameter(m, "a", records=5)
    m.write(gdx_path)

    m = Container()
    with pytest.warns(DeprecationWarning):
        m.read(gdx_path, load_records=False)

    assert m["a"]._records is None

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
        # Bare declarations are deferred and flushed together on the next sync,
        # producing a single .gms file rather than one per declaration.
        m._synch_with_gams()

    test_keep_success()
    gc.collect()
    assert os.path.exists(working_directory)
    assert len(glob.glob(os.path.join(working_directory, "*.gms"))) == 1

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
    # The bare declaration is deferred, so only the failing assignment syncs,
    # producing a single .gms file.
    assert len(glob.glob(os.path.join(working_directory, "*.gms"))) == 1

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
def test_read_from_gdx(data, tmp_path):
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
    gdx_path = str(tmp_path / "out.gdx")
    m.write(gdx_path)
    m.close()

    m = Container(load_from=gdx_path)
    i, k, c, x, supply = m["i"], m["k"], m["c"], m["x"], m["supply"]
    assert supply._records is None
    assert x._records is None
    assert c._records is None
    assert i._records is None
    assert supply.toList() == [
        ("seattle", 350.0),
        ("san-diego", 550.0),
    ]
    assert x.toList() == [
        ("seattle", "new-york", 50.0),
        ("seattle", "chicago", 300.0),
        ("seattle", "topeka", 0.0),
        ("san-diego", "new-york", 275.0),
        ("san-diego", "chicago", 0.0),
        ("san-diego", "topeka", 275.0),
    ]
    assert c.toList() == [
        ("seattle", "new-york", 0.225),
        ("seattle", "chicago", 0.153),
        ("seattle", "topeka", 0.162),
        ("san-diego", "new-york", 0.225),
        ("san-diego", "chicago", 0.162),
        ("san-diego", "topeka", 0.126),
    ]

    assert i.toList() == ["seattle", "san-diego"]
    assert k.toList() == ["seattle", "san-diego"]
    m.close()


@pytest.mark.unit
def test_read_rename(data, tmp_path):
    m, *_ = data
    gdx_path = str(tmp_path / "rename.gdx")

    _ = Parameter(m, "X", records=5)
    m.write(gdx_path)
    m.close()

    # Rename a scalar parameter from a GDX file.
    m2 = Container()
    m2.read(gdx_path, symbol_names={"X": "A"})
    assert "A" in m2.data
    assert "X" not in m2.data
    assert m2["A"].toValue() == 5
    m2.close()


@pytest.mark.unit
def test_read_rename_domain(data, tmp_path):
    m, *_ = data
    gdx_path = str(tmp_path / "rename_domain.gdx")

    i = Set(m, "i", records=["i1", "i2"])
    _ = Parameter(m, "p", domain=i, records=[("i1", 1), ("i2", 2)])
    m.write(gdx_path)
    m.close()

    # Rename a domain set and the parameter indexed over it in one read.
    m2 = Container()
    m2.read(gdx_path, symbol_names={"i": "j", "p": "q"})
    assert list(m2.data.keys()) == ["j", "q"]
    # The renamed parameter's domain should point at the renamed set object.
    assert m2["q"].domain == [m2["j"]]
    assert m2["j"].toList() == ["i1", "i2"]
    assert m2["q"].toList() == [("i1", 1.0), ("i2", 2.0)]
    m2.close()


@pytest.mark.unit
def test_read_rename_from_gamspy_container(data):
    m, *_ = data
    i = Set(m, "i", records=["i1", "i2"])
    _ = Parameter(m, "p", domain=i, records=[("i1", 1), ("i2", 2)])

    m2 = Container()
    m2.read(m, symbol_names={"i": "j", "p": "q"})
    assert list(m2.data.keys()) == ["j", "q"]
    assert m2["q"].domain == [m2["j"]]
    assert m2["j"].toList() == ["i1", "i2"]
    assert m2["q"].toList() == [("i1", 1.0), ("i2", 2.0)]
    m2.close()


@pytest.mark.unit
def test_read_rename_from_transfer_container():
    gt_container = gt.Container(system_directory=gamspy_base.directory)
    gt_i = gt.Set(gt_container, "i", records=["i1", "i2"])
    _ = gt.Parameter(gt_container, "p", domain=gt_i, records=[("i1", 1), ("i2", 2)])

    m2 = Container()
    m2.read(gt_container, symbol_names={"i": "j", "p": "q"})
    assert list(m2.data.keys()) == ["j", "q"]
    assert m2["q"].domain == [m2["j"]]
    assert m2["j"].toList() == ["i1", "i2"]
    assert m2["q"].toList() == [("i1", 1.0), ("i2", 2.0)]
    m2.close()


@pytest.mark.unit
def test_read_rename_errors(data, tmp_path):
    m, *_ = data
    gdx_path = str(tmp_path / "rename_err.gdx")

    _ = Parameter(m, "X", records=5)
    _ = Parameter(m, "Y", records=10)
    m.write(gdx_path)
    m.close()

    # Target name already exists in the container.
    m2 = Container()
    _ = Parameter(m2, "A", records=1)
    with pytest.raises(ValidationError):
        m2.read(gdx_path, symbol_names={"X": "A"})
    m2.close()

    # Two source symbols mapped to the same target name.
    m3 = Container()
    with pytest.raises(ValidationError):
        m3.read(gdx_path, symbol_names={"X": "A", "Y": "A"})
    m3.close()


@pytest.mark.unit
def test_output(tmp_path):
    path = str(tmp_path / (str(uuid.uuid4()) + ".py"))
    with open(path, "w") as file:
        file.write(
            "import sys\nfrom gamspy import Container, Set\nm = Container(output=sys.stdout)\ni = Set(m, records=['i1', 'i2'])\nj = Set(m)"
        )

    process = subprocess.run(
        [sys.executable, path], capture_output=True, check=True, text=True
    )
    assert process.stdout


@pytest.mark.unit
def test_restart():
    m = Container()
    save_path = os.path.join(m.working_directory, "save.g00")
    m._options._set_extra_options({"save": save_path})
    i = Set(m, "i", records=["i1", "i2"])
    _ = Parameter(m, "a", domain=i, records=[("i1", 1), ("i2", 2)])
    assert os.path.exists(save_path)
    m.close()

    m = Container(load_from=save_path)
    assert "i" in m.data
    assert "a" in m.data
    assert m["i"].toList() == ["i1", "i2"]
    assert m["a"].toList() == [("i1", 1), ("i2", 2)]
    assert m["a"].domain == [m["i"]]
    _ = Set(m, "j", records=range(6))
    assert list(m.data.keys()) == ["i", "a", "j"]
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
def test_deserialize_variable_indexed_bound_write(data, tmp_path):
    m, *_ = data
    t = Set(m, "t", records=["a", "b"])
    v = Variable(m, "v", type="positive", domain=t)
    v.lo[t] = 0

    serialization_path = os.path.join(tmp_path, "var_write.zip")
    serialize(m, serialization_path)
    m2 = deserialize(serialization_path)

    assert m2["v"].lo.domain == [m2["t"]]

    m2["v"].lo[m2["t"]] = -5
    assert all(val == -5 for val in m2["v"].records["lower"])


@pytest.mark.unit
def test_deserialize_variable_level_read(data, tmp_path):
    m, *_ = data
    i = Set(m, "i", records=["i1", "i2"])
    x = Variable(m, "x", type="positive", domain=i)
    e = Equation(m, "e", domain=i)
    e[i] = x[i] >= 2

    model = Model(
        m,
        name="m_var_read",
        equations=m.getEquations(),
        problem="LP",
        sense=Sense.MIN,
        objective=Sum(i, x[i]),
    )
    model.solve()

    serialization_path = os.path.join(tmp_path, "var_read.zip")
    serialize(m, serialization_path)
    m2 = deserialize(serialization_path)

    assert m2["x"].l.domain == [m2["i"]]

    out = Parameter(m2, "out", domain=m2["i"])
    out[m2["i"]] = m2["x"].l[m2["i"]]  # used to raise ValidationError
    assert all(abs(val - 2.0) < 1e-6 for val in out.records["value"])


@pytest.mark.unit
def test_deserialize_equation_indexed_bound_write(data, tmp_path):
    m, *_ = data
    t = Set(m, "t", records=["a", "b"])
    v = Variable(m, "v", domain=t)
    eq = Equation(m, "eq", domain=t)
    eq[t] = v[t] == 1

    serialization_path = os.path.join(tmp_path, "eq_write.zip")
    serialize(m, serialization_path)
    m2 = deserialize(serialization_path)

    assert m2["eq"].lo.domain == [m2["t"]]

    m2["eq"].lo[m2["t"]] = -1  # used to raise ValidationError
    assert all(val == -1 for val in m2["eq"].records["lower"])


@pytest.mark.unit
def test_deserialize_equation_level_read(data, tmp_path):
    m, *_ = data
    i = Set(m, "i", records=["i1", "i2"])
    x = Variable(m, "x", type="positive", domain=i)
    e = Equation(m, "e", domain=i)
    e[i] = x[i] >= 2

    model = Model(
        m,
        name="m_eq_read",
        equations=m.getEquations(),
        problem="LP",
        sense=Sense.MIN,
        objective=Sum(i, x[i]),
    )
    model.solve()

    serialization_path = os.path.join(tmp_path, "eq_read.zip")
    serialize(m, serialization_path)
    m2 = deserialize(serialization_path)

    assert m2["e"].l.domain == [m2["i"]]

    levels = Parameter(m2, "lvl", domain=m2["i"])
    levels[m2["i"]] = m2["e"].l[m2["i"]]  # used to raise ValidationError
    assert all(abs(val - 2.0) < 1e-6 for val in levels.records["value"])


@pytest.mark.unit
def test_deserialize_set_alias_subset_regression(data, tmp_path):
    m, *_ = data
    i = Set(m, "i", records=["i1", "i2", "i3"])
    Set(m, "j", domain=i, records=["i1", "i3"])
    Alias(m, "ii", i)

    serialization_path = os.path.join(tmp_path, "set_alias.zip")
    serialize(m, serialization_path)
    m2 = deserialize(serialization_path)

    i2, j2, ii2 = m2["i"], m2["j"], m2["ii"]

    ordinals = Parameter(m2, "ordinals", domain=i2)
    ordinals[i2] = Ord(i2)
    assert list(ordinals.records["value"]) == [1, 2, 3]

    filtered = Parameter(m2, "filtered", domain=i2)
    filtered[i2].where[j2[i2]] = 1
    assert sorted(filtered.records["i"].tolist()) == ["i1", "i3"]

    total = Parameter(m2, "total")
    total[...] = Sum(ii2, Ord(ii2))
    assert total.records["value"][0] == 6


@pytest.mark.unit
def test_deserialize_parameter_indexed_regression(data, tmp_path):
    m, *_ = data
    i = Set(m, "i", records=["i1", "i2"])
    Parameter(m, "a", domain=i, records=[["i1", 10], ["i2", 20]])

    serialization_path = os.path.join(tmp_path, "param.zip")
    serialize(m, serialization_path)
    m2 = deserialize(serialization_path)

    i2, a2 = m2["i"], m2["a"]
    b = Parameter(m2, "b", domain=i2)
    b[i2] = a2[i2]
    assert sorted(b.records["value"].tolist()) == [10, 20]


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

    gp.set_options({"USE_PY_VAR_NAME": "yes-or-autogenerate"})


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
        name="xdice",
        equations=m.getEquations(),
        problem=gp.Problem.MIP,
        sense=gp.Sense.MAX,
        objective=wnx,
    )

    # Should throw license error since we are using the demo license.
    with pytest.raises(GamspyException):
        xdice.solve()


@pytest.mark.unit
def test_writeSolverOptions():
    m = Container()
    m.writeSolverOptions(
        "conopt",
        solver_options={"rtmaxv": "1.e12"},
    )
    solver_options_path = os.path.join(m.working_directory, "conopt4.opt")
    assert os.path.exists(solver_options_path)
    with open(solver_options_path) as file:
        assert "rtmaxv" in file.read()

    m.writeSolverOptions("conopt", solver_options={"rtmaxv": "1.e12"}, file_number=2)
    solver_options_path = os.path.join(m.working_directory, "conopt4.op2")
    assert os.path.exists(solver_options_path)

    m.writeSolverOptions("conopt", solver_options={"rtmaxv": "1.e12"}, file_number=9)
    solver_options_path = os.path.join(m.working_directory, "conopt4.op9")
    assert os.path.exists(solver_options_path)

    m.writeSolverOptions("conopt", solver_options={"rtmaxv": "1.e12"}, file_number=10)
    solver_options_path = os.path.join(m.working_directory, "conopt4.o10")
    assert os.path.exists(solver_options_path)

    m.writeSolverOptions("conopt", solver_options={"rtmaxv": "1.e12"}, file_number=99)
    solver_options_path = os.path.join(m.working_directory, "conopt4.o99")
    assert os.path.exists(solver_options_path)

    m.writeSolverOptions("conopt", solver_options={"rtmaxv": "1.e12"}, file_number=100)
    solver_options_path = os.path.join(m.working_directory, "conopt4.100")
    assert os.path.exists(solver_options_path)

    m.writeSolverOptions("conopt", solver_options={"rtmaxv": "1.e12"}, file_number=999)
    solver_options_path = os.path.join(m.working_directory, "conopt4.999")
    assert os.path.exists(solver_options_path)

    m.writeSolverOptions("conopt", solver_options={"rtmaxv": "1.e12"}, file_number=1234)
    solver_options_path = os.path.join(m.working_directory, "conopt4.1234")
    assert os.path.exists(solver_options_path)

    # Read solver options from an existing file
    with tempfile.TemporaryDirectory() as tmpdir:
        options_path = os.path.join(tmpdir, "my_solver_options.opt")
        with open(options_path, "w") as file:
            file.write("rtmaxv 1.e12")

        m.writeSolverOptions("conopt", options_path)
        assert os.path.exists(os.path.join(m.working_directory, "conopt4.opt"))

        with open(options_path) as file:
            original_file_content = file.read()

        with open(os.path.join(m.working_directory, "conopt4.opt")) as file:
            copied_file_content = file.read()

        assert original_file_content == copied_file_content

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


@pytest.mark.unit
def test_python_name():
    m = gp.Container()

    i = m.addSet()
    assert i.name == "i"

    a = m.addAlias(alias_with=i)
    assert a.name == "a"

    plusminus = m.addParameter(domain=[i])
    assert plusminus.name == "plusminus"

    var = m.addVariable(domain=[i])
    assert var.name == "var"

    eq = m.addEquation(domain=[i])
    assert eq.name == "eq"


@pytest.mark.unit
def test_deepcopy():
    m = gp.Container()

    # deepcopy set
    c = gp.Set(m)
    c2 = copy.deepcopy(c)
    assert id(c) != id(c2)

    t = gp.Set(m)

    # deepcopy alias
    a = gp.Alias(m, alias_with=c)
    a2 = gp.Alias(m, alias_with=c)
    assert id(a) != id(a2)

    # deepcopy parameter
    b = gp.Parameter(m)
    b2 = gp.Parameter(m)
    assert id(b) != id(b2)

    # deepcopy variable
    x = gp.Variable(m, domain=[c, t])
    x2 = gp.Variable(m, domain=[c, t])
    assert id(x) != id(x2)

    # deepcopy equation
    e = gp.Equation(m, domain=[c, t])
    e2 = gp.Equation(m, domain=[c, t])
    assert id(e) != id(e2)

    # deepcopy expression
    expr = gp.Sum((c, t), x[c, t])
    expr2 = copy.deepcopy(expr)
    assert id(expr) != id(expr2)

    # deepcopy model
    model = gp.Model(m, equations=[e])
    model2 = copy.deepcopy(model)
    assert id(model) != id(model2)

    # deepcopy a container
    m2 = copy.deepcopy(m)
    assert id(m) != id(m2)
    assert len(m.data) == len(m2.data)

    # add a new symbol to the copied container
    _ = gp.Parameter(m2)


@pytest.mark.unit
def test_write_with_universe_alias(tmp_path):
    m = gp.Container()
    i = gp.Set(m, "i")
    uni = gp.UniverseAlias(m, "uni")
    i["2"].where[False] = False

    assert "2" in uni.toList()

    gdx_file = str(tmp_path / "myGDX.gdx")
    m.write(gdx_file)

    mm = gp.Container(load_from=gdx_file)

    assert "2" in mm["uni"].toList()


@pytest.mark.unit
def test_write_compress(tmp_path):
    gdx_file_path = tmp_path / "temp.gdx"
    gdx_file_path2 = tmp_path / "temp2.gdx"
    gdx_file_path3 = tmp_path / "temp3.gdx"

    m = gp.Container()
    _ = gp.Set(m, "i", records=range(2000))
    _ = gp.Set(m, "j", records=range(2000))
    _ = gp.Set(m, "k", records=range(2000))
    m.write(gdx_file_path, compress=True)
    m.write(gdx_file_path2, compress=False)
    m.write(gdx_file_path3)  # inherit from env

    assert os.path.getsize(gdx_file_path) < os.path.getsize(gdx_file_path2)
    assert os.path.getsize(gdx_file_path) < os.path.getsize(
        gdx_file_path3  # do not compress by default
    )

    # should be equivalent to compress=True
    os.environ["GDXCOMPRESS"] = "1"
    gdx_file_path4 = tmp_path / "temp4.gdx"
    m = gp.Container()
    _ = gp.Set(m, "i", records=range(2000))
    _ = gp.Set(m, "j", records=range(2000))
    _ = gp.Set(m, "k", records=range(2000))
    m.write(gdx_file_path4)
    assert os.path.getsize(gdx_file_path4) == os.path.getsize(gdx_file_path)

    # should be equivalent to compress=False
    os.environ["GDXCOMPRESS"] = "0"
    gdx_file_path5 = tmp_path / "temp5.gdx"
    m = gp.Container()
    _ = gp.Set(m, "i", records=range(2000))
    _ = gp.Set(m, "j", records=range(2000))
    _ = gp.Set(m, "k", records=range(2000))
    m.write(gdx_file_path5)
    assert os.path.getsize(gdx_file_path5) == os.path.getsize(gdx_file_path2)


@pytest.mark.unit
def test_setRecords_None():
    m = gp.Container()

    i = gp.Set(m, records=["i1", "i2"])
    i.setRecords(None)
    assert i.records is None

    a = gp.Parameter(m, records=5)
    a.setRecords(None)
    assert a.records is None

    i2 = gp.Set(m, records=["i1", "i2"])
    a2 = gp.Parameter(m, domain=i2, records=[("i1", 1), ("i2", 2)])
    a2.setRecords(None)
    assert a2.records is None

    v = gp.Variable(m)
    v.setRecords(None)
    assert v.records is None

    vp = gp.Variable(m, type="positive")
    vp.setRecords(None)
    assert vp.records is None

    e = gp.Equation(m)
    e.setRecords(None)
    assert e.records is None

    el = gp.Equation(m)
    el[...] = v <= vp
    el.setRecords(None)
    assert el.records is None

    i2 = gp.Set(m, records=["i1", "i2"])
    a1 = gp.Alias(m, alias_with=i2)
    a1.setRecords(None)
    assert a1.records is None
    assert i2.records is None


@pytest.mark.unit
def test_getitem():
    m = gp.Container()
    with pytest.raises(KeyError, match="does not exist in the Container"):
        m["i"]

    gp.Set(m, "some_set")

    with pytest.raises(KeyError, match="Did you mean"):
        m["som_set"]


@pytest.mark.unit
def test_gp_to_gp():
    # Test GP to GP with an empty GP container
    m = gp.Container(system_directory=gamspy_base.directory)
    i = gp.Set(m, "i")
    gp.Alias(m, "j", alias_with=i)
    gp.Parameter(m, "p")
    gp.Variable(m, "v")
    gp.Equation(m, "e", type="eq")
    gp.UniverseAlias(m, "universe")

    m2 = gp.Container(m)
    assert list(m2.data.keys()) == ["i", "j", "p", "v", "e", "universe"]

    # Test GP to GP with records
    m = gp.Container(system_directory=gamspy_base.directory)
    i = gp.Set(m, "i", records=["i1", "i2", "i3"])
    j = gp.Alias(m, "j", alias_with=i)
    p = gp.Parameter(m, "p", domain=i, records=[("i1", 1), ("i2", 2), ("i3", 3)])
    v = gp.Variable(m, "v", domain=i)
    v.generateRecords()
    e = gp.Equation(m, "e", domain=i, type="eq")
    e.generateRecords()
    universe = gp.UniverseAlias(m, "universe")

    m2 = gp.Container(m)
    assert list(m2.data.keys()) == ["i", "j", "p", "v", "e", "universe"]
    assert i.records.equals(m2["i"].records)
    assert j.records.equals(m2["j"].records)
    assert p.records.equals(m2["p"].records)
    assert v.records.equals(m2["v"].records)
    assert e.records.equals(m2["e"].records)
    assert universe.records.equals(m2["universe"].records)


@pytest.mark.unit
def test_gtp_to_gp():
    # Test GTP to GP with an empty GTP container
    m = gt.Container(system_directory=gamspy_base.directory)
    i = gt.Set(m, "i")
    gt.Alias(m, "j", alias_with=i)
    gt.Parameter(m, "p")
    gt.Variable(m, "v")
    gt.Equation(m, "e", type="eq")
    gt.UniverseAlias(m, "universe")

    m2 = gp.Container(m)
    assert list(m2.data.keys()) == ["i", "j", "p", "v", "e", "universe"]

    # Test GTP to GP with records
    m = gt.Container(system_directory=gamspy_base.directory)
    i = gt.Set(m, "i", records=["i1", "i2", "i3"])
    j = gt.Alias(m, "j", alias_with=i)
    p = gt.Parameter(m, "p", domain=i, records=[("i1", 1), ("i2", 2), ("i3", 3)])
    v = gt.Variable(m, "v", domain=i)
    v.generateRecords()
    e = gt.Equation(m, "e", domain=i, type="eq")
    e.generateRecords()
    universe = gt.UniverseAlias(m, "universe")

    m2 = gp.Container(m)
    assert list(m2.data.keys()) == ["i", "j", "p", "v", "e", "universe"]
    assert i.records.equals(m2["i"].records)
    assert j.records.equals(m2["j"].records)
    assert p.records.equals(m2["p"].records)
    assert v.records.equals(m2["v"].records)
    assert e.records.equals(m2["e"].records)
    assert universe.records.equals(m2["universe"].records)


@pytest.mark.unit
def test_gtp_to_gp_dirty():
    # Test dirty GTP to GP (should fail)
    m = gt.Container(system_directory=gamspy_base.directory)
    i = gt.Set(m, "i", records=["i2", "i2", "i3"])
    gt.Alias(m, "j", alias_with=i)
    gt.Parameter(m, "p", domain=i, records=[("i2", 1), ("i2", 2), ("i3", 3)])

    with pytest.raises(GdxException):
        _ = gp.Container(m)


@pytest.mark.unit
def test_addGamsCode_with_equations():
    m = gp.Container()
    i = gp.Set(m, name="i", records=range(1, 10))
    Z = gp.Variable(m, name="Z")
    gp.Variable(m, name="X", domain=i, type="binary")

    m.addGamsCode(r"""
    Equation eq;
    eq.. Z =E= Sum(i, X(i));
    """)

    assert m._arbitrary_code_executed

    model = gp.Model(
        m, name="test", equations=m.getEquations(), objective=Z, sense="min"
    )

    model.solve()


@pytest.mark.unit
def test_addGamsCode_with_debugging_level_keep():
    m = gp.Container(debugging_level="keep")
    m.addGamsCode("Set i / i1 /;")


def test_describe_symbols():
    m = gp.Container()
    assert m.describeAliases() is None

    i = gp.Set(m, "i")
    i2 = gp.Set(m, "i2")
    _ = gp.Alias(m, "a", i)
    _ = gp.Alias(m, "a2", i2)
    _ = gp.Parameter(m, "p")
    _ = gp.Parameter(m, "p2")
    _ = gp.Variable(m, "v")
    _ = gp.Variable(m, "v2")
    _ = gp.Equation(m, "e")
    _ = gp.Equation(m, "e2")
    _ = gp.UniverseAlias(m, "u")
    _ = gp.UniverseAlias(m, "u2")

    df = pd.DataFrame(
        {
            "name": ["i", "i2"],
            "is_singleton": [False, False],
            "domain": [["*"], ["*"]],
            "domain_type": ["none", "none"],
            "dimension": [1, 1],
            "number_records": [0, 0],
            "sparsity": [np.nan, np.nan],
        }
    )

    pd.testing.assert_frame_equal(m.describeSets(), df)

    df_aliases = pd.DataFrame(
        {
            "name": ["a", "a2", "u", "u2"],
            "alias_with": ["i", "i2", "*", "*"],
            "is_singleton": [False, False, False, False],
            "domain": [["*"], ["*"], ["*"], ["*"]],
            "domain_type": ["none", "none", "none", "none"],
            "dimension": [1, 1, 1, 1],
            "number_records": [0, 0, 0, 0],
            "sparsity": [np.nan, np.nan, 0.0, 0.0],
        }
    )
    pd.testing.assert_frame_equal(m.describeAliases(), df_aliases)

    df_parameters = pd.DataFrame(
        {
            "name": ["p", "p2"],
            "domain": [[], []],
            "domain_type": ["none", "none"],
            "dimension": [0, 0],
            "number_records": [0, 0],
            "min": [None, None],
            "mean": [None, None],
            "max": [None, None],
            "where_min": [None, None],
            "where_max": [None, None],
            "sparsity": [np.nan, np.nan],
        }
    )
    pd.testing.assert_frame_equal(m.describeParameters(), df_parameters)

    df_variables = pd.DataFrame(
        {
            "name": ["v", "v2"],
            "type": ["free", "free"],
            "domain": [[], []],
            "domain_type": ["none", "none"],
            "dimension": [0, 0],
            "number_records": [0, 0],
            "sparsity": [np.nan, np.nan],
            "min_level": [None, None],
            "mean_level": [None, None],
            "max_level": [None, None],
            "where_max_abs_level": [None, None],
        }
    )
    pd.testing.assert_frame_equal(m.describeVariables(), df_variables)

    df_equations = pd.DataFrame(
        {
            "name": ["e", "e2"],
            "type": ["eq", "eq"],
            "domain": [[], []],
            "domain_type": ["none", "none"],
            "dimension": [0, 0],
            "number_records": [0, 0],
            "sparsity": [np.nan, np.nan],
            "min_level": [None, None],
            "mean_level": [None, None],
            "max_level": [None, None],
            "where_max_abs_level": [None, None],
        }
    )
    pd.testing.assert_frame_equal(m.describeEquations(), df_equations)


@pytest.mark.unit
def test_generateRecords():
    m = gp.Container()
    i = gp.Set(m, "i", records=range(10))
    j = gp.Set(m, "j", records=range(10))
    k = gp.Set(m, "k", domain=i)
    k.generateRecords(density=0.1, seed=42)
    k2 = gp.Set(m, "k2", domain=[i, j])
    k2.generateRecords(density=[0.01, 0.2])
    p = gp.Parameter(m, "p", domain=i)
    p.generateRecords(density=0.1, seed=42)
    p2 = gp.Parameter(m, "p2", domain=[i, j])
    p2.generateRecords(density=[0.1, 0.2])
    v = gp.Variable(m, "v", domain=i)
    v.generateRecords(density=0.15, seed=42)
    v2 = gp.Variable(m, "v2", domain=[i, j])

    with pytest.raises(TypeError, match="must be callable"):
        v2.generateRecords(func={"level": "bla"})

    def marginal_values(seed, size):
        rng = np.random.default_rng(seed)
        return rng.normal(5, 1.2, size=size)

    v2.generateRecords(density=[0.1, 0.2], func={"marginal": marginal_values})

    i2 = gp.Set(m, "i2", records=range(5))
    p3 = gp.Parameter(m, "p3", domain=i2)
    p3.generateRecords(density=0)


@pytest.mark.unit
def test_symbol_equals():
    import gamspy as gp

    m = gp.Container()

    i1 = gp.Set(m, "i1", records=["a", "b"], description="Set A")
    i2 = gp.Set(m, "i2", records=["a", "b"], description="Set B")
    i3 = gp.Set(m, "i3", records=["a", "c"])

    i4 = gp.Set(m, "i4", records=[("a", "text a"), ("b", "text b")])
    i5 = gp.Set(m, "i5", records=[("a", "text a"), ("b", "text c")])

    # Metadata check (names/descriptions differ)
    assert not i1.equals(i2)
    assert i1.equals(i2, check_meta_data=False)

    # Records check
    assert not i1.equals(i3, check_meta_data=False)

    # Element text check
    assert not i4.equals(i5, check_meta_data=False)
    assert i4.equals(i5, check_element_text=False, check_meta_data=False)

    p1 = gp.Parameter(m, "p1", domain=[i1], records=[("a", 1.0001), ("b", 2.0)])
    p2 = gp.Parameter(m, "p2", domain=[i1], records=[("a", 1.0002), ("b", 2.0)])
    p3 = gp.Parameter(m, "p3", domain=[i1], records=[("a", 1.0001), ("b", 2.0)])

    # Exact records check
    assert p1.equals(p3, check_meta_data=False)
    assert not p1.equals(p2, check_meta_data=False)

    # Tolerance check
    assert p1.equals(p2, check_meta_data=False, atol=1e-3)

    v1 = gp.Variable(m, "v1", domain=[i1], type="positive")
    v2 = gp.Variable(m, "v2", domain=[i1], type="positive")
    v3 = gp.Variable(m, "v3", domain=[i1], type="free")

    v1.l[...] = 5
    v2.l[...] = 5

    # Exact equivalence
    assert v1.equals(v2, check_meta_data=False)

    # Type mismatch check
    assert not v1.equals(v3, check_meta_data=False)

    # Specific attribute change
    v2.l["a"] = 10
    assert not v1.equals(v2, check_meta_data=False)

    # Selective column check (marginal is still default 0.0 for both)
    assert v1.equals(v2, columns=["marginal"], check_meta_data=False)

    e1 = gp.Equation(m, "e1", domain=[i1], type="regular")
    e2 = gp.Equation(m, "e2", domain=[i1], type="regular")

    e1.m[...] = 2
    e2.m[...] = 2

    # Exact equivalence
    assert e1.equals(e2, check_meta_data=False)

    # Tolerance on specific attribute change
    e2.m["a"] = 2.0001
    assert not e1.equals(e2, check_meta_data=False)
    assert e1.equals(e2, check_meta_data=False, atol=1e-3)

    # Selective column check (level is still default 0.0 for both)
    assert e1.equals(e2, columns=["level"], check_meta_data=False)

    s1 = gp.Parameter(m, "s1", records=5.0)
    s2 = gp.Parameter(m, "s2", records=5.0)
    s3 = gp.Parameter(m, "s3", records=10.0)
    s4 = gp.Parameter(m, "s4", records=5.0001)

    # Exact match
    assert s1.equals(s2, check_meta_data=False)

    # Mismatch
    assert not s1.equals(s3, check_meta_data=False)

    # Tolerance match
    assert not s1.equals(s4, check_meta_data=False)  # Fails without tolerance
    assert s1.equals(s4, check_meta_data=False, atol=1e-3)  # Passes with tolerance

    # Special values check
    s_eps1 = gp.Parameter(m, "s_eps1", records=gp.SpecialValues.EPS)
    s_eps2 = gp.Parameter(m, "s_eps2", records=gp.SpecialValues.EPS)
    s_na = gp.Parameter(m, "s_na", records=gp.SpecialValues.NA)

    # Exact special value equivalence
    assert s_eps1.equals(s_eps2, check_meta_data=False)

    # Special value mismatch
    assert not s_eps1.equals(s_na, check_meta_data=False)
    assert not s1.equals(s_eps1, check_meta_data=False)


@pytest.mark.unit
def test_symbol_pivot():
    m = gp.Container()

    i = gp.Set(m, "i", records=["seattle", "san-diego"])
    j = gp.Set(m, "j", records=["new-york", "chicago", "topeka"])

    with pytest.raises(
        ValidationError,
        match="Pivoting operations only possible on symbols with dimension > 1",
    ):
        i.pivot()

    p_empty = gp.Parameter(m, "p_empty", domain=[i, j])

    with pytest.raises(ValidationError):
        p_empty.pivot()

    ij = gp.Set(
        m,
        "ij",
        domain=[i, j],
        records=[("seattle", "new-york"), ("san-diego", "chicago")],
    )
    df_set = ij.pivot()

    assert df_set.shape == (2, 2)
    # Default fill_value for Set is False
    assert df_set.loc["seattle", "new-york"]
    assert not df_set.loc["seattle", "chicago"]
    assert df_set.loc["san-diego", "chicago"]

    p = gp.Parameter(
        m,
        "p",
        domain=[i, j],
        records=[("seattle", "new-york", 10), ("san-diego", "chicago", 20)],
    )
    df_p = p.pivot(fill_value=0.0)

    assert df_p.shape == (2, 2)
    assert df_p.loc["seattle", "new-york"] == 10.0
    assert df_p.loc["san-diego", "new-york"] == 0.0

    v = gp.Variable(m, "v", domain=[i, j])
    v.l[i, j] = 5
    v.m["seattle", "new-york"] = 15

    # Variable pivot defaults to 'level'
    df_v_level = v.pivot()
    assert df_v_level.loc["seattle", "new-york"] == 5.0
    assert df_v_level.loc["seattle", "topeka"] == 5.0

    # Variable pivot overriding value to 'marginal'
    df_v_marginal = v.pivot(value="marginal", fill_value=0.0)
    assert df_v_marginal.loc["seattle", "new-york"] == 15.0
    assert df_v_marginal.loc["san-diego", "chicago"] == 0.0

    e = gp.Equation(m, "e", domain=[i, j])
    e.m[i, j] = 2.5
    e.l["san-diego", "topeka"] = -10

    df_e_level = e.pivot(value="level", fill_value=0.0)
    assert df_e_level.loc["san-diego", "topeka"] == -10.0
    assert df_e_level.loc["seattle", "new-york"] == 0.0

    df_e_marginal = e.pivot(value="marginal")
    assert df_e_marginal.loc["seattle", "new-york"] == 2.5

    k = gp.Set(m, "k", records=["k1", "k2"])
    p3 = gp.Parameter(
        m,
        "p3",
        domain=[i, j, k],
        records=[
            ("seattle", "new-york", "k1", 100),
            ("san-diego", "chicago", "k2", 50),
        ],
    )
    df_p3 = p3.pivot(index=["i", "j"], columns="k", fill_value=0.0)

    assert isinstance(df_p3.index, pd.MultiIndex)
    assert df_p3.loc[("seattle", "new-york"), "k1"] == 100.0
    assert df_p3.loc[("seattle", "new-york"), "k2"] == 0.0  # filled with 0.0
    assert df_p3.loc[("san-diego", "chicago"), "k1"] == 0.0  # filled with 0.0
    assert df_p3.loc[("san-diego", "chicago"), "k2"] == 50.0


@pytest.mark.unit
def test_pivot_restore_special_values():
    m = gp.Container()
    i = gp.Set(m, "i", records=["i1", "i2"])
    j = gp.Set(m, "j", records=["j1", "j2"])

    p = gp.Parameter(
        m,
        "p",
        domain=[i, j],
        records=[
            ("i1", "j1", 5.0),
            ("i1", "j2", gp.SpecialValues.NA),
            ("i2", "j1", gp.SpecialValues.UNDEF),
            ("i2", "j2", 10.0),
        ],
    )
    df_p = p.pivot(fill_value=0.0)
    # Check that the special values were correctly restored in the pivoted DataFrame
    assert df_p.shape == (2, 2)
    assert df_p.loc["i1", "j1"] == 5.0
    assert df_p.loc["i2", "j2"] == 10.0
    assert gp.SpecialValues.isNA(df_p.loc["i1", "j2"])
    assert gp.SpecialValues.isUndef(df_p.loc["i2", "j1"])

    v = gp.Variable(m, "v", domain=[i, j])
    v.l["i1", "j1"] = 1.0
    v.l["i2", "j2"] = gp.SpecialValues.NA

    # Variable pivot defaults to value="level"
    df_v = v.pivot(fill_value=0.0)

    assert df_v.shape == (2, 2)
    assert df_v.loc["i1", "j1"] == 1.0
    assert df_v.loc["i1", "j2"] == 0.0  # standard fill value
    assert gp.SpecialValues.isNA(df_v.loc["i2", "j2"])  # NA was restored


@pytest.mark.unit
def test_symbol_toDict():
    m = gp.Container()
    i = gp.Set(m, "i", records=["i1", "i2"])
    j = gp.Set(m, "j", records=["j1", "j2"])

    # Scalar Parameter
    p_scalar = gp.Parameter(m, "p_scalar", records=5)
    with pytest.raises(
        TypeError, match="is a scalar and cannot be converted into a dict"
    ):
        p_scalar.toDict()

    # 1D Parameter
    p1 = gp.Parameter(m, "p1", domain=[i], records=[("i1", 10), ("i2", 20)])
    assert p1.toDict() == {"i1": 10.0, "i2": 20.0}

    # 2D Parameter
    p2 = gp.Parameter(
        m, "p2", domain=[i, j], records=[("i1", "j1", 100), ("i2", "j2", 200)]
    )
    assert p2.toDict() == {("i1", "j1"): 100.0, ("i2", "j2"): 200.0}
    assert p2.toDict(orient="columns") == {
        "i": {0: "i1", 1: "i2"},
        "j": {0: "j1", 1: "j2"},
        "value": {0: 100.0, 1: 200.0},
    }

    # Invalid orient
    with pytest.raises(
        ValueError, match="Argument 'orient' expects one of the following"
    ):
        p1.toDict(orient="invalid_orient")

    # Scalar Variable
    v_scalar = gp.Variable(m, "v_scalar")
    with pytest.raises(
        TypeError, match="is a scalar and cannot be converted into a dict"
    ):
        v_scalar.toDict()

    # 1D Variable
    v1 = gp.Variable(m, "v1", domain=[i])
    v1.l["i1"] = 5
    v1.m["i2"] = 10

    # Variable toDict defaults to 'level'
    assert v1.toDict() == {"i1": 5.0, "i2": 0.0}

    # Multiple attributes requested -> transforms nested dicts
    assert v1.toDict(columns=["level", "marginal"]) == {
        "i1": {"level": 5.0, "marginal": 0.0},
        "i2": {"level": 0.0, "marginal": 10.0},
    }

    # 2D Variable
    v2 = gp.Variable(m, "v2", domain=[i, j])
    v2.l["i1", "j2"] = 15
    assert v2.toDict() == {("i1", "j2"): 15.0}
    assert v2.toDict(orient="columns") == {
        "i": {0: "i1"},
        "j": {0: "j2"},
        "level": {0: 15.0},
    }

    # Invalid columns
    with pytest.raises(
        TypeError, match="Argument 'columns' must be a subset of the following"
    ):
        v1.toDict(columns=["level", "invalid_col"])

    e1 = gp.Equation(m, "e1", domain=[i])
    e1.m["i2"] = 2.5

    # Check explicitly requesting 'marginal' instead of 'level'
    assert e1.toDict(columns="marginal") == {"i2": 2.5}


def test_case_insensitivity(tmp_path):
    gdx_path = tmp_path / "test.gdx"
    m = gp.Container()
    gp.Set(m, "i", records=range(3))
    gp.Set(m, "J", records=range(3))
    m.write(gdx_path)

    m = gp.Container(load_from=gdx_path)
    m["j"]  # should work even though the casing is wrong

    m = gp.Container()
    m.read(gdx_path)
    m["j"]  # should work even though the casing is wrong

    m = gp.Container()
    m.loadRecordsFromGdx(gdx_path)
    m["j"]  # should work even though the casing is wrong

    m = gp.Container()
    gp.Set(m, "j")
    m.loadRecordsFromGdx(gdx_path)
    m["j"]  # should work even though the casing is wrong
    assert m["j"].toList() == ["0", "1", "2"]
