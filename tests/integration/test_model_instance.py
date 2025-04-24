from __future__ import annotations

import glob
import math
import os
import pathlib
import platform
import signal
import subprocess
import sys
import tempfile
import time

import gamspy_base
import pandas as pd
import pytest

import gamspy as gp
import gamspy.utils as utils
from gamspy import (
    Alias,
    Container,
    Equation,
    FreezeOptions,
    Model,
    ModelInstanceOptions,
    ModelStatus,
    Options,
    Parameter,
    Problem,
    Product,
    Sense,
    Set,
    SolveStatus,
    Sum,
    Variable,
    VariableType,
)
from gamspy._database import (
    Database,
    GamsEquation,
    GamsParameter,
    GamsSet,
    GamsVariable,
)
from gamspy._workspace import Workspace
from gamspy.exceptions import GamspyException, ValidationError

pytestmark = pytest.mark.integration

try:
    from dotenv import load_dotenv

    load_dotenv(os.getcwd() + os.sep + ".env")
except Exception:
    pass


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

    m.close()

    files = glob.glob("_*")
    for file in files:
        if os.path.isfile(file):
            os.remove(file)

    if os.path.exists("dict.txt"):
        os.remove("dict.txt")

    if os.path.exists("gams.gms"):
        os.remove("gams.gms")


@pytest.mark.skipif(
    platform.system() == "Darwin" and platform.machine() == "x86_64",
    reason="Darwin runners are not dockerized yet.",
)
def test_parameter_change(data):
    m, canning_plants, markets, capacities, demands, distances = data
    i = Set(m, name="i", records=canning_plants)
    j = Set(m, name="j", records=markets)

    a = Parameter(m, name="a", domain=[i], records=capacities)
    b = Parameter(m, name="b", domain=[j], records=demands)
    d = Parameter(
        m,
        name="d",
        domain=[i, j],
        records=distances,
        is_miro_input=True,
    )
    c = Parameter(m, name="c", domain=[i, j])
    bmult = Parameter(m, name="bmult", records=1)
    c[i, j] = 90 * d[i, j] / 1000

    x = Variable(m, name="x", domain=[i, j], type="Positive")
    z = Variable(m, name="z")

    cost = Equation(m, name="cost")
    supply = Equation(m, name="supply", domain=[i])
    demand = Equation(m, name="demand", domain=[j])

    cost[...] = z == Sum((i, j), c[i, j] * x[i, j])
    supply[i] = Sum(j, x[i, j]) <= a[i]
    demand[j] = Sum(i, x[i, j]) >= bmult * b[j]

    transport = Model(
        m,
        name="transport",
        equations=m.getEquations(),
        problem="LP",
        sense=Sense.MIN,
        objective=z,
    )

    bmult_list = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    results = [
        92.204,
        107.572,
        122.940,
        138.307,
        153.675,
        169.94250000000002,
        185.58,
        201.21750000000003,
    ]

    transport.freeze(modifiables=[bmult])
    for b_value, result in zip(bmult_list, results):
        bmult[...] = b_value
        summary = transport.solve(solver="conopt")
        assert summary["Solver"].item() == "CONOPT4"
        assert "bmult_var" in m.data
        assert x.records.columns.to_list() == [
            "i",
            "j",
            "level",
            "marginal",
            "lower",
            "upper",
            "scale",
        ]
        assert math.isclose(z.toValue(), result, rel_tol=1e-3)
        assert math.isclose(transport.objective_value, result, rel_tol=1e-3)
    transport2 = Model(
        m,
        name="transport2",
        equations=[supply, demand, cost],
        problem="LP",
        sense=Sense.MIN,
        objective=z,
    )

    bmult_list = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

    transport2.freeze(modifiables=[bmult])

    expected_status = (
        ModelStatus.OptimalGlobal,
        ModelStatus.OptimalGlobal,
        ModelStatus.OptimalGlobal,
        ModelStatus.OptimalGlobal,
        ModelStatus.OptimalGlobal,
        ModelStatus.InfeasibleGlobal,
        ModelStatus.InfeasibleGlobal,
        ModelStatus.InfeasibleGlobal,
    )
    for b_value, status in zip(bmult_list, expected_status):
        bmult.setRecords(b_value)
        transport2.solve(solver="conopt")
        assert transport2.status == status

    transport2.unfreeze()
    # different solver
    summary = transport.solve(solver="cplex")
    assert summary["Solver"].item() == "cplex"
    assert math.isclose(
        transport.objective_value, 199.77750000000003, rel_tol=1e-6
    )

    # invalid solver
    with pytest.raises(ValidationError):
        transport.solve(solver="blablabla")

    transport.unfreeze()
    assert not transport._is_frozen


@pytest.mark.skipif(
    platform.system() == "Darwin" and platform.machine() == "x86_64",
    reason="Darwin runners are not dockerized yet.",
)
def test_variable_change(data):
    m, canning_plants, markets, capacities, demands, distances = data
    i = Set(m, name="i", records=canning_plants)
    j = Set(m, name="j", records=markets)

    a = Parameter(m, name="a", domain=[i], records=capacities)
    b = Parameter(m, name="b", domain=[j], records=demands)
    d = Parameter(m, name="d", domain=[i, j], records=distances)
    c = Parameter(m, name="c", domain=[i, j])
    bmult = Parameter(m, name="bmult", records=1)
    c[i, j] = 90 * d[i, j] / 1000

    x = Variable(m, name="x", domain=[i, j], type="Positive")
    z = Variable(m, name="z")

    cost = Equation(m, name="cost")
    supply = Equation(m, name="supply", domain=[i])
    demand = Equation(m, name="demand", domain=[j])

    cost[...] = z == Sum((i, j), c[i, j] * x[i, j])
    supply[i] = Sum(j, x[i, j]) <= a[i]
    demand[j] = Sum(i, x[i, j]) >= bmult * b[j]

    transport = Model(
        m,
        name="transport",
        equations=m.getEquations(),
        problem="LP",
        sense=Sense.MIN,
        objective=z,
    )

    transport.freeze(modifiables=[x.up])
    transport.solve(solver="conopt")
    assert math.isclose(transport.objective_value, 153.675, rel_tol=1e-6)

    x.records.loc[1, "upper"] = 0
    transport.solve(solver="conopt")
    assert math.isclose(transport.objective_value, 156.375, rel_tol=1e-6)

    transport.unfreeze()


@pytest.mark.skipif(
    platform.system() == "Darwin" and platform.machine() == "x86_64",
    reason="Darwin runners are not dockerized yet.",
)
def test_fx(data):
    m, *_ = data
    INCOME0 = Parameter(
        m,
        name="INCOME0",
        description="notional income level",
        records=3.5,
    )

    IADJ = Variable(
        m,
        name="IADJ",
        description=(
            "investment scaling factor (for fixed capital formation)"
        ),
        type="Free",
    )
    MPSADJ = Variable(
        m,
        name="MPSADJ",
        description="savings rate scaling factor",
        type="Free",
    )

    BALANCE = Equation(
        m,
        name="BALANCE",
        description="notional balance constraint",
        definition=(1 + IADJ) + (1 + MPSADJ) == INCOME0,
    )

    mm = Model(m, name="mm", equations=[BALANCE], problem="MCP")
    mm.freeze(modifiables=[INCOME0, IADJ.fx, MPSADJ.fx])
    IADJ.fx[...] = 2

    output_path = os.path.join(m.working_directory, "out.log")
    with open(output_path, "w") as file:
        mm.solve(output=file)

    assert os.path.exists(output_path)

    assert MPSADJ.toValue() == -0.5

    IADJ.fx[...] = 1
    mm.solve(output=sys.stdout)

    assert MPSADJ.toValue() == 0.5
    mm.unfreeze()


@pytest.mark.skipif(
    platform.system() == "Darwin" and platform.machine() == "x86_64",
    reason="Darwin runners are not dockerized yet.",
)
def test_validations(data):
    m, canning_plants, markets, capacities, demands, distances = data
    i = Set(m, name="i", records=canning_plants)
    j = Set(m, name="j", records=markets)

    a = Parameter(m, name="a", domain=[i], records=capacities)
    b = Parameter(m, name="b", domain=[j], records=demands)
    d = Parameter(m, name="d", domain=[i, j], records=distances)
    c = Parameter(m, name="c", domain=[i, j])
    bmult = Parameter(m, name="bmult", records=1)
    c[i, j] = 90 * d[i, j] / 1000

    x = Variable(m, name="x", domain=[i, j], type="Positive")
    z = Variable(m, name="z")

    cost = Equation(m, name="cost")
    supply = Equation(m, name="supply", domain=[i])
    demand = Equation(m, name="demand", domain=[j])

    cost[...] = z == Sum((i, j), c[i, j] * x[i, j])
    supply[i] = Sum(j, x[i, j]) <= a[i]
    demand[j] = Sum(i, x[i, j]) >= bmult * b[j]

    transport = Model(
        m,
        name="transport",
        equations=m.getEquations(),
        problem="LP",
        sense=Sense.MIN,
        objective=z,
    )

    # modifiables is not an iterable.
    with pytest.raises(ValidationError):
        transport.freeze(modifiables=bmult)

    # provide a set as a modifiable
    with pytest.raises(ValidationError):
        transport.freeze(modifiables=[i])

    transport.freeze(modifiables=[x.up], options=Options(lp="conopt"))

    # Test freeze options
    transport.solve(
        solver="conopt",
        freeze_options=FreezeOptions(debug=True),
    )
    assert math.isclose(transport.objective_value, 153.675, rel_tol=1e-6)
    assert os.path.exists("dict.txt")
    assert os.path.exists("gams.gms")

    # ModelInstanceOptions should throw a warning
    with pytest.warns(DeprecationWarning):
        transport.solve(
            solver="conopt",
            model_instance_options=ModelInstanceOptions(debug=True),
        )

    # Test solver options
    with tempfile.NamedTemporaryFile("w", delete=False) as file:
        transport.solve(
            solver="conopt", output=file, solver_options={"rtmaxv": "1.e12"}
        )
        file.close()

        with open(file.name) as f:
            assert ">>  rtmaxv 1.e12" in f.read()

        options_path = os.path.join(m.working_directory, "conopt4.opt")
        assert os.path.exists(options_path)

    with tempfile.NamedTemporaryFile("w", delete=False) as file:
        # Test a second solve call without solver options
        transport.solve(solver="conopt", output=file)
        file.close()

        with open(file.name) as f:
            assert ">>  rtmaxv 1.e12" not in f.read()


@pytest.mark.skipif(
    platform.system() == "Darwin" and platform.machine() == "x86_64",
    reason="Darwin runners are not dockerized yet.",
)
def test_modifiable_in_condition(data):
    m, *_ = data
    td_data = pd.DataFrame(
        [
            ["icbm", "2", 0.05],
            ["icbm", "6", 0.15],
            ["icbm", "7", 0.10],
            ["icbm", "8", 0.15],
            ["icbm", "9", 0.20],
            ["icbm", "18", 0.05],
            ["mrbm-1", "1", 0.16],
            ["mrbm-1", "2", 0.17],
            ["mrbm-1", "3", 0.15],
            ["mrbm-1", "4", 0.16],
            ["mrbm-1", "5", 0.15],
            ["mrbm-1", "6", 0.19],
            ["mrbm-1", "7", 0.19],
            ["mrbm-1", "8", 0.18],
            ["mrbm-1", "9", 0.20],
            ["mrbm-1", "10", 0.14],
            ["mrbm-1", "12", 0.02],
            ["mrbm-1", "14", 0.12],
            ["mrbm-1", "15", 0.13],
            ["mrbm-1", "16", 0.12],
            ["mrbm-1", "17", 0.15],
            ["mrbm-1", "18", 0.16],
            ["mrbm-1", "19", 0.15],
            ["mrbm-1", "20", 0.15],
            ["lr-bomber", "1", 0.04],
            ["lr-bomber", "2", 0.05],
            ["lr-bomber", "3", 0.04],
            ["lr-bomber", "4", 0.04],
            ["lr-bomber", "5", 0.04],
            ["lr-bomber", "6", 0.10],
            ["lr-bomber", "7", 0.08],
            ["lr-bomber", "8", 0.09],
            ["lr-bomber", "9", 0.08],
            ["lr-bomber", "10", 0.05],
            ["lr-bomber", "11", 0.01],
            ["lr-bomber", "12", 0.02],
            ["lr-bomber", "13", 0.01],
            ["lr-bomber", "14", 0.02],
            ["lr-bomber", "15", 0.03],
            ["lr-bomber", "16", 0.02],
            ["lr-bomber", "17", 0.05],
            ["lr-bomber", "18", 0.08],
            ["lr-bomber", "19", 0.07],
            ["lr-bomber", "20", 0.08],
            ["f-bomber", "10", 0.04],
            ["f-bomber", "11", 0.09],
            ["f-bomber", "12", 0.08],
            ["f-bomber", "13", 0.09],
            ["f-bomber", "14", 0.08],
            ["f-bomber", "15", 0.02],
            ["f-bomber", "16", 0.07],
            ["mrbm-2", "1", 0.08],
            ["mrbm-2", "2", 0.06],
            ["mrbm-2", "3", 0.08],
            ["mrbm-2", "4", 0.05],
            ["mrbm-2", "5", 0.05],
            ["mrbm-2", "6", 0.02],
            ["mrbm-2", "7", 0.02],
            ["mrbm-2", "10", 0.10],
            ["mrbm-2", "11", 0.05],
            ["mrbm-2", "12", 0.04],
            ["mrbm-2", "13", 0.09],
            ["mrbm-2", "14", 0.02],
            ["mrbm-2", "15", 0.01],
            ["mrbm-2", "16", 0.01],
        ]
    )

    wa_data = pd.DataFrame(
        [
            ["icbm", 200],
            ["mrbm-1", 100],
            ["lr-bomber", 300],
            ["f-bomber", 150],
            ["mrbm-2", 250],
        ]
    )

    tm_data = pd.DataFrame(
        [
            ["1", 30],
            ["6", 100],
            ["10", 40],
            ["14", 50],
            ["15", 70],
            ["16", 35],
            ["20", 10],
        ]
    )

    mv_data = pd.DataFrame(
        [
            ["1", 60],
            ["2", 50],
            ["3", 50],
            ["4", 75],
            ["5", 40],
            ["6", 60],
            ["7", 35],
            ["8", 30],
            ["9", 25],
            ["10", 150],
            ["11", 30],
            ["12", 45],
            ["13", 125],
            ["14", 200],
            ["15", 200],
            ["16", 130],
            ["17", 100],
            ["18", 100],
            ["19", 100],
            ["20", 150],
        ]
    )

    # Sets
    w = Set(
        m,
        name="w",
        records=["icbm", "mrbm-1", "lr-bomber", "f-bomber", "mrbm-2"],
        description="weapons",
    )
    t = Set(
        m,
        name="t",
        records=[str(i) for i in range(1, 21)],
        description="targets",
    )

    # Parameters
    td = Parameter(
        m,
        name="td",
        domain=[w, t],
        records=td_data,
        description="target data",
    )
    wa = Parameter(
        m,
        name="wa",
        domain=w,
        records=wa_data,
        description="weapons availability",
    )
    tm = Parameter(
        m,
        name="tm",
        domain=t,
        records=tm_data,
        description="minimum number of weapons per target",
    )
    mv = Parameter(
        m,
        name="mv",
        domain=t,
        records=mv_data,
        description="military value of target",
    )

    # Variables
    x = Variable(
        m,
        name="x",
        domain=[w, t],
        type="Positive",
        description="weapons assignment",
    )
    prob = Variable(
        m,
        name="prob",
        domain=t,
        description="probability for each target",
    )

    # Equations
    maxw = Equation(m, name="maxw", domain=w, description="weapons balance")
    minw = Equation(
        m,
        name="minw",
        domain=t,
        description="minimum number of weapons required per target",
    )
    probe = Equation(
        m,
        name="probe",
        domain=t,
        description="probability definition",
    )

    maxw[w] = Sum(t.where[td[w, t]], x[w, t]) <= wa[w]
    minw[t].where[tm[t]] = Sum(w.where[td[w, t]], x[w, t]) >= tm[t]

    probe[t] = prob[t] == 1 - Product(
        w.where[x.l[w, t]], (1 - td[w, t]) ** x[w, t]
    )

    _ = Sum(t, mv[t] * prob[t])
    etd = Sum(
        t,
        mv[t] * (1 - Product(w.where[td[w, t]], (1 - td[w, t]) ** x[w, t])),
    )

    war = Model(
        m,
        name="war",
        equations=[maxw, minw, probe],
        problem=Problem.NLP,
        sense=Sense.MAX,
        objective=etd,
    )

    with pytest.raises(ValidationError):
        war.freeze(modifiables=[td])

    with pytest.raises(ValidationError):
        war.freeze(modifiables=[tm])

    with pytest.raises(ValidationError):
        war.freeze(modifiables=[x.l])


@pytest.mark.skipif(
    platform.system() == "Darwin" and platform.machine() == "x86_64",
    reason="Darwin runners are not dockerized yet.",
)
def test_modifiable_with_domain(data):
    m, *_ = data
    import gamspy as gp

    m = gp.Container()
    i = gp.Set(m, name="i")
    j = gp.Set(m, name="j")
    a = gp.Parameter(m, name="a", domain=[i, j])
    b = gp.Parameter(m, name="b", domain=i)
    c = gp.Parameter(m, name="c", domain=j)

    x = gp.Variable(m, name="x", domain=j, type="positive")
    e = gp.Equation(m, name="e", domain=i)

    e[i] = gp.Sum(j, a[i, j] * x[j]) >= b[i]

    mymodel = gp.Model(
        m,
        name="mymodel",
        equations=[e],
        objective=gp.Sum(j, c[j] * x[j]),
        sense="min",
        problem="lp",
    )

    i.setRecords(range(10))
    j.setRecords(range(20))
    a[i, j] = gp.math.uniform(0, 1)
    b[i] = gp.math.uniform(1, 10)
    c[j] = gp.math.uniform(1, 10)

    mymodel.freeze(modifiables=[b])
    b[i] = gp.math.uniform(1, 10)
    mymodel.solve()
    assert math.isclose(
        mymodel.objective_value, 32.36124699832342, rel_tol=1e-6
    )


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Darwin runners are not dockerized yet.",
)
def test_license():
    license_path = utils._get_license_path(gamspy_base.directory)
    if "gamslice.txt" not in license_path:
        os.remove(license_path)

    m = Container()
    i = Set(m, "i", records=range(5000))
    p = Parameter(m, "p", domain=i)
    p2 = Parameter(m, "p2", records=5)
    p.generateRecords()
    v1 = Variable(m, "v1", domain=i)
    z = Variable(m, "z")
    e1 = Equation(m, "e1", domain=i)

    e1[i] = p2 * v1[i] * p[i] >= z
    model = Model(
        m, name="my_model", equations=[e1], sense=Sense.MIN, objective=z
    )
    with pytest.raises(GamspyException):
        model.freeze(modifiables=[p2])

    m.close()

    subprocess.run(
        [
            sys.executable,
            "-Bm",
            "gamspy",
            "install",
            "license",
            os.environ["MODEL_INSTANCE_LICENSE"],
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    m = Container()
    i = Set(m, "i", records=range(5000))
    p = Parameter(m, "p", domain=i)
    p2 = Parameter(m, "p2", records=5)
    p.generateRecords()
    v1 = Variable(m, "v1", domain=i)
    z = Variable(m, "z")
    e1 = Equation(m, "e1", domain=i)

    e1[i] = p2 * v1[i] * p[i] >= z
    model = Model(
        m, name="my_model", equations=[e1], sense=Sense.MIN, objective=z
    )

    model.freeze(modifiables=[p2])
    model.solve()
    assert model.solve_status == SolveStatus.NormalCompletion
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
        capture_output=True,
        text=True,
    )


def normal_dice():
    m = Container()

    f = Set(
        m,
        name="f",
        description="faces on a dice",
        records=[f"face{idx}" for idx in range(1, 100)],
    )
    dice = Set(
        m,
        name="dice",
        description="number of dice",
        records=[f"dice{idx}" for idx in range(1, 100)],
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
    eq4[dice, f - 1] = fval[dice, f - 1] + 1 <= fval[dice, f]

    xdice = Model(
        m,
        "xdice",
        equations=m.getEquations(),
        problem=Problem.MIP,
        sense=Sense.MAX,
        objective=wnx,
    )
    xdice.solve(options=Options(time_limit=0))
    flo.setRecords(2)
    xdice.solve(options=Options(time_limit=0))

    return xdice.model_generation_time


@pytest.mark.skipif(
    platform.system() == "Darwin" and platform.machine() == "x86_64",
    reason="Darwin runners are not dockerized yet.",
)
def test_timing():
    normal_generation_time = normal_dice()

    m = Container()

    f = Set(
        m,
        name="f",
        description="faces on a dice",
        records=[f"face{idx}" for idx in range(1, 100)],
    )
    dice = Set(
        m,
        name="dice",
        description="number of dice",
        records=[f"dice{idx}" for idx in range(1, 100)],
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
    eq4[dice, f - 1] = fval[dice, f - 1] + 1 <= fval[dice, f]

    xdice = Model(
        m,
        "xdice",
        equations=m.getEquations(),
        problem=Problem.MIP,
        sense=Sense.MAX,
        objective=wnx,
    )
    xdice.freeze(modifiables=[flo], options=Options(time_limit=0))
    xdice.solve(options=Options(time_limit=0))
    flo.setRecords(2)
    xdice.solve(options=Options(time_limit=0))
    frozen_model_generation = xdice.model_generation_time
    # Normal execution should take more time than frozen solve
    assert frozen_model_generation < normal_generation_time

    m.close()


@pytest.mark.skipif(
    platform.system() == "Darwin" and platform.machine() == "x86_64",
    reason="Darwin runners are not dockerized yet.",
)
def test_database():
    ws = Workspace(debugging_level="delete")
    database = Database(ws)
    set = database.add_set("i", 1)
    assert isinstance(set, GamsSet)
    parameter = database.add_parameter("a", 0)
    assert isinstance(parameter, GamsParameter)
    parameter2 = database.add_parameter("a2", 1)
    assert isinstance(parameter2, GamsParameter)
    parameter3 = database.add_parameter("a3", 2)
    assert isinstance(parameter3, GamsParameter)
    variable = database.add_variable("v", 0, 1)
    assert isinstance(variable, GamsVariable)
    equation = database.add_equation("e", 0, 1)
    assert isinstance(equation, GamsEquation)

    assert len(database) == 6

    with pytest.raises(GamspyException):
        database.add_variable("v", 0, 1)

    gdx_path = os.path.join(ws.working_directory, "dump.gdx")
    database.export(gdx_path)
    assert os.path.exists(gdx_path)

    m = Container(gdx_path)
    assert len(m) == 6
    m.close()


def test_feasibility():
    m = gp.Container()
    x = gp.Variable(m, "x", records=2)
    a = gp.Parameter(m, "a", records=6)
    b = gp.Parameter(m, "b", records=3)
    e = gp.Equation(m, "e", definition=a * x / gp.math.sqr(b) == 0)
    mi = gp.Model(m, "mi", equations=[e], problem="LP", sense="FEASIBILITY")
    assert (
        mi._generate_solve_string()
        == "solve mi using LP MIN mi_objective_variable"
    )
    mi.freeze([a, b])
    mi.solve(solver="cplex", solver_options={"writelp": "mi.lp"})
    mi.unfreeze()


def test_output_propagation(data):
    _, canning_plants, markets, capacities, demands, distances = data
    file = tempfile.NamedTemporaryFile("w", delete=False)  # noqa
    m = Container(output=file)
    i = Set(m, name="i", records=canning_plants)
    j = Set(m, name="j", records=markets)

    a = Parameter(m, name="a", domain=[i], records=capacities)
    b = Parameter(m, name="b", domain=[j], records=demands)
    d = Parameter(
        m,
        name="d",
        domain=[i, j],
        records=distances,
        is_miro_input=True,
    )
    c = Parameter(m, name="c", domain=[i, j])
    bmult = Parameter(m, name="bmult", records=1)
    c[i, j] = 90 * d[i, j] / 1000

    x = Variable(m, name="x", domain=[i, j], type="Positive")
    z = Variable(m, name="z")

    cost = Equation(m, name="cost")
    supply = Equation(m, name="supply", domain=[i])
    demand = Equation(m, name="demand", domain=[j])

    cost[...] = z == Sum((i, j), c[i, j] * x[i, j])
    supply[i] = Sum(j, x[i, j]) <= a[i]
    demand[j] = Sum(i, x[i, j]) >= bmult * b[j]

    transport = Model(
        m,
        name="transport",
        equations=m.getEquations(),
        problem="LP",
        sense=Sense.MIN,
        objective=z,
    )

    transport.freeze(modifiables=[bmult])

    with open(file.name) as f:
        assert "Generating LP model transport" in f.read()

    transport.unfreeze()
    file.close()
    os.unlink(file.name)
    m.close()


@pytest.mark.skipif(
    platform.system() != "Linux", reason="Test only for linux."
)
def test_interrupt():
    directory = str(pathlib.Path(__file__).parent.resolve())
    process = subprocess.Popen(
        [sys.executable, os.path.join(directory, "dice2.py")],
        stdout=subprocess.PIPE,
        text=True,
        stderr=subprocess.STDOUT,
    )
    time.sleep(4)
    process.send_signal(signal.SIGINT)
    process.wait()
    output = process.stdout.read()
    assert (
        "[FROZEN MODEL - WARNING] The solve was interrupted! Solve status: UserInterrupt"
        in output
    ), output
