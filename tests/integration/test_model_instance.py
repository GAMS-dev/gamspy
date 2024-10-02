from __future__ import annotations

import glob
import math
import os

import pandas as pd
import pytest

from gamspy import (
    Container,
    Equation,
    Model,
    ModelInstanceOptions,
    Options,
    Parameter,
    Problem,
    Product,
    Sense,
    Set,
    Sum,
    Variable,
)
from gamspy.exceptions import ValidationError

pytestmark = pytest.mark.integration


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
        os.remove(file)

    if os.path.exists("dict.txt"):
        os.remove("dict.txt")

    if os.path.exists("gams.gms"):
        os.remove("gams.gms")


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
        transport.solve(solver="conopt")
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

    # different solver
    transport.solve(solver="cplex")
    assert math.isclose(
        transport.objective_value, 199.77750000000003, rel_tol=1e-6
    )

    # invalid solver
    with pytest.raises(ValidationError):
        transport.solve(solver="blablabla")

    transport.unfreeze()
    assert not transport._is_frozen


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
    IADJ.setRecords({"lower": 0, "upper": 0, "scale": 1})
    mm.solve()

    assert MPSADJ.records["level"].tolist()[0] == 1.5

    MPSADJ.setRecords({"lower": 0, "upper": 0, "scale": 1})
    mm.solve()

    assert MPSADJ.records["level"].tolist()[0] == 0
    mm.unfreeze()


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

    # Test model instance options
    transport.solve(
        solver="conopt",
        model_instance_options=ModelInstanceOptions(debug=True),
    )
    assert math.isclose(transport.objective_value, 153.675, rel_tol=1e-6)
    assert os.path.exists("dict.txt")
    assert os.path.exists("gams.gms")

    # Test solver options
    transport.solve(solver="conopt", solver_options={"rtmaxv": "1.e12"})
    assert os.path.exists(os.path.join(m.working_directory, "conopt.opt"))


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
