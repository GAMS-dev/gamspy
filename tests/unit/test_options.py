from __future__ import annotations

import os
import shutil
import sys
import time

import pytest
from pydantic import ValidationError

import gamspy.exceptions as exceptions
import gamspy.math as math
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
from gamspy.exceptions import GamspyException

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
    savepoint_path = os.path.join(os.getcwd(), "transport_p.gdx")
    if os.path.exists(savepoint_path):
        os.remove(savepoint_path)


def test_options(data):
    with pytest.raises(exceptions.ValidationError):
        options = Options(generate_name_dict=True)
        _ = Container(options=options)

    with pytest.raises(exceptions.ValidationError):
        options = Options(loadpoint="bla.gdx")
        _ = Container(options=options)

    with pytest.raises(ValidationError):
        _ = Options(unknown_option=5)

    with pytest.raises(ValidationError):
        _ = Options(hold_fixed_variables=5)

    options = Options(hold_fixed_variables=True)
    assert options.hold_fixed_variables == 1

    with pytest.raises(ValidationError):
        _ = Options(report_solution=5)

    options = Options(report_solution=1)
    assert options.report_solution == 1

    with pytest.raises(ValidationError):
        _ = Options(merge_strategy=5)

    options = Options(merge_strategy="replace")
    assert options.merge_strategy == "replace"

    with pytest.raises(ValidationError):
        _ = Options(step_summary=5)

    options = Options(step_summary=True)
    assert options.step_summary is True

    with pytest.raises(ValidationError):
        _ = Options(suppress_compiler_listing=5)

    options = Options(suppress_compiler_listing=True)
    assert options.suppress_compiler_listing is True

    with pytest.raises(ValidationError):
        _ = Options(report_solver_status=5)

    options = Options(report_solver_status=True)
    assert options.report_solver_status is True

    with pytest.raises(ValidationError):
        _ = Options(report_underflow=5)

    options = Options(report_underflow=True)
    assert options.report_underflow is True

    options = Options(solve_link_type="disk")
    assert options._get_gams_compatible_options()["solvelink"] == 2

    options_path = os.path.join("tmp", "options.pf")
    options.export(options_path)
    with open(options_path) as file:
        content = file.read()

    assert (
        content
        == 'limcol = "0"\nlimrow = "0"\nsolprint = "0"\nsolvelink = "2"\npreviouswork = "1"\ntraceopt = "3"\nlogoption = "0"'
    )


def test_seed(data):
    m, *_ = data
    m = Container(
        options=Options(seed=1),
    )
    p1 = Parameter(m, "p1")
    p1[...] = math.normal(0, 1)
    assert p1.records.value.item() == 0.45286287828275534

    p2 = Parameter(m, "p2")
    p2[...] = math.normal(0, 1)
    assert p2.records.value.item() == -0.4841775276628964

    # change seed
    m = Container(
        options=Options(seed=5),
    )
    p1 = Parameter(m, "p1")
    p1[...] = math.normal(0, 1)
    assert p1.records.value.item() == 0.14657004110784333

    p2 = Parameter(m, "p2")
    p2[...] = math.normal(0, 1)
    assert p2.records.value.item() == 0.11165956511164217


def test_global_options(data):
    m, canning_plants, markets, capacities, demands, distances = data
    options = Options(lp="conopt")
    m = Container(
        debugging_level="keep",
        options=options,
    )

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

    with open(
        os.path.join(m.working_directory, m.gamsJobName() + ".pf")
    ) as file:
        assert 'lp = "conopt"' in file.read()


def test_gamspy_to_gams_options(data):
    m, canning_plants, markets, capacities, _, distances = data
    options = Options(
        allow_suffix_in_equation=False,
        allow_suffix_in_limited_variables=False,
        merge_strategy="replace",
    )
    gams_options = options._get_gams_compatible_options(output=None)
    assert gams_options["suffixalgebravars"] == "off"
    assert gams_options["suffixdlvars"] == "off"
    assert gams_options["solveopt"] == 0


def test_log_option(data):
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

    transport.solve()  # logoption=0
    transport.solve(output=sys.stdout)  # logoption = 3

    logfile_name = os.path.join(os.getcwd(), "tmp", "log.txt")
    transport.solve(
        output=sys.stdout,
        options=Options(log_file=logfile_name),
    )  # logoption = 4

    # test logfile
    transport.solve(options=Options(log_file=logfile_name))  # logoption = 2
    assert os.path.exists(logfile_name)

    # test listing file
    listing_file_name = os.path.join(os.getcwd(), "tmp", "listing.lst")
    transport.solve(options=Options(listing_file=listing_file_name))
    assert os.path.exists(listing_file_name)

    listing_file_name = os.path.join("tmp", "listing2.lst")
    transport.solve(options=Options(listing_file=listing_file_name))
    assert os.path.exists(listing_file_name)


def test_from_file(data):
    option_file = os.path.join("tmp", "option_file")
    with open(option_file, "w") as file:
        file.write("lp = conopt\n\n")

    options = Options.fromFile(option_file)
    assert options.lp == "conopt"

    with pytest.raises(exceptions.ValidationError):
        _ = Options.fromFile("unknown_path")


def test_profile(data):
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

    profile_path = os.path.join("tmp", "bla.profile")
    transport.solve(
        output=sys.stdout,
        options=Options(
            profile=1,
            profile_file=profile_path,
            monitor_process_tree_memory=True,
        ),
    )
    assert os.path.exists(profile_path)

    # solprint should be 0 by default
    with open(m.gamsJobName() + ".lst") as file:
        assert "---- EQU supply" not in file.read()


def test_solprint(data):
    m, canning_plants, markets, capacities, demands, distances = data
    m = Container(options=Options(report_solution=1))

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

    # solprint is 1
    with open(m.gamsJobName() + ".lst") as file:
        assert "---- EQU supply" in file.read()


def test_exception_on_solve_with_listing_file(data):
    m, *_ = data
    x = Variable(m, name="x")

    transport = Model(
        m,
        name="transport",
        problem="LP",
        sense=Sense.MIN,
        objective=x / 0,
    )

    with pytest.raises(GamspyException):
        transport.solve(
            options=Options(
                listing_file=os.path.join("tmp", "bla.lst"),
            ),
        )


def test_model_attribute_options(data):
    m, canning_plants, markets, capacities, demands, distances = data
    m = Container(debugging_level="keep")

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
    transport.solve(options=Options(infeasibility_tolerance=1e-6))
    transport.solve(options=Options(infeasibility_tolerance=1e-6))
    assert "transport.tolInfRep = 1e-06;" in m.generateGamsString(
        show_raw=True
    )


def test_scaling(data):
    m, *_ = data
    m = Container()

    x1 = Variable(m, "x1", type="positive")
    x2 = Variable(m, "x2", type="positive")

    z = Variable(m, "z")
    eq = Equation(m, "eq")
    eq[...] = 200 * x1 + 0.5 * x2 == z

    x1.up[...] = 0.01
    x2.up[...] = 10
    x1.scale[...] = 0.01
    x2.scale[...] = 10

    eq.scale = 1e-6

    model = Model(m, "my_model", equations=[eq], sense="MIN", objective=z)
    listing_file_path = os.path.join("tmp", "scaling.lst")
    model.solve(
        options=Options(
            enable_scaling=True,
            equation_listing_limit=100,
            listing_file=listing_file_path,
        )
    )
    assert eq.records.scale.item() == 1e-6
    with open(listing_file_path) as file:
        assert "eq..  2000000*x1 + 5000000*x2 - 1000000*z =E= 0" in file.read()


def test_loadpoint(data):
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

    transport.solve(options=Options(savepoint=1))
    assert transport.num_iterations == 4

    transport.solve(options=Options(loadpoint="transport_p.gdx"))
    assert transport.num_iterations == 0


def test_solver_options_twice(data):
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

    log_file_path = os.path.join("tmp", "log.log")
    transport.solve(
        solver="cplex",
        solver_options={"lpmethod": 4},
        options=Options(log_file=log_file_path),
    )
    with open(log_file_path) as file:
        assert "OptFile 1" in file.read()

    transport.solve(options=Options(log_file=log_file_path))
    with open(log_file_path) as file:
        assert "OptFile 0" in file.read()


def test_debug_options():
    m = Container()
    save_path = os.path.join(m.working_directory, "save.g00")
    m._options._set_debug_options({"save": save_path})
    _ = Set(m, records=["i1", "i2"])
    assert os.path.exists(save_path)
    os.remove(save_path)
    m.close()


def test_solver_options_highs(data):
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

    log_path = os.path.join("tmp", "log.log")
    with open(log_path, "w") as file:
        transport.solve(
            output=file, solver="highs", solver_options={"random_seed": 999}
        )

    with open(log_path) as file:
        assert "random_seed" in file.read()


def test_bypass_solver():
    m = Container()

    f = Set(
        m,
        name="f",
        description="faces on a dice",
        records=[f"face{idx}" for idx in range(1, 20)],
    )
    dice = Set(
        m,
        name="dice",
        description="number of dice",
        records=[f"dice{idx}" for idx in range(1, 20)],
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

    start = time.time()
    xdice.solve(options=Options(bypass_solver=True))
    end = time.time()

    # It should not take more than 5 seconds since we don't pass it to the solver
    # If we pass it to the solver, it should take hours maybe days.
    assert end - start < 5

    m.close()
