from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

import gamspy.exceptions as exceptions
import gamspy.math as math
from gamspy import (
    Alias,
    Card,
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
    Variable,
    VariableType,
)
from gamspy.exceptions import GamspyException


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
    savepoint_path = os.path.join(os.getcwd(), "transport_p.gdx")
    if os.path.exists(savepoint_path):
        os.remove(savepoint_path)


@pytest.mark.unit
def test_options(data, tmp_path):
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

    options_path = str(tmp_path / "options.pf")
    options.export(options_path)
    with open(options_path) as file:
        content = file.read()

    assert (
        content
        == 'optfile = "0"\nlimcol = "0"\nlimrow = "0"\nsolprint = "0"\nsolvelink = "2"\npreviouswork = "1"\nlogoption = "0"'
    )


@pytest.mark.unit
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


@pytest.mark.unit
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

    with open(os.path.join(m.working_directory, m.gamsJobName() + ".pf")) as file:
        assert 'lp = "conopt"' in file.read()


@pytest.mark.unit
def test_gamspy_to_gams_options(data):
    _m, _canning_plants, _markets, _capacities, _, _distances = data
    options = Options(
        allow_suffix_in_equation=False,
        allow_suffix_in_limited_variables=False,
        merge_strategy="replace",
    )
    gams_options = options._get_gams_compatible_options(output=None)
    assert gams_options["suffixalgebravars"] == "off"
    assert gams_options["suffixdlvars"] == "off"
    # merge_strategy is now emitted as a model attribute (solveopt), not as a
    # global GAMS option, so that it does not stick across solves.
    assert "solveopt" not in gams_options


@pytest.mark.unit
def test_format_model_attr_value():
    from gamspy._options import _format_model_attr_value

    # merge_strategy keywords are converted to their solveopt integer values.
    assert _format_model_attr_value("merge_strategy", "replace") == "0"
    assert _format_model_attr_value("merge_strategy", "merge") == "1"
    assert _format_model_attr_value("merge_strategy", "clear") == "2"

    # Booleans are converted to integers.
    assert _format_model_attr_value("hold_fixed_variables", True) == "1"
    assert _format_model_attr_value("hold_fixed_variables", False) == "0"

    # Numeric values are passed through.
    assert _format_model_attr_value("iteration_limit", 5) == "5"
    assert _format_model_attr_value("time_limit", 10.0) == "10.0"


@pytest.mark.unit
def test_model_attr_options_excluded_from_pf():
    from gamspy._options import MODEL_ATTR_OPTION_MAP

    options = Options(
        iteration_limit=5,
        time_limit=10,
        node_limit=3,
        merge_strategy="replace",
        threads=2,
        seed=7,
    )
    gams_options = options._get_gams_compatible_options(output=None)

    # Options that map to GAMS model attributes must not leak into the pf file;
    # they are emitted as model attributes at solve time instead.
    for gamspy_name in (
        "iteration_limit",
        "time_limit",
        "node_limit",
        "merge_strategy",
        "threads",
    ):
        assert gamspy_name in MODEL_ATTR_OPTION_MAP
        assert MODEL_ATTR_OPTION_MAP[gamspy_name] not in gams_options

    assert "iterlim" not in gams_options
    assert "reslim" not in gams_options
    assert "solveopt" not in gams_options

    # Non-model-attribute options are still written to the pf file, and
    # solve_link_type stays a global option.
    assert gams_options["seed"] == 7
    assert "solvelink" in gams_options


@pytest.mark.unit
def test_container_creation_option_restrictions():
    for opt in (
        Options(cutoff=5),
        Options(enable_scaling=True),
        Options(generate_name_dict=True),
    ):
        with pytest.raises(exceptions.ValidationError):
            _ = Container(options=opt)

    # Options that are set as model attributes are allowed as
    # Container-level defaults.
    for opt in (
        Options(iteration_limit=100),
        Options(time_limit=60),
        Options(merge_strategy="replace"),
    ):
        m = Container(options=opt)
        m.close()


@pytest.mark.unit
def test_model_attr_options_reset_across_solves(data):
    m, *_ = data
    x = Variable(m, "x")
    e = Equation(m, "e")
    e[...] = x >= 1
    model = Model(m, "mdl", equations=[e], problem="LP", sense=Sense.MIN, objective=x)

    # When set, the option is emitted as a model attribute.
    m._unsaved_statements = []
    model._add_runtime_options(
        Options(iteration_limit=2, time_limit=10, merge_strategy="clear")
    )
    statements = "".join(s for s in m._unsaved_statements if isinstance(s, str))
    assert "mdl.iterlim = 2;" in statements
    assert "mdl.reslim = 10.0;" in statements
    assert "mdl.solveopt = 2;" in statements

    # When not set, the attribute is reset to NA so a previously set value does
    # not stick to subsequent solves.
    m._unsaved_statements = []
    model._add_runtime_options(Options())
    statements = "".join(s for s in m._unsaved_statements if isinstance(s, str))
    assert "mdl.iterlim = NA;" in statements
    assert "mdl.reslim = NA;" in statements
    assert "mdl.solveopt = NA;" in statements


@pytest.mark.unit
def test_iteration_limit_not_sticky():
    m = Container()

    # SETS #
    t = Set(
        m,
        name="t",
        records=[f"t{t}" for t in range(1, 51)],
        description="time periods",
    )
    tfirst = Set(m, name="tfirst", domain=t, description="first interval (t0)")
    tlast = Set(m, name="tlast", domain=t, description="last intervat [T]")
    tnotlast = Set(m, name="tnotlast", domain=t, description="all intervals but last")

    tfirst[t].where[Ord(t) == 1] = True
    tlast[t].where[Ord(t) == Card(t)] = True
    tnotlast[t] = ~tlast[t]

    # SCALARS #
    rho = Parameter(m, name="rho", records=0.04, description="discount factor")
    g = Parameter(m, name="g", records=0.03, description="labor growth rate")
    delta = Parameter(
        m,
        name="delta",
        records=0.02,
        description="capital depreciation factor",
    )
    K0 = Parameter(m, name="K0", records=3.00, description="initial capital")
    I0 = Parameter(m, name="I0", records=0.07, description="initial investment")
    C0 = Parameter(m, name="C0", records=0.95, description="initial consumption")
    L0 = Parameter(m, name="L0", records=1.00, description="initial labor")
    b = Parameter(m, name="b", records=0.25, description="Cobb Douglas coefficient")
    a = Parameter(m, name="a", description="Cobb Douglas coefficient")

    # PARAMETERS #
    L = Parameter(m, name="L", domain=t, description="labor (production input)")
    beta = Parameter(
        m,
        name="beta",
        domain=t,
        description="weight factor for future utilities",
    )
    tval = Parameter(m, name="tval", domain=t, description="numerical value of t")

    tval[t] = Ord(t) - 1

    # The terminal weight beta(tlast) computation.
    beta[tnotlast[t]] = math.power(1 + rho, -tval[t])
    beta[tlast[t]] = (1 / rho) * math.power(1 + rho, 1 - tval[t])
    # display beta

    # Labor is determined using an exponential growth process.
    L[t] = math.power(1 + g, tval[t]) * L0

    # Cobb-Douglas coefficient a computation.
    a = (C0 + I0) / (K0**b * L0 ** (1 - b))

    # VARIABLES #
    C = Variable(m, name="C", domain=t, description="consumption")
    Y = Variable(m, name="Y", domain=t, description="production")
    K = Variable(m, name="K", domain=t, description="capital")
    I = Variable(m, name="I", domain=t, description="investment")

    # EQUATIONS #
    production = Equation(
        m,
        name="production",
        type="regular",
        domain=t,
        description="Cobb-Douglas production function",
    )
    allocation = Equation(
        m,
        name="allocation",
        type="regular",
        domain=t,
        description="household choose between consumption and saving",
    )
    accumulation = Equation(
        m,
        name="accumulation",
        type="regular",
        domain=t,
        description="capital accumulation",
    )
    final = Equation(
        m,
        name="final",
        type="regular",
        domain=t,
        description="minimal investment in final period",
    )

    # Objective function; total utility
    utility = Sum(t, beta[t] * math.log(C[t]))
    production[t] = Y[t] == a * (K[t] ** b) * (L[t] ** (1 - b))
    allocation[t] = Y[t] == C[t] + I[t]
    accumulation[tnotlast[t]] = K[t + 1] == (1 - delta) * K[t] + I[t]
    final[tlast] = I[tlast] >= (g + delta) * K[tlast]

    # Bounds.
    K.lo[t] = 0.001
    C.lo[t] = 0.001

    # Initial conditions
    K.fx[tfirst] = K0
    I.fx[tfirst] = I0
    C.fx[tfirst] = C0

    ramsey = Model(
        m,
        name="ramsey",
        equations=m.getEquations(),
        problem="nlp",
        sense="MAX",
        objective=utility,
    )

    ramsey.solve(solver="minos", options=Options(iteration_limit=2))
    assert ramsey.num_iterations == 2

    ramsey.solve(solver="minos")
    assert ramsey.num_iterations != 2


@pytest.mark.unit
def test_log_option(data, tmp_path):
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

    logfile_name = str(tmp_path / "log.txt")
    transport.solve(
        output=sys.stdout,
        options=Options(log_file=logfile_name),
    )  # logoption = 4

    # test logfile
    transport.solve(options=Options(log_file=logfile_name))  # logoption = 2
    assert os.path.exists(logfile_name)

    # test listing file
    listing_file_name = str(tmp_path / "listing.lst")
    transport.solve(options=Options(listing_file=listing_file_name))
    assert os.path.exists(listing_file_name)

    listing_file_name = str(tmp_path / "listing2.lst")
    transport.solve(options=Options(listing_file=listing_file_name))
    assert os.path.exists(listing_file_name)

    logfile_name = str(tmp_path / "log2.txt")
    transport.solve(options=Options(log_file=logfile_name, append_to_log_file=True))
    transport.solve(options=Options(log_file=logfile_name, append_to_log_file=True))
    with open(logfile_name) as log_file:
        content = log_file.read()
        matches = re.findall("Status: Normal completion", content)
        assert len(matches) == 2


@pytest.mark.unit
def test_from_file(data, tmp_path):
    option_file = str(tmp_path / "option_file")
    with open(option_file, "w") as file:
        file.write("lp = conopt\n\n")

    options = Options.fromFile(option_file)
    assert options.lp == "conopt"

    with pytest.raises(exceptions.ValidationError):
        _ = Options.fromFile("unknown_path")


@pytest.mark.unit
def test_profile(data, tmp_path):
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

    profile_path = str(tmp_path / "bla.profile")
    transport.solve(
        output=sys.stdout,
        options=Options(
            profile=1,
            profile_file=profile_path,
        ),
    )
    assert os.path.exists(profile_path)

    # solprint should be 0 by default
    with open(m.gamsJobName() + ".lst") as file:
        assert "---- EQU supply" not in file.read()

    with pytest.raises(exceptions.ValidationError):
        transport.solve(
            output=sys.stdout, options=Options(monitor_process_tree_memory=True)
        )


@pytest.mark.unit
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


@pytest.mark.unit
def test_exception_on_solve_with_listing_file(data, tmp_path):
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
                listing_file=str(tmp_path / "bla.lst"),
            ),
        )


@pytest.mark.unit
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
    assert "transport.tolInfRep = 1e-06;" in m.generateGamsString(show_raw=True)


@pytest.mark.unit
def test_scaling(data, tmp_path):
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
    listing_file_path = str(tmp_path / "scaling.lst")
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


@pytest.mark.unit
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

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as temp_file:
        transport.solve(
            output=temp_file,
            options=Options(loadpoint=Path("transport_p.gdx")),
        )
        temp_file.close()
        with open(temp_file.name) as file:
            content = file.read()
            assert "GDX File (execute_load)" in content

        os.remove(temp_file.name)


@pytest.mark.unit
def test_solver_options_twice(data, tmp_path):
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

    log_file_path = str(tmp_path / "log.log")
    transport.solve(
        solver="cplex",
        solver_options={"lpmethod": 4},
        options=Options(log_file=log_file_path, solve_link_type="memory"),
    )
    with open(log_file_path) as file:
        content = file.read()
        assert "Solvelink=5" in content
        assert "OptFile 1" in content

    transport.solve(options=Options(log_file=log_file_path))
    with open(log_file_path) as file:
        content = file.read()
        assert "Solvelink=2" in content
        assert "OptFile 0" in content

    transport2 = Model(
        m,
        name="transport2",
        equations=m.getEquations(),
        problem="LP",
        sense=Sense.MIN,
        objective=Sum((i, j), 2 * c[i, j] * x[i, j]),
    )
    transport2.solve(options=Options(log_file=log_file_path))
    with open(log_file_path) as file:
        content = file.read()
        assert "Solvelink=2" in content


@pytest.mark.unit
def test_extra_options():
    m = Container()
    save_path = os.path.join(m.working_directory, "save.g00")
    m._options._set_extra_options({"save": save_path})
    _ = Set(m, records=["i1", "i2"])
    assert os.path.exists(save_path)
    os.remove(save_path)
    m.close()


@pytest.mark.unit
def test_solver_options_highs(data, tmp_path):
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

    log_path = str(tmp_path / "log.log")
    with open(log_path, "w") as file:
        transport.solve(
            output=file, solver="highs", solver_options={"random_seed": 999}
        )

    with open(log_path) as file:
        assert "random_seed" in file.read()


@pytest.mark.requires_license
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


def test_options_from_gams():
    options = Options.fromGams(
        {"reslim": 5, "lp": "cplex", "solvelink": 5, "solveopt": "replace"}
    )
    assert options.time_limit == 5
    assert options.lp == "cplex"
    assert options.solve_link_type == "memory"

    with pytest.raises(exceptions.ValidationError):
        _ = Options.fromGams({"solvelink": 4})

    with pytest.raises(exceptions.ValidationError):
        _ = Options.fromGams({"bla": 4})


def test_monitor_process_tree_memory(tmp_path):
    gamspy_script_path = tmp_path / "test.py"
    with open(gamspy_script_path, "w") as file:
        file.write(
            "import gamspy as gp; m = gp.Container(options=gp.Options(monitor_process_tree_memory=True))"
        )

    process = subprocess.run(
        [sys.executable, gamspy_script_path], capture_output=True, text=True
    )
    assert process.returncode == 0, process.stderr

    assert "Process-tree memory monitor is finished" in process.stdout
