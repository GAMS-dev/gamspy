import math
import os
import subprocess

import model_builders
import pytest

import gamspy.exceptions as exceptions
from gamspy import (
    Equation,
    Model,
    Options,
    Parameter,
    Problem,
    Sense,
    Set,
    Sum,
    Variable,
)

pytestmark = pytest.mark.integration


def test_lp_transport(container, tmp_path):
    m = container
    transport = model_builders.lp_transport(m)

    to_gams_path = str(tmp_path / "to_gams")
    trace_path = os.path.join(to_gams_path, "trace.txt")
    transport.toGams(
        to_gams_path,
        options=Options.fromGams({"lp": "cplex"}),
    )
    with pytest.raises(exceptions.ValidationError):
        transport.toGams(to_gams_path, options={"lp": "cplex"})

    transport.toGams(
        to_gams_path,
        options=Options(generate_name_dict=False, lp="CPLEX"),
    )

    process = subprocess.run(
        [
            os.path.join(m.system_directory, "gams"),
            os.path.join(to_gams_path, "transport.gms"),
            "traceopt=2",
            f"trace={trace_path}",
            f"output={os.path.join(to_gams_path, 'transport.lst')}",
        ],
        capture_output=True,
        text=True,
    )

    assert process.returncode == 0

    with open(trace_path) as trace:
        lines = trace.read().splitlines()
        objective = lines[-1].split("//")[0].split(" ")[-3]
        assert objective.startswith("153.675")

    reference_path = os.path.join(
        "tests", "integration", "gms_references", "transport.gms"
    )
    with open(os.path.join(to_gams_path, "transport.gms")) as file1:
        content1 = [
            line
            for line in file1.read().splitlines()
            if not line.startswith("$gdxLoad")
        ]
    with open(reference_path) as file2:
        content2 = [
            line
            for line in file2.read().splitlines()
            if not line.startswith("$gdxLoad")
        ]

    assert content1 == content2

    folder_name = "folder with spaces"
    folder_path = str(tmp_path / folder_name)
    transport.toGams(folder_path)

    process = subprocess.run(
        [
            os.path.join(m.system_directory, "gams"),
            os.path.join(folder_path, "transport.gms"),
            "traceopt=2",
            f"trace={trace_path}",
            f"output={os.path.join(str(tmp_path), 'transport.lst')}",
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr


def test_mip_cutstock(container, tmp_path):
    m = container
    master = model_builders.mip_cutstock(m)

    to_gams_path = str(tmp_path / "to_gams")
    trace_path = os.path.join(to_gams_path, "trace.txt")
    master.toGams(to_gams_path)

    process = subprocess.run(
        [
            os.path.join(m.system_directory, "gams"),
            os.path.join(to_gams_path, "master.gms"),
            "traceopt=2",
            f"trace={trace_path}",
            f"output={os.path.join(to_gams_path, 'master.lst')}",
        ],
        capture_output=True,
        text=True,
    )

    assert process.returncode == 0
    with open(trace_path) as trace:
        lines = trace.read().splitlines()
        objective = lines[-1].split("//")[0].split(" ")[-3]
        assert objective.startswith("452.25")

    reference_path = os.path.join(
        "tests", "integration", "gms_references", "master.gms"
    )
    with open(os.path.join(to_gams_path, "master.gms")) as file1:
        content1 = [
            line
            for line in file1.read().splitlines()
            if not line.startswith("$gdxLoad")
        ]
    with open(reference_path) as file2:
        content2 = [
            line
            for line in file2.read().splitlines()
            if not line.startswith("$gdxLoad")
        ]

    assert content1 == content2


def test_nlp_weapons(container, tmp_path):
    m = container
    war = model_builders.nlp_weapons(m)

    to_gams_path = str(tmp_path / "to_gams")
    trace_path = os.path.join(to_gams_path, "trace.txt")
    war.toGams(to_gams_path)

    process = subprocess.run(
        [
            os.path.join(m.system_directory, "gams"),
            os.path.join(to_gams_path, "war.gms"),
            "traceopt=2",
            f"trace={trace_path}",
            f"output={os.path.join(to_gams_path, 'war.lst')}",
        ],
        capture_output=True,
        text=True,
    )

    assert process.returncode == 0
    with open(trace_path) as trace:
        lines = trace.read().splitlines()
        objective = lines[-1].split("//")[0].split(" ")[-3]
        assert objective.startswith("1735.569579")

    reference_path = os.path.join("tests", "integration", "gms_references", "war.gms")
    with open(os.path.join(to_gams_path, "war.gms")) as file1:
        content1 = [
            line
            for line in file1.read().splitlines()
            if not line.startswith("$gdxLoad")
        ]
    with open(reference_path) as file2:
        content2 = [
            line
            for line in file2.read().splitlines()
            if not line.startswith("$gdxLoad")
        ]

    assert content1 == content2


def test_mcp_qp6(container, tmp_path):
    m = container
    qp6 = model_builders.mcp_qp6()

    to_gams_path = str(tmp_path / "to_gams")
    trace_path = os.path.join(to_gams_path, "trace.txt")
    qp6.toGams(to_gams_path)

    process = subprocess.run(
        [
            os.path.join(m.system_directory, "gams"),
            os.path.join(to_gams_path, "qp6.gms"),
            "traceopt=2",
            f"trace={trace_path}",
            f"output={os.path.join(to_gams_path, 'qp6.lst')}",
        ],
        capture_output=True,
        text=True,
    )

    assert process.returncode == 0

    with open(trace_path) as trace:
        lines = trace.read().splitlines()
        objective = lines[-1].split("//")[0].split(" ")[-3]
        assert objective.startswith("8.499300")

    reference_path = os.path.join("tests", "integration", "gms_references", "qp6.gms")
    with open(os.path.join(to_gams_path, "qp6.gms")) as file1:
        content1 = [
            line
            for line in file1.read().splitlines()
            if not line.startswith("$gdxLoad")
        ]
    with open(reference_path) as file2:
        content2 = [
            line
            for line in file2.read().splitlines()
            if not line.startswith("$gdxLoad")
        ]

    assert content1 == content2


def test_dnlp_inscribedsquare(container, tmp_path):
    m = container
    square = model_builders.dnlp_inscribedsquare(m)

    to_gams_path = str(tmp_path / "to_gams")
    trace_path = os.path.join(to_gams_path, "trace.txt")
    square.toGams(to_gams_path)

    process = subprocess.run(
        [
            os.path.join(m.system_directory, "gams"),
            os.path.join(to_gams_path, "square.gms"),
            "traceopt=2",
            f"trace={trace_path}",
            f"output={os.path.join(to_gams_path, 'square.lst')}",
        ],
        capture_output=True,
        text=True,
    )

    assert process.returncode == 0

    with open(trace_path) as trace:
        lines = trace.read().splitlines()
        objective = float(lines[-1].split("//")[0].split(" ")[-3])
        assert math.isclose(objective, 1.60087540678543, rel_tol=0.01)

    reference_path = os.path.join(
        "tests", "integration", "gms_references", "square.gms"
    )
    with open(os.path.join(to_gams_path, "square.gms")) as file1:
        content1 = [
            line
            for line in file1.read().splitlines()
            if not line.startswith("$gdxLoad")
        ]
    with open(reference_path) as file2:
        content2 = [
            line
            for line in file2.read().splitlines()
            if not line.startswith("$gdxLoad")
        ]

    assert content1 == content2


def test_minlp_minlphix(container, tmp_path):
    m = container
    skip = model_builders.minlp_minlphix(m)

    to_gams_path = str(tmp_path / "to_gams")
    skip.toGams(to_gams_path)
    trace_path = os.path.join(to_gams_path, "trace.txt")

    process = subprocess.run(
        [
            os.path.join(m.system_directory, "gams"),
            os.path.join(to_gams_path, "skip.gms"),
            "traceopt=2",
            f"trace={trace_path}",
            "domlim=100",
            f"output={os.path.join(to_gams_path, 'skip.lst')}",
        ],
        capture_output=True,
        text=True,
    )

    assert process.returncode == 0

    with open(trace_path) as trace:
        lines = trace.read().splitlines()
        objective = float(lines[-1].split("//")[0].split(" ")[-3])
        assert math.isclose(objective, 316.692694176485, rel_tol=1e-4)

    reference_path = os.path.join("tests", "integration", "gms_references", "skip.gms")
    with open(os.path.join(to_gams_path, "skip.gms")) as file1:
        content1 = [
            line
            for line in file1.read().splitlines()
            if not line.startswith("$gdxLoad")
        ]
    with open(reference_path) as file2:
        content2 = [
            line
            for line in file2.read().splitlines()
            if not line.startswith("$gdxLoad")
        ]

    assert content1 == content2


def test_qcp_EDsensitivity(container, tmp_path):
    m = container
    ECD = model_builders.qcp_EDsensitivity(m)

    to_gams_path = str(tmp_path / "to_gams")
    trace_path = os.path.join(to_gams_path, "trace.txt")
    ECD.toGams(to_gams_path)

    process = subprocess.run(
        [
            os.path.join(m.system_directory, "gams"),
            os.path.join(to_gams_path, "ECD.gms"),
            "traceopt=2",
            f"trace={trace_path}",
            f"output={os.path.join(to_gams_path, 'ECD.lst')}",
        ],
        capture_output=True,
        text=True,
    )

    assert process.returncode == 0

    with open(trace_path) as trace:
        lines = trace.read().splitlines()
        objective = lines[-1].split("//")[0].split(" ")[-3]
        print(f"{objective=}")
        assert objective.startswith("911044")

    reference_path = os.path.join("tests", "integration", "gms_references", "ECD.gms")
    with open(os.path.join(to_gams_path, "ECD.gms")) as file1:
        content1 = [
            line
            for line in file1.read().splitlines()
            if not line.startswith("$gdxLoad")
        ]
    with open(reference_path) as file2:
        content2 = [
            line
            for line in file2.read().splitlines()
            if not line.startswith("$gdxLoad")
        ]

    assert content1 == content2


def test_set_attributes(container, tmp_path):
    m = container
    s = m.addSet("s", records=[1, 2])
    x = m.addVariable("x", type="positive", domain=s)
    eq = m.addEquation("eq", domain=s)
    eq[s].where[~s.first] = x[s] >= 1

    model = m.addModel(
        problem="LP",
        name="attr",
        sense=Sense.MIN,
        equations=[eq],
        objective=Sum(s, x[s]),
    )

    to_gams_path = str(tmp_path / "to_gams")
    trace_path = os.path.join(to_gams_path, "trace.txt")
    model.toGams(to_gams_path)

    process = subprocess.run(
        [
            os.path.join(m.system_directory, "gams"),
            os.path.join(to_gams_path, "attr.gms"),
            "traceopt=2",
            f"trace={trace_path}",
            f"output={os.path.join(to_gams_path, 'attr.lst')}",
        ],
        capture_output=True,
        text=True,
    )

    assert process.returncode == 0

    with open(trace_path) as trace:
        lines = trace.read().splitlines()
        objective = lines[-1].split("//")[0].split(" ")[-3]
        assert float(objective) == 1

    reference_path = os.path.join("tests", "integration", "gms_references", "attr.gms")
    with open(os.path.join(to_gams_path, "attr.gms")) as file1:
        content1 = [
            line
            for line in file1.read().splitlines()
            if not line.startswith("$gdxLoad")
        ]
    with open(reference_path) as file2:
        content2 = [
            line
            for line in file2.read().splitlines()
            if not line.startswith("$gdxLoad")
        ]

    assert content1 == content2


def test_math_op(container, tmp_path):
    ct = container

    S = ct.addSet("S", records=["a"])
    p = ct.addParameter("p", domain=S)
    p[S] = 2
    x = ct.addVariable("x", domain=S, type="positive")
    e = ct.addEquation("e", domain=S)

    e[S] = (p[S] * x[S]) ** 2 <= 4

    m = ct.addModel(
        "math",
        problem=Problem.QCP,
        equations=ct.getEquations(),
        sense=Sense.MIN,
        objective=Sum(S, x[S]),
    )

    tmp_path_folder = str(tmp_path / "to_gams")
    m.toGams(tmp_path_folder)

    reference_path = os.path.join("tests", "integration", "gms_references", "math.gms")
    with open(os.path.join(tmp_path_folder, "math.gms")) as file1:
        content1 = [
            line
            for line in file1.read().splitlines()
            if not line.startswith("$gdxLoad")
        ]
    with open(reference_path) as file2:
        content2 = [
            line
            for line in file2.read().splitlines()
            if not line.startswith("$gdxLoad")
        ]

    assert content1 == content2


def test_dump_gams_state(container, tmp_path):
    m = container

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
        description="canning plants",
    )
    j = Set(
        m,
        name="j",
        records=["new-york", "chicago", "topeka"],
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
    path = str(tmp_path / "transport")
    transport.toGams(path, dump_gams_state=True)
    assert os.path.exists(os.path.join(path, "transport.g00"))


def test_toGams_with_alias_as_domain(tmp_path):
    import gamspy as gp

    m = gp.Container()

    i = gp.Set(m, "i", records=range(10))
    ii = gp.Alias(m, "ii", alias_with=i)
    j = gp.Set(m, "j", domain=[ii])

    j[i].where[gp.Ord(i) <= 5] = True

    X = gp.Variable(m, "X", domain=[ii])

    eq_obj = gp.Equation(m, "eq_obj", domain=[ii])
    eq_obj[j] = X[j] <= 5

    test = gp.Model(m, name="test", problem="LP", equations=[eq_obj])

    folder = str(tmp_path)
    test.toGams(folder)
    with open(os.path.join(folder, "test.gms")) as file:
        content = file.read()
        assert "Set i(*);" in content
