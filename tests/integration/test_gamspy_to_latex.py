import os

import model_builders
import pytest

import gamspy as gp
from gamspy import (
    Container,
    Equation,
    Model,
    Ord,
    Parameter,
    Sense,
    Set,
    Sum,
    Variable,
)
from gamspy.exceptions import ValidationError

pytestmark = pytest.mark.integration


def test_lp_transport(container, tmp_path):
    m = container
    transport = model_builders.lp_transport(m)
    i, j, c, x = m.getSymbols(["i", "j", "c", "x"])

    output_path = str(tmp_path / "to_latex")
    transport.toLatex(output_path, generate_pdf=False)

    output_path2 = tmp_path / "folder with blank"
    transport.toLatex(str(output_path2), generate_pdf=False)
    with open(os.path.join(output_path, "transport.tex")) as file1:
        content1 = file1.read()

    with open(output_path2 / "transport.tex") as file2:
        content2 = file2.read()

    assert content1 == content2

    # assert(
    #     os.path.exists(os.path.join(output_path, "transport.pdf"))
    # )
    reference_path = os.path.join(
        "tests", "integration", "tex_references", "transport.tex"
    )
    with open(reference_path, encoding="utf-8") as file:
        reference_tex = file.read()

    with open(os.path.join(output_path, "transport.tex"), encoding="utf-8") as file:
        generated_tex = file.read()

    assert reference_tex == generated_tex

    freeLinks = Set(m, "freeLinks", domain=[i, j], records=[("seattle", "chicago")])
    transport2 = Model(
        m,
        name="transport2",
        equations=m.getEquations(),
        problem="LP",
        sense=Sense.MIN,
        objective=Sum((i, j), c[i, j] * x[i, j]),
        limited_variables=[x[freeLinks]],
    )
    transport2.toLatex(output_path, generate_pdf=False)

    reference_path = os.path.join(
        "tests", "integration", "tex_references", "transport2.tex"
    )
    with open(reference_path, encoding="utf-8") as file:
        reference_tex = file.read()

    with open(os.path.join(output_path, "transport2.tex"), encoding="utf-8") as file:
        generated_tex = file.read()

    assert reference_tex == generated_tex


def test_mip_cutstock(container, tmp_path):
    m = container
    master = model_builders.mip_cutstock(m)

    output_path = str(tmp_path / "to_latex")
    master.toLatex(output_path)
    reference_path = os.path.join(
        "tests", "integration", "tex_references", "master.tex"
    )
    with open(reference_path) as file:
        reference_tex = file.read()

    with open(os.path.join(output_path, "master.tex")) as file:
        generated_tex = file.read()

    assert reference_tex == generated_tex


def test_nlp_weapons(container, tmp_path):
    m = container
    war = model_builders.nlp_weapons(m)

    output_path = str(tmp_path / "to_latex")
    war.toLatex(output_path)
    reference_path = os.path.join("tests", "integration", "tex_references", "war.tex")
    with open(reference_path) as file:
        reference_tex = file.read()

    with open(os.path.join(output_path, "war.tex")) as file:
        generated_tex = file.read()

    assert reference_tex == generated_tex


def test_mcp_qp6(tmp_path):
    qp6 = model_builders.mcp_qp6()

    output_path = str(tmp_path / "to_latex")
    qp6.toLatex(output_path)
    reference_path = os.path.join("tests", "integration", "tex_references", "qp6.tex")
    with open(reference_path) as file:
        reference_tex = file.read()

    with open(os.path.join(output_path, "qp6.tex")) as file:
        generated_tex = file.read()

    assert reference_tex == generated_tex


def test_dnlp_inscribedsquare(container, tmp_path):
    m = container
    square = model_builders.dnlp_inscribedsquare(m)

    output_path = str(tmp_path / "to_latex")
    square.toLatex(output_path)
    reference_path = os.path.join(
        "tests", "integration", "tex_references", "square.tex"
    )
    with open(reference_path) as file:
        reference_tex = file.read()

    with open(os.path.join(output_path, "square.tex")) as file:
        generated_tex = file.read()

    assert reference_tex == generated_tex


def test_minlp_minlphix(container, tmp_path):
    m = container
    skip = model_builders.minlp_minlphix(m)

    output_path = str(tmp_path / "to_latex")
    skip.toLatex(output_path)
    reference_path = os.path.join("tests", "integration", "tex_references", "skip.tex")
    with open(reference_path) as file:
        reference_tex = file.read()

    with open(os.path.join(output_path, "skip.tex")) as file:
        generated_tex = file.read()

    assert reference_tex == generated_tex


def test_qcp_EDsensitivity(container, tmp_path):
    m = container
    ECD = model_builders.qcp_EDsensitivity(m)

    output_path = str(tmp_path / "to_latex")
    ECD.toLatex(output_path)
    reference_path = os.path.join("tests", "integration", "tex_references", "ECD.tex")
    with open(reference_path) as file:
        reference_tex = file.read()

    with open(os.path.join(output_path, "ECD.tex")) as file:
        generated_tex = file.read()

    assert reference_tex == generated_tex


def test_renaming(container, tmp_path):
    m = container
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

    greek = Model(
        m,
        name="greek",
        equations=m.getEquations(),
        problem="LP",
        sense=Sense.MIN,
        objective=Sum((i, j), c[i, j] * x[i, j]),
    )
    output_path = str(tmp_path / "to_latex")
    greek.toLatex(output_path, rename={"x": "ρ"})  # noqa: RUF001

    reference_path = os.path.join("tests", "integration", "tex_references", "greek.tex")
    with open(reference_path, encoding="utf-8") as file:
        reference_tex = file.read()

    with open(os.path.join(output_path, "greek.tex"), encoding="utf-8") as file:
        generated_tex = file.read()

    assert reference_tex == generated_tex


def test_latex_repr(container):
    # Other tests are check the equivalency of the generated tex file and the reference tex file.
    # Here we test the latex representation of individual components.
    m = container

    i = Set(m, "i", records=["i1", "i2"])
    assert i.latexRepr() == "i"
    j = Set(m, "j", domain=i, records=["i1"])
    assert j.latexRepr() == "j"
    assert j[i].latexRepr() == r"j_{i}"
    assert j["i1"].latexRepr() == r"j_{\text{`i1'}}"

    a = Parameter(m, "a")
    assert a.latexRepr() == "a"
    b = Parameter(m, "b", domain=[i, j])
    assert b.latexRepr() == "b"
    assert b[i, j].latexRepr() == r"b_{i,j}"
    assert b[i, "i1"].latexRepr() == r"b_{i,\text{`i1'}}"

    c = Variable(m, "c")
    assert c.latexRepr() == "c"
    d = Variable(m, "d", domain=[i, j])
    assert d.latexRepr() == "d"
    assert d[i, j].latexRepr() == r"d_{i,j}"
    assert d[i, "i1"].latexRepr() == r"d_{i,\text{`i1'}}"

    e = Equation(m, "e")

    # Equations must be defined to get its latex representation
    with pytest.raises(ValidationError):
        e.latexRepr()

    e[...] = c * c - a >= 0
    assert e.latexRepr() == "$\nc \\cdot c - a \\geq 0\n$"

    assert gp.Number(5).latexRepr() == "5"


def test_symbol_name_with_underscore():
    m = Container()

    cities = Set(m, name="cities", records=["LA", "HOU", "NY", "MIA"])

    distance_for_next_city = Parameter(m, name="distance_for_next_city", domain=cities)
    distance_for_next_city.setRecords(
        [("LA", 1500), ("HOU", 1700), ("NY", 1300), ("MIA", 2700)]
    )

    fuel_purchased = Variable(
        m, name="fuel_purchased", domain=cities, type="positive"
    )  # Fuel purchased in city
    fuel_at_takeoff = Variable(
        m, name="fuel_at_takeoff", domain=cities, type="positive"
    )  # Fuel at takeoff city
    fuel_at_landing = Variable(
        m, name="fuel_at_landing", domain=cities, type="positive"
    )  # Fuel at landing city

    fuel_balance_ground = Equation(m, name="fuel_balance_ground", domain=cities)
    fuel_balance_ground[cities] = (
        fuel_at_landing[cities] + fuel_purchased[cities] == fuel_at_takeoff[cities]
    )

    fuel_balance_air = Equation(m, name="fuel_balance_air", domain=cities)
    fuel_balance_air[cities].where[Ord(cities) > 1] = (
        fuel_at_landing[cities]
        == fuel_at_takeoff[cities - 1]
        - (1 + ((0.5 * (fuel_at_takeoff[cities - 1] + fuel_at_landing[cities])) / 2000))
        * distance_for_next_city[cities - 1]
    )

    assert (
        fuel_balance_air.latexRepr()
        == r"""$
fuel\_at\_landing_{cities} = fuel\_at\_takeoff_{cities - 1} - (1 + \frac{0.5 \cdot (fuel\_at\_takeoff_{cities - 1} + fuel\_at\_landing_{cities})}{2000}) \cdot distance\_for\_next\_city_{cities - 1}\hfill \forall cities ~ | ~ ord(cities) > 1
$"""
    )

    m = gp.Container()
    A = gp.Set(m, name="a_underscore")
    S = gp.Set(m, name="s_underscore")
    AS = gp.Set(m, domain=[A, S])
    p1 = gp.Parameter(m, domain=[A, S])
    p2 = gp.Parameter(
        m,
    )
    p2[...] = gp.Sum(AS[A, S], p1[A, S])
    assert (
        p2._assignment.latexRepr()
        == r"p2 = \sum_{AS_{a\_underscore,s\_underscore}} p1_{a\_underscore,s\_underscore}"
    )

    v = gp.Variable(m)
    e = gp.Equation(m, domain=[A, S])
    e[...].where[AS[A, S]] = v == 5
    assert (
        e._definition.latexRepr()
        == r"e_{a\_underscore,s\_underscore} ~ | ~ AS_{a\_underscore,s\_underscore} .. v = 5"
    )

    p3 = gp.Parameter(m, name="p3_underscore")
    p4 = gp.Parameter(m, name="p4_underscore")
    assert (
        gp.math.regularized_beta(p3, p4, 2).latexRepr()
        == r"betaReg(p3\_underscore,p4\_underscore,2)"
    )
    assert gp.math.abs(p3).latexRepr() == r"\lvert{p3\_underscore}"
