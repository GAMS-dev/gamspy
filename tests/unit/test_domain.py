from __future__ import annotations

import pandas as pd
import pytest

import gamspy as gp
from gamspy.exceptions import GamspyException, ValidationError

pytestmark = pytest.mark.unit


def test_domain():
    m = gp.Container()
    i = gp.Set(m, name="i", records=["seattle", "san-diego"])
    j = gp.Set(m, name="j", records=["new-york", "chicago", "topeka"])

    domain = gp.Domain(i, j)
    assert domain.gamsRepr() == "(i,j)"

    # Domain with less than two sets
    with pytest.raises(ValidationError):
        gp.Domain(i)

    # Domain with no set or alias symbols
    with pytest.raises(ValidationError):
        gp.Domain("i", "j")
    m.close()


def test_domain_forwarding():
    m = gp.Container()
    i = gp.Set(m, name="i")
    _ = gp.Parameter(
        m,
        name="p",
        domain=[i],
        domain_forwarding=True,
        records=[["i1", 1]],
    )
    assert i.toList() == ["i1"]

    k = gp.Set(m, name="k")
    j = gp.Set(m, name="j")
    _ = gp.Parameter(
        m,
        name="p2",
        domain=[k, j],
        domain_forwarding=[True, True],
        records=[["k1", "j1", 1]],
    )
    assert k.toList() == ["k1"]
    assert j.toList() == ["j1"]

    k2 = gp.Set(m, name="k2")
    j2 = gp.Set(m, name="j2")
    _ = gp.Set(
        m,
        name="p3",
        domain=[k2, j2],
        domain_forwarding=[True, True],
        records=[("k2", "j2")],
    )
    assert k2.toList() == ["k2"]
    assert j2.toList() == ["j2"]

    i2 = gp.Set(m, "i2", description="plant locations")

    _ = gp.Parameter(
        m,
        "tran",
        description="transport cost for interplant shipments (us$ per ton)",
        domain=[i2, i2],
        domain_forwarding=True,
        records=[
            ("pto-suarez", "palmasola", 87.22),
            ("potosi", "palmasola", 31.25),
            ("potosi", "pto-suarez", 55.97),
            ("baranquill", "palmasola", 89.80),
            ("baranquill", "pto-suarez", 114.56),
            ("baranquill", "potosi", 70.68),
            ("cartagena", "palmasola", 89.80),
            ("cartagena", "pto-suarez", 114.56),
            ("cartagena", "potosi", 70.68),
            ("cartagena", "baranquill", 5.00),
        ],
    )

    assert i2.toList() == [
        "pto-suarez",
        "potosi",
        "baranquill",
        "cartagena",
        "palmasola",
    ]


def test_partial_domain_forwarding():
    m = gp.Container()

    # Test partial domain forwarding of sets
    i = gp.Set(m, "i")
    j = gp.Set(m, "j", records=["san-diego"])
    _ = gp.Set(
        m,
        "ij",
        domain=[i, j],
        domain_forwarding=[True, False],
        records=[("seattle", "san-diego")],
    )
    assert i.toList() == ["seattle"]

    # Test partial domain forwarding of parameters
    i2 = gp.Set(m, "i2")
    j2 = gp.Set(m, "j2", records=["san-diego"])
    _ = gp.Parameter(
        m,
        "ij2",
        domain=[i2, j2],
        domain_forwarding=[True, False],
        records=[("seattle", "san-diego", 5)],
    )
    assert i2.toList() == ["seattle"]

    # Test partial domain forwarding of variables
    i3 = gp.Set(m, "i3")
    j3 = gp.Set(m, "j3", records=["san-diego"])
    _ = gp.Variable(
        m,
        "ij3",
        domain=[i3, j3],
        domain_forwarding=[True, False],
        records=pd.DataFrame([("seattle", "san-diego")]),
    )
    assert i3.toList() == ["seattle"]

    # Test partial domain forwarding of equations
    i4 = gp.Set(m, "i4")
    j4 = gp.Set(m, "j4", records=["san-diego"])
    _ = gp.Equation(
        m,
        "ij4",
        domain=[i4, j4],
        domain_forwarding=[True, False],
        records=pd.DataFrame([("seattle", "san-diego")]),
    )
    assert i4.toList() == ["seattle"]


def test_domain_validation():
    m = gp.Container()
    times = gp.Set(m, "times", records=["release", "duration"])
    job = gp.Set(m, "job", records=["job1", "job2"])
    data = gp.Parameter(m, "data", domain=[times, job])

    M = m.addParameter("M")
    M[...] = gp.Sum(job, data["release", job] + data["duration", job])
    with pytest.raises(ValidationError):
        M[...] = gp.Sum(job, data["rbla", job] + data["bla", job])

    job2 = gp.Set(m, "job2", records=["job1", "job2"])
    data2 = gp.Parameter(m, "data2", domain=["times", "job"])

    M2 = m.addParameter("M2")
    M2[...] = gp.Sum(job2, data2["release", job2] + data2["duration", job2])

    with pytest.raises(GamspyException):
        M[...] = gp.Sum(job2, data2["rbla", job2] + data2["bla", job2])
