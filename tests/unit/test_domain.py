from __future__ import annotations

import pytest

from gamspy import Container, Domain, Parameter, Set, Sum
from gamspy.exceptions import GamspyException, ValidationError

pytestmark = pytest.mark.unit


@pytest.fixture
def data():
    m = Container()
    yield m
    m.close()


def test_domain(data):
    m = data
    i = Set(m, name="i", records=["seattle", "san-diego"])
    j = Set(m, name="j", records=["new-york", "chicago", "topeka"])

    domain = Domain(i, j)
    assert domain.gamsRepr() == "(i,j)"

    # Domain with less than two sets
    pytest.raises(ValidationError, Domain, i)

    # Domain with no set or alias symbols
    pytest.raises(ValidationError, Domain, "i", "j")


def test_domain_forwarding(data):
    m = data
    i = Set(m, name="i")
    _ = Parameter(
        m,
        name="p",
        domain=[i],
        domain_forwarding=True,
        records=[["i1", 1]],
    )
    assert i.toList() == ["i1"]

    k = Set(m, name="k")
    j = Set(m, name="j")
    _ = Parameter(
        m,
        name="p2",
        domain=[k, j],
        domain_forwarding=[True, True],
        records=[["k1", "j1", 1]],
    )
    assert k.toList() == ["k1"]
    assert j.toList() == ["j1"]

    k2 = Set(m, name="k2")
    j2 = Set(m, name="j2")
    _ = Set(
        m,
        name="p3",
        domain=[k2, j2],
        domain_forwarding=[True, True],
        records=[("k2", "j2")],
    )
    assert k2.toList() == ["k2"]
    assert j2.toList() == ["j2"]

    i2 = Set(m, "i2", description="plant locations")

    _ = Parameter(
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


def test_domain_validation(data):
    m = data
    times = Set(m, "times", records=["release", "duration"])
    job = Set(m, "job", records=["job1", "job2"])
    data = Parameter(m, "data", domain=[times, job])

    M = m.addParameter("M")
    M[...] = Sum(job, data["release", job] + data["duration", job])
    with pytest.raises(ValidationError):
        M[...] = Sum(job, data["rbla", job] + data["bla", job])

    job2 = Set(m, "job2", records=["job1", "job2"])
    data2 = Parameter(m, "data2", domain=["times", "job"])

    M2 = m.addParameter("M2")
    M2[...] = Sum(job2, data2["release", job2] + data2["duration", job2])

    with pytest.raises(GamspyException):
        M[...] = Sum(job2, data2["rbla", job2] + data2["bla", job2])
