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
        "palmasola",
        "potosi",
        "baranquill",
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

    job2 = gp.Set(m, "job2", records=["job1", "job2"])
    data2 = gp.Parameter(m, "data2", domain=["times", "job"])

    M2 = m.addParameter("M2")
    M2[...] = gp.Sum(job2, data2["release", job2] + data2["duration", job2])

    with pytest.raises(GamspyException):
        M[...] = gp.Sum(job2, data2["rbla", job2] + data2["bla", job2])


def test_domain_validation_conditional_expression():
    m = gp.Container()
    ALLYEAR = gp.Set(m, "ALLYEAR")
    ALLSOW = gp.Set(m, "ALLSOW", records=["1"])
    YEAR = gp.Alias(m, "YEAR", alias_with=ALLYEAR)
    ALL_REG = gp.Set(m, "ALL_REG")
    COM_GRP = gp.Set(m, "COM_GRP")
    SPE = gp.Set(m, "SPE")
    COM = gp.Set(m, "COM", domain=[COM_GRP])
    OPR = gp.Set(m, "Opr", domain=COM)
    BLE = gp.Set(m, "BLE", domain=COM)
    MILESTONYR = gp.Set(m, "MILESTONYR", domain=ALLYEAR)
    T = gp.Alias(m, "T", alias_with=MILESTONYR)

    REG = gp.Set(m, "REG", domain=ALL_REG)
    R = gp.Alias(m, "R", alias_with=REG)
    BLE_TP = gp.Set(m, "BLE_TP", domain=[R, ALLYEAR, "*"])
    VAR_BLND = gp.Variable(m, "VAR_BLND", domain=[R, ALLYEAR, COM, COM])
    BleOpr = gp.Set(m, "BLE_OPR", domain=[R, COM, COM])
    BL_COM = gp.Parameter(m, "BL_COM", domain=[R, COM, OPR, SPE])
    RU_CVT = gp.Parameter(m, "RU_CVT", domain=[R, BLE, SPE, OPR])
    BL_SPEC = gp.Parameter(m, "BL_SPEC", domain=[R, COM, SPE])
    EQL_BLND = gp.Equation(m, "EQL_BLND", domain=[R, YEAR, BLE, SPE, ALLSOW])
    SUBT = gp.Set(m, "SUBT", domain=[T])
    BL_TYPE = gp.Parameter(m, "BL_TYPE", domain=[R, COM, SPE])

    EQL_BLND[BLE_TP[R, SUBT[T], BLE], SPE, "1"].where[
        gp.Number(1) & (BL_TYPE[R, BLE, SPE] == 1)
    ] = (
        gp.Sum(
            OPR.where[BleOpr[R, BLE, OPR]],
            (BL_COM[R, BLE, OPR, SPE] - BL_SPEC[R, BLE, SPE])
            * RU_CVT[R, BLE, SPE, OPR]
            * VAR_BLND[R, T, BLE, OPR],
        )
        == 0
    )


def test_domain_validation_with_implicit_set():
    m = gp.Container()
    R = gp.Set(m, "R")
    LL = gp.Set(m, "LL")
    P = gp.Set(m, "P")
    L = gp.Set(m, "L", records=["bla"])
    Rvp = gp.Set(m, "Rvp", domain=[R, LL, P])
    ncap_bnd = gp.Parameter(m, "ncap_bnd", domain=[R, LL, P, L])
    ncap_bnd[Rvp[R, LL, P], L["bla"]]


def test_domain_validation_multidim_set_index():
    m = gp.Container()
    i = gp.Set(m, "i", records=["i1", "i2"])
    j = gp.Set(m, "j", records=["j1", "j2"])
    k = gp.Set(m, "k", records=["k1", "k2"])
    p = gp.Parameter(m, "p", domain=[i, j], records=[("i1", "j1", 1)])

    # A matching tuple set is a valid index.
    ij = gp.Set(m, "ij", domain=[i, j], records=[("i1", "j1")])
    s = gp.Parameter(m, "s")
    s[...] = gp.Sum(ij, p[ij])

    # A tuple set unrelated in any component is a domain violation.
    kl = gp.Set(m, "kl", domain=[k, k], records=[("k1", "k2")])
    with pytest.raises(ValidationError):
        s[...] = gp.Sum(kl, p[kl])

    # A tuple set wrong in a single component is a domain violation.
    ik = gp.Set(m, "ik", domain=[i, k], records=[("i1", "k1")])
    with pytest.raises(ValidationError):
        s[...] = gp.Sum(ik, p[ik])


def test_domain_validation_implicit_set_index():
    m = gp.Container()
    i = gp.Set(m, "i", records=["i1", "i2"])
    k = gp.Set(m, "k", records=["k1", "k2"])
    p = gp.Parameter(m, "p", domain=[i], records=[("i1", 1)])

    # Re-indexing a subset of an unrelated set is a domain violation.
    sub_k = gp.Set(m, "sub_k", domain=[k], records=["k1"])
    r = gp.Parameter(m, "r", domain=[k])
    with pytest.raises(ValidationError):
        r[sub_k] = p[sub_k[k]]


def test_domain_validation_implicit_set_narrower_than_parent_domain():
    m = gp.Container()
    ALLYEAR = gp.Set(m, "ALLYEAR")
    PRC = gp.Set(m, "PRC")
    REG = gp.Set(m, "REG")
    P = gp.Alias(m, "P", alias_with=PRC)
    R = gp.Alias(m, "R", alias_with=REG)
    RTPCPTYR = gp.Set(m, "RTPCPTYR", domain=[R, ALLYEAR, ALLYEAR, P])
    MODLYEAR = gp.Set(m, "MODLYEAR", domain=ALLYEAR)
    V = gp.Alias(m, "V", alias_with=MODLYEAR)
    MILESTONYR = gp.Set(m, "MILESTONYR", domain=ALLYEAR)
    T = gp.Alias(m, "T", alias_with=MILESTONYR)
    COEF_AF = gp.Parameter(m, "COEF_AF", domain=[R, ALLYEAR, T, PRC])

    # [R, V, T, P] line up with COEF_AF's domain.
    COEF_AF[RTPCPTYR[R, V, T, P]]

    # V is a subset of ALLYEAR but not of the T.
    with pytest.raises(ValidationError):
        COEF_AF[RTPCPTYR[R, V, V, P]]


def test_domain_validation_one_dimensional_implicit_set():
    m = gp.Container()
    UC_NAME = gp.Set(m, "UC_NAME")
    TSLVL = gp.Set(m, "TSLVL")
    TSL = gp.Alias(m, "TSL", alias_with=TSLVL)
    UC_ATTR = gp.Set(m, "UC_ATTR", domain=UC_NAME)

    # Must not raise: the position is filled by UC_NAME, matching the declaration.
    UC_ATTR[UC_NAME[TSL]]

    # A parent that is not compatible with the declared domain is still rejected.
    other = gp.Set(m, "other")
    sub_other = gp.Set(m, "sub_other", domain=other)
    p = gp.Parameter(m, "p", domain=UC_NAME)
    with pytest.raises(ValidationError):
        p[sub_other[other]]


def test_domain_validation_universe_offset():
    # A universe ("*") domain position must not shift validation of the
    # following positions.
    m = gp.Container()
    i = gp.Set(m, "i", records=["i1", "i2"])
    j = gp.Set(m, "j", records=["j1", "j2"])
    k = gp.Set(m, "k", records=["k1", "k2"])
    p = gp.Parameter(m, "p", domain=["*", j])
    r = gp.Parameter(m, "r", domain=[i])

    # Second position must be j-compatible; k is unrelated -> violation.
    with pytest.raises(ValidationError):
        r[i] = gp.Sum(k, p[i, k])

    # Same shape but with a valid second index is accepted.
    r[i] = gp.Sum(j, p[i, j])


def test_uncontrolled_set_in_operation():
    m = gp.Container()
    i = gp.Set(m, "i", records=["i1", "i2"])
    j = gp.Set(m, "j", records=["j1", "j2"])
    p = gp.Parameter(m, "p", domain=[i, j], records=[("i1", "j1", 1)])
    r = gp.Parameter(m, "r")

    # Uncontrolled j directly under a Sum / Product.
    with pytest.raises(ValidationError):
        r[...] = gp.Sum(i, p[i, j])

    with pytest.raises(ValidationError):
        r[...] = gp.Product(i, p[i, j])

    # Uncontrolled j nested inside a math function under a Sum.
    with pytest.raises(ValidationError):
        r[...] = gp.Sum(i, gp.math.sqr(p[i, j]))

    # Uncontrolled j only in the left operand of the Sum.
    q = gp.Parameter(m, "q", domain=[i], records=[("i1", 1)])
    with pytest.raises(ValidationError):
        r[...] = gp.Sum(i, p[i, j] + q[i])

    # Uncontrolled j only inside a where condition of the Sum.
    with pytest.raises(ValidationError):
        r[...] = gp.Sum(i, q[i].where[p[i, j] > 0])


def test_uncontrolled_set_in_math_assignment():
    m = gp.Container()
    i = gp.Set(m, "i", records=["i1", "i2"])
    j = gp.Set(m, "j", records=["j1", "j2"])
    a = gp.Parameter(m, "a", domain=[i, j], records=[("i1", "j1", 1)])
    b = gp.Parameter(m, "b", domain=[i])

    # Uncontrolled j inside a math function on the right-hand side.
    with pytest.raises(ValidationError):
        b[i] = gp.math.sqr(a[i, j])


def test_conditioned_sum_iterator_is_controlled():
    m = gp.Container()
    t = gp.Set(m, "t", records=[f"t{n}" for n in range(1, 6)])
    g = gp.Set(m, "g", records=["g1"])
    tt = gp.Set(m, "tt", domain=[t], records=["t1", "t2"])
    t1 = gp.Alias(m, "t1", alias_with=t)
    pMinDown = gp.Parameter(m, "pMinDown", domain=[g, t])
    vStart = gp.Variable(m, "vStart", domain=[g, t])
    eStartFast = gp.Equation(m, "eStartFast", domain=[g, t])

    # Must not raise.
    eStartFast[g, t1] = (
        gp.Sum(
            tt[t].where[gp.Ord(t) <= pMinDown[g, t1]],
            vStart[g, t.lead(gp.Ord(t1) - pMinDown[g, t1])],
        )
        <= 1
    )
