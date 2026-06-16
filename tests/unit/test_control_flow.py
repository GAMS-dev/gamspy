import platform
import time

import numpy as np
import pytest

import gamspy as gp
from gamspy.exceptions import ValidationError

pytestmark = pytest.mark.unit


def test_loop():
    m = gp.Container()

    t = gp.Set(m, records=["1985", "1986", "1987", "1988", "1989", "1990"])
    pop = gp.Parameter(m, domain=t, records=[("1985", "3456")])
    growth = gp.Parameter(
        m, domain=t, records=np.array([25.3, 27.3, 26.2, 27.1, 26.6, 26.6])
    )

    with gp.Loop(t):
        pop[t + 1] = pop[t] + growth[t]

    assert pop.toList() == [
        ("1985", 3456.0),
        ("1986", 3481.3),
        ("1987", 3508.6000000000004),
        ("1988", 3534.8),
        ("1989", 3561.9),
        ("1990", 3588.5),
    ]

    m = gp.Container()
    i = gp.Set(m, records=[f"i{idx}" for idx in range(1, 4)])
    j = gp.Set(m, records=[f"j{idx}" for idx in range(1, 6)])
    k = gp.Set(
        m,
        domain=[i, j],
        records=[("i1", "j1"), ("i1", "j3"), ("i3", "j3"), ("i3", "j5")],
    )

    c = gp.Parameter(m, domain=i, records=[("i1", 3), ("i2", 1)])
    q = gp.Parameter(
        m, domain=[i, j], records=[("i1", "j1", 1), ("i1", "j2", 3), ("i1", "j4", 2)]
    )
    x = gp.Parameter(m, records=1)
    y = gp.Parameter(m, records=3)
    z = gp.Parameter(m, records=1)

    with gp.Loop(gp.Domain(i, j).where[q[i, j] > 0]):
        x[...] = x[...] + q[i, j]

    assert x.toValue() == 7.0

    with gp.Loop(i.where[c[i] + c[i] ** 2]):
        z[...] = z[...] + 1

    assert z.toValue() == 3.0

    with gp.Loop(i.where[gp.Sum(j, gp.math.abs(q[i, j]))]):
        z[...] = z[...] + 1

    assert z.toValue() == 4.0

    with gp.Loop(j.where[(gp.Ord(j) > 1) & (gp.Ord(j) < gp.Card(j))]):
        z[...] = z[...] + 1

    assert z.toValue() == 7.0

    with gp.Loop(gp.Domain(i, j).where[k[i, j]]):
        y[...] = y[...] + gp.Ord(i) + 2 * gp.Ord(j)

    assert y.toValue() == 35.0


def test_loop_with_solve():
    m = gp.Container()

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
    i = gp.Set(
        m,
        name="i",
        records=["seattle", "san-diego"],
        description="canning plants",
    )
    j = gp.Set(
        m,
        name="j",
        records=["new-york", "chicago", "topeka"],
        description="markets",
    )

    # Data
    a = gp.Parameter(
        m,
        name="a",
        domain=i,
        records=capacities,
        description="capacity of plant i in cases",
    )
    b = gp.Parameter(
        m,
        name="b",
        domain=j,
        records=demands,
        description="demand at market j in cases",
    )
    d = gp.Parameter(
        m,
        name="d",
        domain=[i, j],
        records=distances,
        description="distance in thousands of miles",
    )
    c = gp.Parameter(
        m,
        name="c",
        domain=[i, j],
        description="transport cost in thousands of dollars per case",
    )
    c[i, j] = 90 * d[i, j] / 1000

    # Variable
    x = gp.Variable(
        m,
        name="x",
        domain=[i, j],
        type="Positive",
        description="shipment quantities in cases",
    )

    # Equation
    supply = gp.Equation(
        m,
        name="supply",
        domain=i,
        description="observe supply limit at plant i",
    )
    demand = gp.Equation(
        m, name="demand", domain=j, description="satisfy demand at market j"
    )

    supply[i] = gp.Sum(j, x[i, j]) <= a[i]
    demand[j] = gp.Sum(i, x[i, j]) >= b[j]

    transport = gp.Model(
        m,
        name="transport",
        equations=m.getEquations(),
        problem="LP",
        sense=gp.Sense.MIN,
        objective=gp.Sum((i, j), c[i, j] * x[i, j]),
    )

    dd = gp.Parameter(m, domain=i)
    mode = gp.Parameter(m, records=1)
    cnt = gp.Parameter(m, records=0)
    k = gp.Set(m, domain=[i, j])
    kval = gp.Set(m, domain=[i, j])
    k[i, j] = True

    if mode.toValue() == 1:
        with gp.Loop(k):
            kval[k] = True
            transport.solve()
            cnt[...] = cnt[...] + 1
            c[i, j] = c[i, j] * 1.1

        dd[i] = 10

    assert kval.toList() == [
        ("seattle", "new-york"),
        ("seattle", "chicago"),
        ("seattle", "topeka"),
        ("san-diego", "new-york"),
        ("san-diego", "chicago"),
        ("san-diego", "topeka"),
    ]


def test_loop_domain_tree():
    m = gp.Container()
    i = gp.Set(m, records=[f"i{idx}" for idx in range(1, 11)])
    ii = gp.Set(m, domain=i)
    j = gp.Set(m, domain=i, records=[f"i{idx}" for idx in range(1, 10)])
    jj = gp.Set(m, domain=j, records=[f"i{idx}" for idx in range(1, 9)])
    jjj = gp.Set(m, domain=jj, records=[f"i{idx}" for idx in range(1, 8)])

    with gp.Loop(i[jjj]):
        ii[i] = True

    assert ii.toList() == ["i1", "i2", "i3", "i4", "i5", "i6", "i7"]


def test_nested_loops():
    m = gp.Container()
    i = gp.Set(m, records=range(3))
    j = gp.Set(m, records=range(4))
    a = gp.Parameter(m, domain=[i, j])
    a.generateRecords()
    b = gp.Parameter(m, records=0)

    with gp.Loop(i):
        with gp.Loop(j):
            b[...] += a[i, j]

    assert np.isclose(a.records["value"].sum(), b.toValue())


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="Windows build machines stall time to time which affects the perf tests in general.",
)
def test_loop_perf():
    m = gp.Container()

    i = gp.Set(m, records=range(20))
    j = gp.Set(m, records=range(20))
    c = gp.Parameter(m, domain=[i, j])
    c.generateRecords()

    start = time.time()
    for ival in i.toList():
        for jval in j.toList():
            c[ival, jval] = c[ival, jval] * 2

    python_loop_time = time.time() - start

    start = time.time()
    with gp.Loop((i, j)):
        c[i, j] = c[i, j] * 2

    gams_loop_time = time.time() - start

    assert python_loop_time > gams_loop_time


def test_invalid_indices():
    m = gp.Container()
    i = gp.Set(m, records=range(3))
    a = gp.Parameter(m, domain=i)
    b = gp.Parameter(m)

    with pytest.raises(ValidationError):
        with gp.Loop(5):
            b[...] = a[i]

    with pytest.raises(ValidationError):
        with gp.Loop("5"):
            b[...] = a[i]


def test_if():
    m = gp.Container()
    i = gp.Set(m, records=[f"i{idx}" for idx in range(1, 11)])
    cnt = gp.Parameter(m, records=0)

    with pytest.raises(ValidationError):
        with gp.If(gp.Ord(i) == 2):
            ...

    with gp.Loop(i) as loop:
        with gp.If(gp.Ord(i) == 2):
            loop.Continue  # noqa: B018

        with gp.If(i.sameAs("i6")):
            loop.Break  # noqa: B018

        cnt[...] += 1

    assert cnt.toValue() == 4.0

    with gp.Loop(i) as loop:
        with gp.If(gp.math.mod(gp.Ord(i), 2) == 0):
            loop.Continue  # noqa: B018

        cnt[...] += 1

    assert cnt.toValue() == 9.0


def test_nested_if():
    m = gp.Container()
    i = gp.Set(m, records=[f"i{idx}" for idx in range(1, 11)])
    j = gp.Set(m, records=[f"j{idx}" for idx in range(1, 11)])
    cnt = gp.Parameter(m, records=0)

    with gp.Loop(i):
        with gp.Loop(j) as loop:
            with gp.If(gp.Ord(i) == 2):
                with gp.If(gp.Ord(j) == 2):
                    loop.Continue  # noqa: B018

            cnt[...] += 1

    assert cnt.toValue() == 99.0


def test_break():
    m = gp.Container()
    i = gp.Set(m, records=[f"i{idx}" for idx in range(1, 11)])
    j = gp.Set(m, records=[f"j{idx}" for idx in range(1, 11)])
    cnt = gp.Parameter(m, records=0)

    with gp.Loop(i):
        with gp.Loop(j) as loop2:
            cnt[...] += 1
            loop2.Break  # noqa: B018

    assert cnt.toValue() == 10

    cnt[...] = 0

    with gp.Loop(i) as loop:
        with gp.Loop(j) as loop2:
            cnt[...] += 1
            loop2.Break  # noqa: B018

        loop.Break  # noqa: B018

    assert cnt.toValue() == 1

    with gp.Loop(i) as loop:
        with pytest.raises(ValidationError):
            with gp.Loop(j) as loop2:
                loop.Break  # noqa: B018


def test_for():
    # example from GAMS docs
    m = gp.Container()

    s = gp.Parameter(m)
    p = gp.Parameter(m)

    with gp.For(s, -3.8, -0.1, 1.4):
        p[...] = s

    assert p.toValue() == -1.0

    # example from TIMES coef_ext.cli
    m_model = gp.Container()

    t = gp.Set(m_model, name="T")
    cm_var = gp.Set(m_model, name="CM_VAR")
    cm_box = gp.Set(m_model, name="CM_BOX")
    cm_buck = gp.Set(m_model, name="CM_BUCK")
    cm_q = gp.Set(m_model, name="CM_Q")

    altobj = gp.Parameter(m_model, name="ALTOBJ")
    cm_calib = gp.Parameter(m_model, name="CM_CALIB")
    m = gp.Parameter(m_model, name="M", domain=[t])
    b = gp.Parameter(m_model, name="B", domain=[t])
    lead = gp.Parameter(m_model, name="LEAD", domain=[t])

    # Scalars for control flow
    my_f = gp.Parameter(m_model, name="MY_F")
    z = gp.Parameter(m_model, name="Z")
    f = gp.Parameter(m_model, name="F")

    # Multi-dimensional parameters (using "*" for varying alias/subset domains to avoid domain violations)
    cm_bb = gp.Parameter(m_model, name="CM_BB", domain=[cm_var, t, "*", cm_box])
    cm_cc = gp.Parameter(m_model, name="CM_CC", domain=[cm_var, t, "*", cm_box])
    cm_aa = gp.Parameter(m_model, name="CM_AA", domain=[cm_var, t, "*", "*", "*"])
    cm_phi = gp.Parameter(m_model, name="CM_PHI", domain=[cm_var, "*", "*"])

    # Control Flow implementation
    with gp.Loop(t.where[altobj != 3]):
        # IF(ORD(T) GT 1) ... ELSE ...
        with gp.If(gp.Ord(t) > 1):
            my_f[...] = m[t] - b[t] + 1
            z[...] = lead[t]

        with gp.If(gp.Ord(t) <= 1):
            my_f[...] = cm_calib
            z[...] = my_f

        # FOR(F = 1 TO Z)
        with gp.For(f, 1, z):
            # IF(F LE MY_F) ... ELSE ...
            with gp.If(f <= my_f):
                cm_bb[cm_var, t, "1", cm_box] = cm_bb[cm_var, t, "1", cm_box] + gp.Sum(
                    cm_buck,
                    cm_phi[cm_var, cm_buck, cm_var]
                    * cm_aa[cm_var, t, "1", cm_box, cm_buck],
                )

            with gp.If(f > my_f):
                cm_cc[cm_var, t, "1", cm_box] = cm_cc[cm_var, t, "1", cm_box] + gp.Sum(
                    cm_buck,
                    cm_phi[cm_var, cm_buck, cm_var]
                    * cm_aa[cm_var, t, "1", cm_box, cm_buck],
                )

            # Statement outside the inner IF/ELSE but inside the FOR loop
            cm_aa[cm_var, t, "1", cm_buck, cm_box] = gp.Sum(
                cm_q.where[cm_box[cm_q]],
                cm_aa[cm_var, t, "1", cm_buck, cm_q] * cm_phi[cm_var, cm_q, cm_box],
            )

    # Another example with expressions: https://git.gams.com/devel/gamspy/-/merge_requests/776#note_313522
    m = gp.Container()

    i = gp.Set(m, name="i", records=["1", "2"])
    j = gp.Set(m, name="j", is_singleton=True, records=["a"])

    p = gp.Parameter(
        m, name="p", domain=[i, j], records=[("1", "a", 3), ("2", "a", -2)]
    )
    k = gp.Parameter(m, name="k")
    cnt = gp.Parameter(m, name="cnt")

    with gp.Loop(i):
        cnt[...] = 0

        # IF block (p(i,j) > 0)
        with gp.If(p[i, j] > 0):
            # direction defaults to "to", but specified for clarity
            with gp.For(k, 1, 3 * p[i, "a"], p[i, j], direction="to"):
                cnt[...] = cnt + 1

        # ELSE block (translated using a mutually exclusive IF)
        with gp.If(p[i, j] <= 0):
            # Explicitly set direction="downto" and use dynamic GAMS expression for step
            with gp.For(k, -3 * p[i, "a"], 1, -p[i, j], direction="downto"):
                cnt[...] = cnt + 1

    assert cnt.toValue() == 3.0


def test_loop_with_math_op():
    m = gp.Container()

    base_R = gp.Set(m, name="base_R")
    base_V = gp.Set(m, name="base_V")
    base_P = gp.Set(m, name="base_P")
    base_BD = gp.Set(m, name="base_BD")
    base_LL = gp.Set(m, name="base_LL")
    base_YEAR = gp.Set(m, name="base_YEAR")
    base_JJ = gp.Set(m, name="base_JJ")
    base_T = gp.Set(m, name="base_T")

    AGE = gp.Set(m, name="AGE")
    J = gp.Set(m, name="J")
    EOHYEARS = gp.Set(m, name="EOHYEARS")

    R = gp.Alias(m, name="R", alias_with=base_R)
    V = gp.Alias(m, name="V", alias_with=base_V)
    P = gp.Alias(m, name="P", alias_with=base_P)
    BD = gp.Alias(m, name="BD", alias_with=base_BD)
    LL = gp.Alias(m, name="LL", alias_with=base_LL)
    YEAR = gp.Alias(m, name="YEAR", alias_with=base_YEAR)
    JJ = gp.Alias(m, name="JJ", alias_with=base_JJ)
    T = gp.Alias(m, name="T", alias_with=base_T)

    RTP_ISHPR = gp.Set(m, name="RTP_ISHPR", domain=[R, V, P])
    RTP_SHAPI = gp.Set(m, name="RTP_SHAPI", domain=[R, V, P, BD, J, JJ, LL, YEAR])
    PERIODYR = gp.Set(m, name="PERIODYR", domain=[T, EOHYEARS])

    arg2 = [R, V, P]

    YEARVAL = gp.Parameter(m, name="YEARVAL", domain=["*"])
    B = gp.Parameter(m, name="B", domain=[T])
    E = gp.Parameter(m, name="E", domain=[T])
    MULTI = gp.Parameter(m, name="MULTI", domain=[JJ, T])
    SHAPE = gp.Parameter(m, name="SHAPE", domain=[J, "*"])

    arg1 = gp.Parameter(m, name="arg1", domain=[R, V, P])
    arg4 = gp.Parameter(m, name="arg4", domain=[R, V, P])
    pass_var = gp.Parameter(m, name="pass_var", domain=[R, V, P])

    with gp.Loop(gp.math.same_as(AGE, "1")):
        arg4.where[pass_var.where[RTP_ISHPR[R, V, P]]] = arg1[arg2] * gp.Sum(
            RTP_SHAPI[R, V, P, BD, J, JJ, LL, YEAR],
            gp.SpecialValues.EPS
            + gp.Sum(
                # First argument: The domain controlling T and EOHYEARS
                PERIODYR[T, EOHYEARS].where[
                    YEARVAL[EOHYEARS] <= gp.math.Max(B[T], YEARVAL[YEAR])
                ],
                # Second argument: The entire expression evaluated within the sum
                (
                    MULTI[JJ, T]
                    * SHAPE[
                        J,
                        AGE
                        + (gp.math.Min(YEARVAL[EOHYEARS], YEARVAL[YEAR]))
                        - YEARVAL[LL],
                    ]
                )
                / (
                    gp.math.Max(
                        1,
                        gp.math.Min(E[T], YEARVAL[YEAR])
                        - gp.math.Max(B[T], YEARVAL[LL] + 1),
                    )
                ),
            ),
        )


def test_while():
    m = gp.Container()
    x = gp.Parameter(m, records=100)
    cnt = gp.Parameter(m, records=0)

    # Basic While loop test
    with gp.While(x > 1):
        x[...] = x / 2
        cnt[...] += 1

    assert (
        cnt.toValue() == 7.0
    )  # 100 -> 50 -> 25 -> 12.5 -> 6.25 -> 3.125 -> 1.5625 -> 0.78125

    # Reset parameters for break/continue test
    x[...] = 10
    cnt[...] = 0

    with gp.While(x > 0) as w:
        x[...] = x - 1

        with gp.If(x == 5):
            w.Continue  # noqa: B018

        with gp.If(x == 2):
            w.Break  # noqa: B018

        cnt[...] += 1

    # Tracing execution:
    # x=9 (cnt=1), x=8 (cnt=2), x=7 (cnt=3), x=6 (cnt=4)
    # x=5 (Continue skips cnt)
    # x=4 (cnt=5), x=3 (cnt=6)
    # x=2 (Break immediately exits the loop)
    assert cnt.toValue() == 6.0


def test_elseif_else():
    m = gp.Container()
    i = gp.Set(m, records=[f"i{idx}" for idx in range(1, 5)])
    cnt_if = gp.Parameter(m, records=0)
    cnt_elseif = gp.Parameter(m, records=0)
    cnt_else = gp.Parameter(m, records=0)

    # Test valid execution flow chaining If -> ElseIf -> Else
    with gp.Loop(i):
        with gp.If(gp.Ord(i) == 1):
            cnt_if[...] += 1
        with gp.ElseIf(gp.Ord(i) == 2):
            cnt_elseif[...] += 1
        with gp.Else():
            cnt_else[...] += 1

    assert cnt_if.toValue() == 1.0
    assert cnt_elseif.toValue() == 1.0
    assert cnt_else.toValue() == 2.0  # triggered for i3 and i4

    # Test ValidationErrors for orphaned control structures
    with pytest.raises(
        ValidationError,
        match=r"`gp.ElseIf` context manager can only be used in `gp.Loop` context managers.",
    ):
        with gp.ElseIf(gp.Ord(i) == 1):
            pass

    with pytest.raises(
        ValidationError,
        match=r"`gp.Else` context manager can only be used in `gp.Loop` context managers.",
    ):
        with gp.Else():
            pass

    # Test ValidationError when statements intervene between If and ElseIf
    with gp.Loop(i):
        with gp.If(gp.Ord(i) == 1):
            pass
        cnt_if[...] += 1
        with pytest.raises(
            ValidationError,
            match=r"`gp.ElseIf` must immediately follow a `gp.If` or `gp.ElseIf` block without any intervening statements.",
        ):
            with gp.ElseIf(gp.Ord(i) == 2):
                pass

    # Test ValidationError when statements intervene between If and Else
    with gp.Loop(i):
        with gp.If(gp.Ord(i) == 1):
            pass
        cnt_if[...] += 1
        with pytest.raises(
            ValidationError,
            match=r"`gp.Else` must immediately follow a `gp.If` or `gp.ElseIf` block without any intervening statements.",
        ):
            with gp.Else():
                pass
