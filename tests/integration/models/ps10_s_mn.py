"""
## GAMSSOURCE: https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_ps10_s_mn.html
## LICENSETYPE: Demo
## MODELTYPE: NLP
## KEYWORDS: nonlinear programming, contract theory, principal-agent problem, adverse selection, parts supply problem


Parts Supply Problem w/ 10 Types w/ Random p(i) (PS10_S_MN)

Hideo Hashimoto, Kojun Hamada, and Nobuhiro Hosoe, "A Numerical Approach
to the Contract Theory: the Case of Adverse Selection", GRIPS Discussion
Paper 11-27, National Graduate Institute for Policy Studies, Tokyo, Japan,
March 2012.
"""

from __future__ import annotations

import time

from gamspy import (
    Alias,
    Card,
    Container,
    Equation,
    Model,
    Number,
    Options,
    Ord,
    Parameter,
    Problem,
    Sense,
    Set,
    Sum,
    Variable,
)
from gamspy.math import Round, uniform


def main():
    start = time.time()

    # Decreased no. of draw from 1001 to 11 for convenience
    # Otherwise, it takes a lot of time
    NUM_DRAWS = 11

    m = Container()

    # Sets
    i = Set(
        m,
        name="i",
        records=[str(i) for i in range(10)],
        description="type of supplier",
    )

    t = Set(
        m,
        name="t",
        records=[str(i) for i in range(1, NUM_DRAWS)],
        description="no. of Monte-Carlo draws",
    )

    j = Alias(m, name="j", alias_with=i)

    # Parameters
    theta = Parameter(m, name="theta", domain=i, description="efficiency")
    pt = Parameter(
        m, name="pt", domain=[i, t], description="probability of type"
    )
    p = Parameter(m, name="p", domain=i, description="probability of type")

    theta[i] = Ord(i) / Card(i)

    # Generating probability
    for tt, _ in t.records.itertuples(index=False):
        pt[i, tt] = uniform(0, 1)
    pt[i, t] = pt[i, t] / Sum(j, pt[j, t])

    # Parameters
    F = Parameter(
        m,
        name="F",
        domain=[i, t],
        description="cumulative probability (Itho p. 42)",
    )
    noMHRC0 = Parameter(
        m,
        name="noMHRC0",
        domain=[i, t],
        description="no MHRC combination between i and i-1",
    )
    # (MHRC: monotone hazard rate condition)
    noMHRC = Parameter(
        m, name="noMHRC", domain=t, description=">=1: no MHRC case"
    )

    F[i, t] = Sum(j.where[Ord(j) <= Ord(i)], pt[j, t])
    noMHRC0[i, t].where[Ord(i) < Card(i)] = Number(1).where[
        F[i, t] / pt[i + 1, t] < F[i - 1, t] / pt[i, t]
    ]
    noMHRC[t].where[Sum(i, noMHRC0[i, t]) >= 1] = 1

    ru = Parameter(m, name="ru", records=0, description="reservation utility")

    # Definition of Primal/Dual Variables
    x = Variable(m, name="x", type="positive", domain=i, description="quality")
    b = Variable(
        m, name="b", type="positive", domain=i, description="maker's revenue"
    )
    w = Variable(m, name="w", type="positive", domain=i, description="price")

    # Equations
    rev = Equation(
        m,
        name="rev",
        domain=i,
        description="maker's revenue function",
    )
    pc = Equation(
        m,
        name="pc",
        domain=i,
        description="participation constraint",
    )
    licd = Equation(
        m,
        name="licd",
        domain=i,
        description="incentive compatibility constraint",
    )
    licu = Equation(
        m,
        name="licu",
        domain=i,
        description="incentive compatibility constraint",
    )
    ic = Equation(
        m,
        name="ic",
        domain=[i, j],
        description="global incentive compatibility constraint",
    )
    mn = Equation(
        m,
        name="mn",
        domain=i,
        description="monotonicity constraint",
    )

    obj = Sum(i, p[i] * (b[i] - w[i]))  # Objective Function; maker's utility

    rev[i] = b[i] == x[i] ** (0.5)

    pc[i] = w[i] - theta[i] * x[i] >= ru

    licd[i] = w[i] - theta[i] * x[i] >= w[i + 1] - theta[i] * x[i + 1]

    licu[i] = w[i] - theta[i] * x[i] >= w[i - 1] - theta[i] * x[i - 1]

    ic[i, j] = w[i] - theta[i] * x[i] >= w[j] - theta[i] * x[j]

    mn[i] = x[i] >= x[i + 1]

    # Setting Lower Bounds on Variables to Avoid Division by Zero
    x.lo[i] = 0.0001

    # Models
    SB_lic = Model(
        m,
        name="SB_lic",
        equations=[rev, pc, licd],
        problem=Problem.NLP,
        sense=Sense.MAX,
        objective=obj,
    )
    SB_lic2 = Model(
        m,
        name="SB_lic2",
        equations=[rev, pc, licd, mn],
        problem=Problem.NLP,
        sense=Sense.MAX,
        objective=obj,
    )

    # Parameters
    Util_lic = Parameter(
        m, name="Util_lic", domain=t, description="util solved w/o MN"
    )
    Util_lic2 = Parameter(
        m, name="Util_lic2", domain=t, description="util solved w/ MN"
    )
    Util_gap = Parameter(
        m,
        name="Util_gap",
        domain=t,
        description="gap between these two util",
    )
    x_lic = Parameter(
        m, name="x_lic", domain=[i, t], description="x solved in w/o MN"
    )
    x_lic2 = Parameter(
        m, name="x_lic2", domain=[i, t], description="x solved in w/ MN"
    )
    MN_lic = Parameter(
        m,
        name="MN_lic",
        domain=t,
        description="monotonicity of x solved w/o MN",
    )
    MN_lic2 = Parameter(
        m,
        name="MN_lic2",
        domain=t,
        description="monotonicity of x solved w/ MN",
    )

    for tt, _ in t.records.itertuples(index=False):
        p[i] = pt[i, tt]

        #  Solving the model w/o MN
        SB_lic.solve(options=Options(solve_link_type="memory"))

        Util_lic[tt] = SB_lic.objective_value
        x_lic[i, tt] = x.l[i]
        MN_lic[tt] = Sum(
            i, Number(1).where[Round(x.l[i], 10) < Round(x.l[i + 1], 10)]
        )

        #  Solving the model w/ MN
        SB_lic2.solve(options=Options(solve_link_type="memory"))

        Util_lic2[tt] = SB_lic2.objective_value
        x_lic2[i, tt] = x.l[i]
        MN_lic2[tt] = Sum(
            i, Number(1).where[Round(x.l[i], 10) < Round(x.l[i + 1], 10)]
        )

    Util_gap[t] = Number(1).where[
        Round(Util_lic[t], 10) != Round(Util_lic2[t], 10)
    ]

    # Computing probability that MHRC and MN holds.
    p_noMHRC = Parameter(
        m, name="p_noMHRC", description="no MHRC case          [%]"
    )
    p_noMN_lic = Parameter(
        m, name="p_noMN_lic", description="no MN case            [%]"
    )
    p_Util_gap = Parameter(
        m, name="p_Util_gap", description="no util-equality case [%]"
    )

    p_noMHRC[...] = Sum(t.where[noMHRC[t] > 0], 1) / Card(t) * 100
    p_noMN_lic[...] = Sum(t.where[MN_lic[t] > 0], 1) / Card(t) * 100
    p_Util_gap[...] = Sum(t.where[Util_gap[t] > 0], 1) / Card(t) * 100

    print(f"no MHRC case: {p_noMHRC.records.value[0]}%")
    print(f"no MN case: {p_noMN_lic.records.value[0]}%")
    print(f"no util-equality case: {p_Util_gap.records.value[0]}%")
    print(f"Time Elapsed: {round(time.time() - start, 2)}s")


if __name__ == "__main__":
    main()
