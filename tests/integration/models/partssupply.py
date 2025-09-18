"""
## GAMSSOURCE: https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_partssupply.html
## LICENSETYPE: Demo
## MODELTYPE: NLP
## KEYWORDS: nonlinear programming, contract theory, principal-agent problem, adverse selection, parts supply problem


Parts Supply Problem (PARTSSUPPLY)

This model is based on the ps2_f_s.358 .. ps10_s_mn.396 models by
Hideo Hashimoto, Kojun Hamada, and Nobuhiro Hosoe.

Using the following options, these models can be run:

ps2_f              : default
ps2_f_eff          : --nsupplier=1
ps2_f_inf          : --nsupplier=1 --alttheta=1
ps2_f_s            : --useic=1
ps2_s              : --useic=1
ps3_f              : --nsupplier=3
ps3_s              : --nsupplier=3  --uselicd=1
ps3_s_gic          : --nsupplier=3  --useic=1
ps3_s_mn  1st solve: --nsupplier=3  --uselicd=1
          2nd solve: --nsupplier=3  --uselicd=1 --altpi=1
          3rd solve: --nsupplier=3  --uselicd=1 --alttheta=1
ps3_s_scp 1st solve: --nsupplier=3  --alttheta=2 --modweight=1 --useic=1
          2nd solve: --nsupplier=3  --alttheta=2 --modweight=1 --uselicd=1
          --uselicu=1
ps5_s_mn           : --nsupplier=5  --uselicd=1 --nsamples=1000
ps10_s             : --nsupplier=10 --uselicd=1
ps10_s_mn          : --nsupplier=10 --uselicd=1 --nsamples=1000

Alternatively, the corresponding original model files can be found in
the GAMS model library.
"""

from __future__ import annotations

import gamspy.math as gams_math
from gamspy import (
    Alias,
    Card,
    Container,
    Equation,
    Model,
    Ord,
    Parameter,
    Problem,
    Sense,
    Set,
    Sum,
    Variable,
)


def main():
    cont = Container()

    # Set
    i = Set(cont, name="i", records=["1", "2"], description="type of supplier")
    t = Set(cont, name="t", records=["1"], description="Monte-Carlo draws")
    j = Alias(cont, name="j", alias_with=i)

    # Parameter
    theta = Parameter(
        cont,
        name="theta",
        domain=i,
        records=[[1, 0.2], [2, 0.3]],
        description="efficiency",
    )
    pt = Parameter(cont, name="pt", domain=[i, t], description="probability of type")
    p = Parameter(
        cont,
        name="p",
        domain=i,
        records=[[1, 0.2], [2, 0.8]],
        description="probability of type for currently evaluated scenario",
    )
    icweight = Parameter(
        cont, name="icweight", domain=i, description="weight in ic constraints"
    )
    ru = Parameter(cont, name="ru", records=0, description="reservation utility")

    pt[i, t] = gams_math.uniform(0, 1)
    pt[i, t] = pt[i, t] / Sum(j, pt[j, t])
    pt[i, t] = p[i]

    # Variable
    x = Variable(cont, name="x", domain=i, type="Positive", description="quality")
    b = Variable(
        cont,
        name="b",
        domain=i,
        type="Positive",
        description="maker's revenue",
    )
    w = Variable(cont, name="w", domain=i, type="Positive", description="price")

    # Equation
    rev = Equation(cont, name="rev", domain=i, description="maker's revenue function")
    pc = Equation(cont, name="pc", domain=i, description="participation constraint")
    ic = Equation(
        cont,
        name="ic",
        domain=[i, j],
        description="incentive compatibility constraint",
    )
    licd = Equation(
        cont,
        name="licd",
        domain=i,
        description="incentive compatibility constraint",
    )
    licu = Equation(
        cont,
        name="licu",
        domain=i,
        description="incentive compatibility constraint",
    )
    mn = Equation(cont, name="mn", domain=i, description="monotonicity constraint")

    # maker's utility function
    obj = Sum(i, p[i] * (b[i] - w[i]))

    rev[i] = b[i] == gams_math.sqrt(x[i])
    pc[i] = w[i] - theta[i] * x[i] >= ru
    ic[i, j] = w[i] - icweight[i] * x[i] >= w[j] - icweight[i] * x[j]
    licd[i].where[Ord(i) < Card(i)] = (
        w[i] - icweight[i] * x[i] >= w[i + 1] - icweight[i] * x[i + 1]
    )
    licu[i].where[Ord(i) > 1] = (
        w[i] - icweight[i] * x[i] >= w[i - 1] - icweight[i] * x[i - 1]
    )
    mn[i].where[Ord(i) < Card(i)] = x[i] >= x[i + 1]

    x.lo[i] = 0.0001

    m = Model(
        cont,
        name="m",
        equations=[rev, pc, licu],
        problem=Problem.NLP,
        sense=Sense.MAX,
        objective=obj,
    )
    m_mn = Model(
        cont,
        name="m_mn",
        equations=[rev, pc, licu, mn],
        problem=Problem.NLP,
        sense=Sense.MAX,
        objective=obj,
    )

    for iter, _ in t.records.itertuples(index=False):
        p[i] = pt[i, iter]
        icweight[i] = theta[i]
        m.solve()
        m_mn.solve()

    import math

    assert math.isclose(m_mn.objective_value, 0.9167, rel_tol=0.001)


if __name__ == "__main__":
    main()
