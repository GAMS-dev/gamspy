"""
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

Keywords: nonlinear programming, contract theory, principal-agent problem,
          adverse selection, parts supply problem
"""

from gamspy import Alias, Set, Parameter, Variable, Equation, Model, Container
from gamspy import Sum, Card, Ord
import gamspy.math as gams_math
from gamspy import Problem, Sense


def main():
    cont = Container()

    # Set
    i = Set(cont, name="i", records=["1", "2"])
    t = Set(cont, name="t", records=["1"])
    j = Alias(cont, name="j", alias_with=i)

    # Parameter
    theta = Parameter(
        cont, name="theta", domain=[i], records=[[1, 0.2], [2, 0.3]]
    )
    pt = Parameter(cont, name="pt", domain=[i, t])
    p = Parameter(cont, name="p", domain=[i], records=[[1, 0.2], [2, 0.8]])
    icweight = Parameter(cont, name="icweight", domain=[i])
    ru = Parameter(cont, name="ru", records=0)

    pt[i, t] = gams_math.uniform(0, 1)
    pt[i, t] = pt[i, t] / Sum(j, pt[j, t])
    pt[i, t] = p[i]

    # Variable
    x = Variable(cont, name="x", domain=[i], type="Positive")
    b = Variable(cont, name="b", domain=[i], type="Positive")
    w = Variable(cont, name="w", domain=[i], type="Positive")
    util = Variable(cont, name="util")

    # Equation
    obj = Equation(cont, name="obj")
    rev = Equation(cont, name="rev", domain=[i])
    pc = Equation(cont, name="pc", domain=[i])
    ic = Equation(cont, name="ic", domain=[i, j])
    licd = Equation(cont, name="licd", domain=[i])
    licu = Equation(cont, name="licu", domain=[i])
    mn = Equation(cont, name="mn", domain=[i])

    obj.expr = util == Sum(i, p[i] * (b[i] - w[i]))
    rev[i] = b[i] == gams_math.sqrt(x[i])
    pc[i] = w[i] - theta[i] * x[i] >= ru
    ic[i, j] = w[i] - icweight[i] * x[i] >= w[j] - icweight[i] * x[j]
    licd[i].where[Ord(i) < Card(i)] = (
        w[i] - icweight[i] * x[i]
        >= w[i.lead(1, "linear")] - icweight[i] * x[i.lead(1, "linear")]
    )
    licu[i].where[Ord(i) > 1] = (
        w[i] - icweight[i] * x[i]
        >= w[i.lag(1, "linear")] - icweight[i] * x[i.lag(1, "linear")]
    )
    mn[i].where[Ord(i) < Card(i)] = x[i] >= x[i.lead(1, "linear")]

    x.lo[i] = 0.0001

    m = Model(
        cont,
        name="m",
        equations=[obj, rev, pc, licu],
        problem=Problem.NLP,
        sense=Sense.MAX,
        objective=util,
    )
    m_mn = Model(
        cont,
        name="m_mn",
        equations=[obj, rev, pc, licu, mn],
        problem=Problem.NLP,
        sense=Sense.MAX,
        objective=util,
    )

    cont.addOptions({"limRow": 0, "limCol": 0})

    for iter, _ in t.records.itertuples(index=False):
        p[i] = pt[i, iter]
        icweight[i] = theta[i]
        m.solve()
        m_mn.solve()
        cont.addOptions({"solPrint": "off"})


if __name__ == "__main__":
    main()
