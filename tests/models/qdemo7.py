"""
Nonlinear Simple Agricultural Sector Model (QDEMO7)

This is a QCP version of the gamslib model DEMO7. The original NLP
formulation was concerned with good starting points. QCPs do not
need starting a point.

This is the last in a series of agricultural farm level and sector
models, this model simulates the market behavior of the sector
using a partial equilibrium framework. The technique is
the maximization of consumers and producers surplus.


Kutcher, G P, Meeraus, A, and O'Mara, G T, Agriculture Sector and
Policy Models. The World Bank, 1988.

Keywords: quadratic constraint programming, farming, agricultural economics,
          partial equilibrium, market behavior
"""

from pathlib import Path
from gamspy import Parameter, Variable, Equation, Container, Model, Sum, Number
from gamspy.math import power
import numpy as np


def sqr(x):
    return power(x, 2)


def main():
    m = Container(
        load_from=str(Path(__file__).parent.absolute()) + "/qdemo7.gdx",
    )

    # Sets
    c, cl, t, r, s, sc, cn, ce, cm = m.getSymbols(
        ["c", "cl", "t", "r", "s", "sc", "cn", "ce", "cm"]
    )

    # Parameters
    a, lc, lio, demdat = m.getSymbols(["a", "lc", "lio", "demdat"])

    # Scalar
    fnum = Parameter(m, name="fnum", records=1000)
    land = Parameter(m, name="land", records=4)
    famlab = Parameter(m, name="famlab", records=25)
    dpm = Parameter(m, name="dpm", records=25)
    rwage = Parameter(m, name="rwage", records=3)
    twage = Parameter(m, name="twage", records=4)
    llab = Parameter(m, name="llab", records=2)
    trent = Parameter(m, name="trent", records=40)
    hpa = Parameter(m, name="hpa", records=2)
    straw = Parameter(m, name="straw", records=1.75)

    # Parameter
    yields = Parameter(
        m,
        name="yields",
        domain=[c],
        records=np.array([1.5, 6, 1, 3, 1.5, 2, 3]),
    )
    miscost = Parameter(
        m,
        name="miscost",
        domain=[c],
        records=np.array([10, 0, 5, 50, 80, 5, 50]),
    )
    price = Parameter(m, name="price", domain=[c])
    pe = Parameter(m, name="pe", domain=[c])
    pm = Parameter(m, name="pm", domain=[c])
    alpha = Parameter(m, name="alpha", domain=[c])
    beta = Parameter(m, name="beta", domain=[c])

    cn[c] = Number(1).where[demdat[c, "ref-p"]]
    ce[c] = Number(1).where[demdat[c, "exp-p"]]
    cm[c] = Number(1).where[(demdat[c, "imp-p"] < np.inf)]
    cm["clover"] = Number(0)
    price[c] = demdat[c, "ref-p"]
    pe[ce] = demdat[ce, "exp-p"]
    pm[cm] = demdat[cm, "imp-p"]

    beta[cn].where[demdat[cn, "ref-q"]] = (
        demdat[cn, "ref-p"] / demdat[cn, "ref-q"] / demdat[cn, "elas"]
    )
    alpha[cn] = demdat[cn, "ref-p"] - beta[cn] * demdat[cn, "ref-q"]
    demdat[cn, "dem-a"] = alpha[cn]
    demdat[cn, "dem-b"] = beta[cn]

    # Variables
    xcrop = Variable(m, name="xcrop", type="positive", domain=[c])
    mcost = Variable(m, name="mcost")
    pcost = Variable(m, name="pcost")
    labcost = Variable(m, name="labcost")
    rescost = Variable(m, name="rescost")
    tcost = Variable(m, name="tcost")
    flab = Variable(m, name="flab", type="positive", domain=[t])
    tlab = Variable(m, name="tlab", type="positive", domain=[t])
    xlive = Variable(m, name="xlive", type="positive", domain=[r])
    natprod = Variable(m, name="natprod", type="positive", domain=[c])
    thire = Variable(m, name="thire", type="positive", domain=[s])
    natcon = Variable(m, name="natcon", type="positive", domain=[c])
    exports = Variable(m, name="exports", type="positive", domain=[c])
    imports = Variable(m, name="imports", type="positive", domain=[c])
    cps = Variable(m, name="cps")

    # Equation
    landbal = Equation(m, name="landbal", type="leq", domain=[t])
    laborbal = Equation(m, name="laborbal", type="leq", domain=[t])
    plow = Equation(m, name="plow", type="leq", domain=[s])
    ares = Equation(m, name="ares", type="eq")
    acost = Equation(m, name="acost", type="eq")
    amisc = Equation(m, name="amisc", type="eq")
    aplow = Equation(m, name="aplow", type="eq")
    alab = Equation(m, name="alab", type="eq")
    lclover = Equation(m, name="lclover", type="geq")
    lstraw = Equation(m, name="lstraw", type="geq")
    proc = Equation(m, name="proc", type="eq", domain=[c])
    dem = Equation(m, name="dem", type="eq", domain=[c])
    objn = Equation(m, name="objn", type="eq")

    landbal[t] = Sum(c, xcrop[c] * a[t, c]) <= land * fnum

    laborbal[t] = (
        Sum(c, xcrop[c] * lc[t, c]) + Sum(r, xlive[r]) * llab
        <= flab[t] + tlab[t]
    )

    amisc.definition = mcost == Sum(c, xcrop[c] * miscost[c])

    alab.definition = labcost == Sum(t, tlab[t] * twage)

    ares.definition = rescost == Sum(t, flab[t] * rwage)

    aplow.definition = pcost == Sum(s, thire[s] * trent)

    acost.definition = tcost == mcost + labcost + rescost + pcost

    lclover.definition = xcrop["clover"] * yields["clover"] >= Sum(
        r, xlive[r] * lio["clover", r]
    )

    lstraw.definition = xcrop["wheat"] * straw >= Sum(
        r, xlive[r] * lio["straw", r]
    )

    plow[s] = (
        Sum(c.where[sc[s, c]], xcrop[c]) <= Sum(r, xlive[r]) * hpa + thire[s]
    )

    proc[c] = natprod[c] == xcrop[c] * yields[c]

    dem[cn] = (
        natcon[cn]
        == natprod[cn] + imports[cn].where[cm[cn]] - exports[cn].where[ce[cn]]
    )

    objn.definition = cps == (
        Sum(cn, alpha[cn] * natcon[cn] + 0.5 * beta[cn] * sqr(natcon[cn]))
        + Sum(ce, exports[ce] * pe[ce])
        - Sum(cm, imports[cm] * pm[cm])
        - tcost
    )

    flab.up[t] = famlab * fnum

    demo7n = Model(
        m,
        name="demo7n",
        equations=[
            landbal,
            laborbal,
            plow,
            ares,
            alab,
            acost,
            dem,
            proc,
            amisc,
            aplow,
            lclover,
            lstraw,
            objn,
        ],
    )

    m.addOptions({"limCol": 0, "limRow": 0})

    m.solve(demo7n, problem="qcp", sense="max", objective_variable=cps)

    print("Value of objective:  ", round(cps.records.level[0], 3))


if __name__ == "__main__":
    main()
