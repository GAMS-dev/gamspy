"""
## GAMSSOURCE: https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_qdemo7.html
## LICENSETYPE: Demo
## MODELTYPE: QCP
## DATAFILES: qdemo7.gdx
## KEYWORDS: quadratic constraint programming, farming, agricultural economics, partial equilibrium, market behavior


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
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from gamspy import (
    Container,
    Equation,
    Model,
    Number,
    Parameter,
    Problem,
    Sense,
    Sum,
    Variable,
)
from gamspy.math import sqr


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
    fnum = Parameter(
        m, name="fnum", records=1000, description="number  of  farms in sector"
    )
    land = Parameter(
        m,
        name="land",
        records=4,
        description="farmsize                           (hectares)",
    )
    famlab = Parameter(
        m,
        name="famlab",
        records=25,
        description="family labor available       (days per month)",
    )
    rwage = Parameter(
        m,
        name="rwage",
        records=3,
        description="reservation wage rate       (dollars per day)",
    )
    twage = Parameter(
        m,
        name="twage",
        records=4,
        description="temporary labor wage        (dollars per day)",
    )
    llab = Parameter(
        m,
        name="llab",
        records=2,
        description="livestock labor requirements (days per month)",
    )
    trent = Parameter(
        m,
        name="trent",
        records=40,
        description="tractor rental cost      (dollar per hectare)",
    )
    hpa = Parameter(
        m,
        name="hpa",
        records=2,
        description="land plowed by animals  (hectares per animal)",
    )
    straw = Parameter(
        m, name="straw", records=1.75, description="straw yield from wheat"
    )

    # Parameter
    yields = Parameter(
        m,
        name="yields",
        domain=c,
        records=np.array([1.5, 6, 1, 3, 1.5, 2, 3]),
        description="crop yield         (tons per hectare)",
    )
    miscost = Parameter(
        m,
        name="miscost",
        domain=c,
        records=np.array([10, 0, 5, 50, 80, 5, 50]),
        description="misc cash costs (dollars per hectare)",
    )
    price = Parameter(
        m,
        name="price",
        domain=c,
        description="reference (observed) price  (dollars)",
    )
    pe = Parameter(
        m,
        name="pe",
        domain=c,
        description="commodity export prices     (dollars)",
    )
    pm = Parameter(
        m,
        name="pm",
        domain=c,
        description="commodity import prices     (dollars)",
    )
    alpha = Parameter(
        m, name="alpha", domain=c, description="demand curve intercept"
    )
    beta = Parameter(
        m, name="beta", domain=c, description="demand curve gradient"
    )

    cn[c] = Number(1).where[demdat[c, "ref-p"]]
    ce[c] = Number(1).where[demdat[c, "exp-p"]]
    cm[c] = Number(1).where[(demdat[c, "imp-p"] < np.inf)]
    cm["clover"] = False
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
    xcrop = Variable(
        m,
        name="xcrop",
        type="positive",
        domain=c,
        description="cropping activity                 (hectares)",
    )
    mcost = Variable(
        m,
        name="mcost",
        description="misc cash cost                     (dollars)",
    )
    pcost = Variable(m, name="pcost", description="tractor plowing cost")
    labcost = Variable(
        m,
        name="labcost",
        description="labor cost                         (dollars)",
    )
    rescost = Variable(
        m,
        name="rescost",
        description="family labor reservation wage cost (dollars)",
    )
    tcost = Variable(
        m, name="tcost", description="total farm cost including rescost"
    )
    flab = Variable(
        m,
        name="flab",
        type="positive",
        domain=t,
        description="family labor use                      (days)",
    )
    tlab = Variable(
        m,
        name="tlab",
        type="positive",
        domain=t,
        description="temporary labor                       (days)",
    )
    xlive = Variable(
        m,
        name="xlive",
        type="positive",
        domain=r,
        description="livestock activity                   (units)",
    )
    natprod = Variable(
        m,
        name="natprod",
        type="positive",
        domain=c,
        description="net production                        (tons)",
    )
    thire = Variable(
        m,
        name="thire",
        type="positive",
        domain=s,
        description="tractor rental             (hectares plowes)",
    )
    natcon = Variable(
        m,
        name="natcon",
        type="positive",
        domain=c,
        description="domestic consumption             (1000 tons)",
    )
    exports = Variable(
        m,
        name="exports",
        type="positive",
        domain=c,
        description="national exports                 (1000 tons)",
    )
    imports = Variable(
        m,
        name="imports",
        type="positive",
        domain=c,
        description="national imports                 (1000 tons)",
    )

    # Equation
    landbal = Equation(
        m,
        name="landbal",
        domain=t,
        description="land balance             (hectares)",
    )
    laborbal = Equation(
        m,
        name="laborbal",
        domain=t,
        description="labor balance                (days)",
    )
    plow = Equation(
        m,
        name="plow",
        domain=s,
        description="land plowed   (hectares per season)",
    )
    ares = Equation(
        m, name="ares", description="reservation labor cost    (dollars)"
    )
    acost = Equation(
        m, name="acost", description="total cost accounting     (dollars)"
    )
    amisc = Equation(m, name="amisc", description="misc cost accounting")
    aplow = Equation(m, name="aplow")
    alab = Equation(
        m, name="alab", description="labor cost accounting     (dollars)"
    )
    lclover = Equation(m, name="lclover", description="clover balance")
    lstraw = Equation(m, name="lstraw", description="straw balance")
    proc = Equation(
        m,
        name="proc",
        domain=c,
        description="net production definition    (tons)",
    )
    dem = Equation(
        m,
        name="dem",
        domain=c,
        description="national demand balance (1000 tons)",
    )

    landbal[t] = Sum(c, xcrop[c] * a[t, c]) <= land * fnum

    laborbal[t] = (
        Sum(c, xcrop[c] * lc[t, c]) + Sum(r, xlive[r]) * llab
        <= flab[t] + tlab[t]
    )

    amisc[...] = mcost == Sum(c, xcrop[c] * miscost[c])

    alab[...] = labcost == Sum(t, tlab[t] * twage)

    ares[...] = rescost == Sum(t, flab[t] * rwage)

    aplow[...] = pcost == Sum(s, thire[s] * trent)

    acost[...] = tcost == mcost + labcost + rescost + pcost

    lclover[...] = xcrop["clover"] * yields["clover"] >= Sum(
        r, xlive[r] * lio["clover", r]
    )

    lstraw[...] = xcrop["wheat"] * straw >= Sum(r, xlive[r] * lio["straw", r])

    plow[s] = (
        Sum(c.where[sc[s, c]], xcrop[c]) <= Sum(r, xlive[r]) * hpa + thire[s]
    )

    proc[c] = natprod[c] == xcrop[c] * yields[c]

    dem[cn] = (
        natcon[cn]
        == natprod[cn] + imports[cn].where[cm[cn]] - exports[cn].where[ce[cn]]
    )

    # Objective Function; consumers and producers surplus
    objn = (
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
        ],
        problem=Problem.QCP,
        sense=Sense.MAX,
        objective=objn,
    )

    demo7n.solve()

    print("Value of objective:  ", round(demo7n.objective_value, 3))


if __name__ == "__main__":
    main()
