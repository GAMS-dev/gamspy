"""
## GAMSSOURCE: https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_chenery.html
## LICENSETYPE: Demo
## MODELTYPE: NLP
## DATAFILES: chenery.gdx
## KEYWORDS: nonlinear programming, econometrics, economic development


Substitution and Structural Change (CHENERY)

This model follows conventional input-output formulations for production
with nonlinear demand functions, import and export functions and production
functions for direct factor use.


Chenery, H B, and Raduchel, W J, Substitution and Structural Change.
In Chenery, H B, Ed, Structural Change and Development Policy. Oxford
University Press, New York and Oxford, 1979.
"""

from __future__ import annotations

import os
from pathlib import Path

from gamspy import Container
from gamspy import Model
from gamspy import Number
from gamspy import Problem
from gamspy import Sense
from gamspy import Sum


def main():
    container = Container(
        system_directory=os.getenv("SYSTEM_DIRECTORY", None),
        load_from=str(Path(__file__).parent.absolute()) + "/chenery.gdx",
    )

    # Sets
    i, t, j = container.getSymbols(["i", "t", "j"])

    # Parameters
    (
        aio,
        pdat,
        ddat,
        tdat,
        mew,
        xsi,
        gam,
        alp,
        ynot,
        sig,
        thet,
        rho,
        deli,
        efy,
    ) = container.getSymbols([
        "aio",
        "pdat",
        "ddat",
        "tdat",
        "mew",
        "xsi",
        "gam",
        "alp",
        "ynot",
        "sig",
        "thet",
        "rho",
        "del",
        "efy",
    ])
    lbar, plab, kbar, dbar = container.getSymbols(
        ["lbar", "plab", "kbar", "dbar"]
    )

    # Variables
    x, v, y, p, l, k, e = container.getSymbols(
        ["x", "v", "y", "p", "l", "k", "e"]
    )
    m, g, h, pk, pi, pd, td, vv = container.getSymbols(
        ["m", "g", "h", "pk", "pi", "pd", "td", "vv"]
    )

    # Equations
    dty, mb, tb, dg, dh, dem = container.getSymbols(
        ["dty", "mb", "tb", "dg", "dh", "dem"]
    )
    lc, kc, sup, fpr, dvv, dl, dk, dv = container.getSymbols(
        ["lc", "kc", "sup", "fpr", "dvv", "dl", "dk", "dv"]
    )

    dty[...] = td == Sum(i, y[i])
    mb[i] = x[i] >= y[i] + Sum(j, aio[i, j] * x[j]) + (e[i] - m[i]).where[t[i]]
    tb[...] = Sum(t, g[t] * m[t] - h[t] * e[t]) <= dbar
    dg[t] = g[t] == mew[t] + xsi[t] * m[t]
    dh[t] = h[t] == gam[t] - alp[t] * e[t]
    dem[i] = y[i] == ynot[i] * (pd * p[i]) ** thet[i]
    lc[...] = Sum(i, l[i] * x[i]) <= lbar
    kc[...] = Sum(i, k[i] * x[i]) == kbar
    sup[i] = p[i] == Sum(j, aio[j, i] * p[j]) + v[i]
    fpr[...] = pi == pk / plab
    dvv[i].where[sig[i] != 0] = vv[i] == (pi * (1 - deli[i]) / deli[i]) ** (
        -rho[i] / (1 + rho[i])
    )
    dl[i] = (
        l[i] * efy[i]
        == ((deli[i] / vv[i] + (1 - deli[i])) ** (1 / rho[i])).where[
            sig[i] != 0
        ]
        + Number(1).where[sig[i] == 0]
    )
    dk[i] = (
        k[i] * efy[i]
        == ((deli[i] + (1 - deli[i]) * vv[i]) ** (1 / rho[i])).where[
            sig[i] != 0
        ]
        + deli[i].where[sig[i] == 0]
    )
    dv[i] = v[i] == pk * k[i] + plab * l[i]

    # Model chenrad 'chenery raduchel model' / all /;
    chenrad = Model(
        container,
        name="chenrad",
        equations=container.getEquations(),
        problem=Problem.NLP,
        sense=Sense.MAX,
        objective=td,
    )

    y.up[i] = 2000
    x.up[i] = 2000
    e.up[t] = 400
    m.up[t] = 400
    g.up[t] = 4
    h.up[t] = 4
    p.up[i] = 100
    p.lo[i] = 0.1
    l.up[i] = 1
    k.up[i] = 1
    pk.lo[...] = 0.25
    pk.up[...] = 4
    pi.lo[...] = 0.25
    pi.up[...] = 4
    v.up[i] = 100
    vv.lo[i] = 0.001

    # select coefficient values for this run
    mew[t] = 1
    xsi[t] = tdat["medium", "xsi", t]
    gam[t] = tdat["medium", "gam", t]
    alp[t] = tdat["medium", "alp", t]
    ynot[i] = ddat["medium", "ynot", i]
    thet[i] = ddat["medium", "p-elas", i]
    sig[i] = pdat["medium", "a", "subst", i]
    deli[i] = pdat["medium", "a", "distr", i]
    efy[i] = pdat["medium", "a", "effic", i]
    rho[i].where[sig[i] != 0] = 1 / sig[i] - 1

    # * initial values for variables
    y.l[i] = 250
    x.l[i] = 200
    e.l[t] = 0
    m.l[t] = 0
    g.l[t] = mew[t] + xsi[t] * m.l[t]
    h.l[t] = gam[t] - alp[t] * e.l[t]
    pd.l[...] = 0.3
    p.l[i] = 3
    pk.l[...] = 3.5
    pi.l[...] = pk.l / plab

    vv.l[i].where[sig[i]] = (pi.l * (1 - deli[i]) / deli[i]) ** (
        -rho[i] / (1 + rho[i])
    )
    l.l[i] = (
        ((deli[i] / vv.l[i] + (1 - deli[i])) ** (1 / rho[i])).where[
            sig[i] != 0
        ]
        + Number(1).where[sig[i] == 0]
    ) / efy[i]
    k.l[i] = (
        ((deli[i] + (1 - deli[i]) * vv.l[i]) ** (1 / rho[i])).where[
            sig[i] != 0
        ]
        + deli[i].where[sig[i] == 0]
    ) / efy[i]
    v.l[i] = pk.l * k.l[i] + plab * l.l[i]
    pd.lo[...] = 0.01
    p.lo[i] = 0.1

    chenrad.solve()

    import math

    assert math.isclose(chenrad.objective_value, 1058.9199, rel_tol=0.001)


if __name__ == "__main__":
    main()
