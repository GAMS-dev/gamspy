"""
## GAMSSOURCE: https://gams.com/latest/gamslib_ml/libhtml/gamslib_traffic.html
## LICENSETYPE: Requires license
## MODELTYPE: NLP, MCP
## KEYWORDS: mixed complementarity problem, nonlinear programming, traffic equilibria, transportation, multicommodity flows


Traffic Equilibrium Problem

Three different models are used to compute traffic equilibria. These are
a mixed complementarity formulation and a primal and dual formulation
using NLPs.


Ferris, M C, Meeraus, A, and Rutherford, T F, Computing Wardropian
Equilibria in a Complementarity Framework. Optimization Methods and
Software 10 (1999), 669-685.
"""

from pathlib import Path

import gamspy as gp


def main():
    m = gp.Container()

    n = gp.Set(m, "n", description="nodes", records=range(1, 25))
    a = gp.Set(m, "a", domain=[n, n], description="directed arcs")
    param = gp.Set(m, "param", records=["a", "b", "k"])

    i, j, k = gp.Alias(m, "i", n), gp.Alias(m, "j", n), gp.Alias(m, "k", n)

    trip = gp.Parameter(m, "trip", domain=[n, n], description="trip table")
    ca = gp.Parameter(m, "ca", domain=[n, n], description="cost coef A")
    cb = gp.Parameter(m, "cb", domain=[n, n], description="cost coef B")
    ck = gp.Parameter(m, "ck", domain=[n, n], description="cost coef K")
    arc_cost = gp.Parameter(
        m, "arc_cost", domain=[n, n, param], description="arc cost data"
    )
    m.loadRecordsFromGdx(
        str(Path(__file__).parent.absolute()) + "/traffic.gdx",
        symbol_names=["arc_cost", "trip"],
    )

    arc_cost[i, j, param] << arc_cost[j, i, param]

    ca[i, j] = arc_cost[i, j, "a"]
    cb[i, j] = arc_cost[i, j, "b"]
    ck[i, j] = arc_cost[i, j, "k"]

    trip[i, j] << trip[j, i]
    trip[i, j] = trip[i, j] * 0.11

    a[i, j] = ca[i, j]

    t = gp.Variable(
        m, "t", domain=[i, j], description="time to get from node i to node j"
    )
    v = gp.Variable(
        m, "v", domain=[i, j], description="time to traverse arc form i to j"
    )
    y = gp.Variable(
        m,
        "y",
        domain=[i, j, k],
        type="Positive",
        description="flow to k along arc i-j",
    )
    x = gp.Variable(
        m, "x", domain=[i, j], description="aggregate flow on arc i-j"
    )
    objpnlp = gp.Variable(
        m, "objpnlp", description="objective for nlp formulation"
    )
    objdnlp = gp.Variable(
        m, "objdnlp", description="objective for nlp formulation"
    )

    balance = gp.Equation(
        m, "balance", domain=[i, j], description="material balance"
    )
    vdef = gp.Equation(
        m, "vdef", domain=[i, j], description="arc travel time definition"
    )
    rational = gp.Equation(
        m,
        "rational",
        domain=[i, j, k],
        description="cost minimization condition",
    )
    xdef = gp.Equation(
        m, "xdef", domain=[i, j], description="aggregate flow definition"
    )
    defpnlp = gp.Equation(
        m,
        "defpnlp",
        description="defines objective for primal nlp formulation",
    )
    defdnlp = gp.Equation(
        m, "defdnlp", description="defines objective for dual nlp formulation"
    )

    balance[i, k].where[~i.sameAs(k)] = (
        gp.Sum(a[i, j], y[a, k]) == gp.Sum(a[j, i], y[a, k]) + trip[i, k]
    )
    rational[a[i, j], k].where[~i.sameAs(k)] = v[a] + t[j, k] >= t[i, k]
    vdef[a] = v[a] == ca[a] + cb[a] * gp.math.power(x[a] / ck[a], 4)
    xdef[a] = x[a] == gp.Sum(k, y[a, k])
    defpnlp[...] = objpnlp == gp.Sum(
        a, ca[a] * x[a] + cb[a] * gp.math.power(x[a] / ck[a], 5) * ck[a] / 5
    )
    defdnlp[...] = objdnlp == gp.Sum((i, k), trip[i, k] * t[i, k]) - gp.Sum(
        a, (4 / 5) * (ck[a] / cb[a] ** (1 / 4)) * (v[a] - ca[a]) ** 1.25
    )

    pnlp = gp.Model(
        m,
        "pnlp",
        equations=[defpnlp, balance, xdef],
        problem=gp.Problem.NLP,
        sense=gp.Sense.MIN,
        objective=objpnlp,
    )
    dnlp = gp.Model(
        m,
        "dnlp",
        equations=[defdnlp, rational],
        problem=gp.Problem.NLP,
        sense=gp.Sense.MAX,
        objective=objdnlp,
    )
    mcp = gp.Model(
        m,
        "mcp",
        equations=[xdef],
        matches={rational: y, balance: t, vdef: v},
        problem=gp.Problem.MCP,
    )

    t.fx[i, i] = 0
    v.lo[a] = ca[a]
    y.fx[a[i, j], i] = 0

    rep = gp.Parameter(
        m, "rep", domain=[i, k, "*"], description="summary report"
    )

    mcp.solve(options=gp.Options(domain_violation_limit=100000))
    pnlp.solve(
        solver="conopt", options=gp.Options(domain_violation_limit=100000)
    )
    dnlp.solve(
        solver="conopt", options=gp.Options(domain_violation_limit=100000)
    )
    rep[i, j, "mcp"] = t.l[i, j]
    rep[i, j, "primal"] = balance.m[i, j]
    rep[i, j, "dual"] = t.l[i, j]

    print(rep.records)

    import math

    assert math.isclose(
        mcp.objective_value, 5.485113163672395e-08, abs_tol=1e-6
    )
    assert math.isclose(pnlp.objective_value, 5148.830098447451, abs_tol=1e-6)
    assert math.isclose(dnlp.objective_value, 5148.830098449257, abs_tol=1e-6)


if __name__ == "__main__":
    main()
