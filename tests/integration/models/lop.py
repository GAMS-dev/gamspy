"""
## GAMSSOURCE: https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_lop.html
## LICENSETYPE: Demo
## MODELTYPE: LP, MIP
## DATAFILES: lop.gdx
## KEYWORDS: linear programming, mixed integer linear programming, passenger railway optimization, shortest path, dutch railway, public rail transport, network optimization


Line Optimization (LOP)

The problem finds line plans for a given rail network and origin
destination demand data. Models for minimum cost and direct traveler
objectives are given. The set of possible lines is defined by the
shortest paths in the rail network.


Bussieck, M R, Optimal Lines in Public Rail Transport. PhD thesis,
TU Braunschweig, 1998.

Bussieck, M R, Kreuzer, P, and Zimmermann, U T, Optimal Lines for
Railway Systems. European Journal of Operation Research 96, 1 (1996),
54-63.

Claessens, M T, van Dijk, N M, and Zwaneveld, P J, Cost Optimal
Allocation of Rail Passenger Lines. European Journal Operation
Research 110, 3 (1998), 474-489.
"""

# flake8: noqa
from __future__ import annotations

import os
import sys
from pathlib import Path

import gamspy.math as gams_math
from gamspy import Alias
from gamspy import Card
from gamspy import Container
from gamspy import Domain
from gamspy import Equation
from gamspy import Model
from gamspy import Options
from gamspy import Ord
from gamspy import Parameter
from gamspy import Sense
from gamspy import Set
from gamspy import Smax
from gamspy import Sum
from gamspy import Variable
import networkx as nx


def main():
    m = Container(
        system_directory=os.getenv("SYSTEM_DIRECTORY", None),
        load_from=str(Path(__file__).parent.absolute()) + "/lop.gdx",
    )

    # Sets
    s, lf, ac, d = m.getSymbols(["s", "lf", "ac", "d"])

    # Parameters
    rt, tt, lfr, od = m.getSymbols(["rt", "tt", "lfr", "od"])

    # create graph

    G = nx.Graph()
    G.add_nodes_from(s.records["uni"])
    edges = list(
        zip(rt.records["s_0"], rt.records["s_1"], rt.records["value"])
    )
    G.add_weighted_edges_from(edges)

    # Scalars
    mincars, ccap, cfx, crm, trm, cmp, maxtcap = m.getSymbols(
        ["mincars", "ccap", "cfx", "crm", "trm", "cmp", "maxtcap"]
    )

    # Alias
    s1, s2, s3 = m.getSymbols(["s1", "s2", "s3"])

    # Variables
    f, spobj = m.getSymbols(["f", "spobj"])

    predecessors, _ = nx.floyd_warshall_predecessor_and_distance(G)

    lines = dict()
    for n1 in G.nodes():
        for n2 in G.nodes():
            if n1 != n2 and lines.get((n2, n1), None) is None:
                lines[(n1, n2)] = nx.reconstruct_path(n1, n2, predecessors)

    l_rec = []
    rp_rec = []
    for l, path in lines.items():
        for i in range(len(path)):
            if i != len(path) - 1:
                arc = (path[i], path[i + 1])
                l_rec.append(l + (arc))
                l_rec.append(l + (arc[::-1]))
            rp_rec.append(l + (path[i], i + 1))

    l = Set(
        m,
        "l",
        domain=[s, s1, s2, s3],
        description="line from s to s1 with edge s2s3",
        records=l_rec,
    )  # noqa: E741
    root = Alias(m, "root", s)

    ll = Set(
        m, "ll", domain=[s, s], description="station pair represening a line"
    )
    ll[s1, s2] = Ord(s1) < Ord(s2)

    l[root, s, s1, s2].where[~rt[s1, s2]] = False

    rp = Parameter(
        m, "rp", domain=[s, s, s], description="rank of node", records=rp_rec
    )
    lastrp = Parameter(
        m, "lastrp", domain=[s, s], description="rank of the last node in line"
    )
    lastrp[ll] = Smax(s, rp[ll, s])

    load = Parameter(
        m, "load", domain=[s1, s2], description="passenger load of an edge"
    )
    load[s1, s2].where[rt[s1, s2]] = Sum(
        l[root, s, s1, s2].where[od[root, s]], od[root, s]
    )

    dt = Variable(
        m,
        "dt",
        domain=[s1, s2],
        description="direct traveler between s1 and s2",
    )
    freq = Variable(
        m, "freq", domain=[s1, s2], description="frequency on arc s1s2"
    )
    phi = Variable(
        m,
        "phi",
        domain=[s1, s2],
        type="integer",
        description="frequency of line between s1 and s2",
    )
    obj = Variable(m, "obj", description="objective variable")

    deffreqlop = Equation(
        m,
        "deffreqlop",
        domain=[s1, s2],
        description="definition of the frequency for each edge",
    )
    dtlimit = Equation(
        m, "dtlimit", domain=[s1, s2], description="limit the direct travelers"
    )
    defobjdtlop = Equation(m, "defobjdtlop", description="objective function")

    deffreqlop[s1, s2].where[rt[s1, s2]] = freq[s1, s2] == Sum(
        l[ll, s1, s2], phi[ll]
    )

    dtlimit[s1, s2].where[od[s1, s2]] = dt[s1, s2] <= gams_math.Min(
        od[s1, s2], maxtcap
    ) * Sum(ll.where[rp[ll, s1] & rp[ll, s2]], phi[ll])

    defobjdtlop[...] = obj == Sum(Domain(s1, s2).where[od[s1, s2]], dt[s1, s2])

    lopdt = Model(
        m,
        "lopdt",
        equations=[deffreqlop, dtlimit, defobjdtlop],
        problem="mip",
        sense=Sense.MAX,
        objective=obj,
    )

    freq.lo[s1, s2].where[rt[s1, s2]] = gams_math.Max(
        lfr[s1, s2], gams_math.ceil(load[s1, s2] / maxtcap)
    )
    freq.up[s1, s2].where[rt[s1, s2]] = freq.lo[s1, s2]
    dt.up[s1, s2].where[od[s1, s2]] = od[s1, s2]

    lopdt.solve()

    solrep = Parameter(m, "solrep", domain=["*", "*", "*", "*"])
    solsum = Parameter(m, "solsum", domain=["*", "*"])

    solrep["DT", ll, "freq"] = phi.l[ll]
    solrep["DT", ll, "cars"].where[phi.l[ll]] = mincars + Card(ac) - 1

    xcost = Parameter(
        m,
        "xcost",
        domain=[root, s, lf],
        description="operating and capcital cost for line with mincars cars",
    )
    ycost = Parameter(
        m,
        "ycost",
        domain=[root, s, lf],
        description="operating and capcital cost for additional cars",
    )
    length = Parameter(m, "len", domain=[s, s], description="length of line")
    sigma = Parameter(
        m, "sigma", domain=[s, s], description="line circulation factor"
    )

    length[ll] = Sum(l[ll, s1, s2], rt[s1, s2])
    sigma[ll] = (
        length[ll]
        + Sum(s.where[rp[ll, s] == 1], tt[s])
        + Sum(s.where[rp[ll, s] == lastrp[ll]], tt[s])
    ) / 60

    xcost[ll, lf] = (
        Ord(lf) * length[ll] * (trm + mincars * crm)
        + mincars * gams_math.ceil(sigma[ll] * Ord(lf)) * cfx
    )
    ycost[ll, lf] = (
        Ord(lf) * length[ll] * crm + gams_math.ceil(sigma[ll] * Ord(lf)) * cfx
    )

    x = Variable(
        m,
        "x",
        type="binary",
        domain=[s1, s2, lf],
        description="line frequency indicator of line s1-s2",
    )
    y = Variable(
        m,
        "y",
        type="integer",
        domain=[s1, s2, lf],
        description="additional cars on line s1-s2 with frequency lf",
    )

    deffreqilp = Equation(
        m,
        "deffreqilp",
        domain=[s, s],
        description="definition of the frequency for each edge",
    )
    defloadilp = Equation(
        m,
        "defloadilp",
        domain=[s, s],
        description="capacity of lines fulfill the demand",
    )
    oneilp = Equation(
        m, "oneilp", domain=[s, s], description="only one frequency per line"
    )
    couplexy = Equation(
        m, "couplexy", domain=[s, s, lf], description="coupling constraints"
    )
    defobjilp = Equation(
        m, "defobjilp", description="definition of the objective"
    )

    deffreqilp[s1, s2].where[rt[s1, s2]] = freq[s1, s2] == Sum(
        (l[ll, s1, s2], lf), Ord(lf) * x[ll, lf]
    )

    defloadilp[s1, s2].where[rt[s1, s2]] = gams_math.ceil(
        load[s1, s2] / ccap
    ) <= Sum((l[ll, s1, s2], lf), Ord(lf) * (mincars * x[ll, lf] + y[ll, lf]))

    oneilp[ll] = Sum(lf, x[ll, lf]) <= 1

    couplexy[ll, lf] = y[ll, lf] <= y.up[ll, lf] * x[ll, lf]

    defobjilp[...] = obj == Sum(
        (ll, lf), xcost[ll, lf] * x[ll, lf] + ycost[ll, lf] * y[ll, lf]
    )

    ilp = Model(
        m,
        "ilp",
        equations=[defobjilp, deffreqilp, defloadilp, oneilp, couplexy],
        problem="mip",
        sense=Sense.MIN,
        objective=obj,
    )

    y.up[ll, lf] = Card(ac) - 1
    freq.up[s1, s2].where[rt[s1, s2]] = 100

    ilp.solve(options=Options(relative_optimality_gap=0))

    solrep["ILP", ll, "freq"] = Sum(lf.where[x.l[ll, lf]], Ord(lf))
    solrep["ILP", ll, "cars"] = Sum(
        lf.where[x.l[ll, lf]], mincars + y.l[ll, lf]
    )
    solsum["ILP", "cost"] = obj.l

    cap = Parameter(
        m, "cap", domain=[s, s], description="the capacity of a line"
    )
    sol = Set(
        m, "sol", domain=[s, s], description="the actual lines in a line plan"
    )

    dtr = Variable(
        m,
        "dtr",
        domain=[s, s, s, s],
        type="positive",
        description="direct travelers of OD pair u v in line on route s s",
    )

    dtllimit = Equation(
        m,
        "dtllimit",
        domain=[s, s1, s2, s3],
        description="limit direct travelers in line s-s1 on edge s2-s3",
    )
    sumbound = Equation(
        m,
        "sumbound",
        domain=[s, s],
        description="sum of direct travels <= total number of travelers",
    )

    dtllimit[l[sol, s, s1]] = (
        Sum(
            Domain(s2, s3).where[
                od[s2, s3]
                & rp[sol, s2]
                & rp[sol, s3]
                & (
                    (gams_math.Min(rp[sol, s], rp[sol, s1]) >= rp[sol, s2])
                    & (gams_math.Max(rp[sol, s], rp[sol, s1]) <= rp[sol, s3])
                    | (gams_math.Min(rp[sol, s], rp[sol, s1]) >= rp[sol, s3])
                    & (gams_math.Max(rp[sol, s], rp[sol, s1]) <= rp[sol, s2])
                )
            ],
            dtr[sol, s2, s3],
        )
        <= cap[sol]
    )

    sumbound[s2, s3].where[od[s2, s3]] = (
        Sum(sol.where[rp[sol, s2] & rp[sol, s3]], dtr[sol, s2, s3])
        == dt[s2, s3]
    )

    evaldt = Model(
        m,
        "evaldt",
        equations=[dtllimit, sumbound, defobjdtlop],
        problem="lp",
        sense=Sense.MAX,
        objective=obj,
    )

    sol[ll] = solrep["DT", ll, "freq"]
    cap[sol] = solrep["DT", sol, "freq"] * solrep["DT", sol, "cars"] * ccap

    evaldt.solve()

    solsum["DT", "dtrav"] = obj.l
    solsum["DT", "cost"] = Sum(
        sol,
        solrep["DT", sol, "freq"] * length[sol] * trm
        + (
            solrep["DT", sol, "freq"] * length[sol] * crm
            + gams_math.ceil(sigma[sol] * solrep["DT", sol, "freq"]) * cfx
        )
        * solrep["DT", sol, "cars"],
    )

    sol[ll] = solrep["ILP", ll, "freq"]
    cap[sol] = solrep["DT", sol, "freq"] * solrep["DT", sol, "cars"] * ccap

    evaldt.solve()
    solsum["ILP", "dtrav"] = obj.l

    print(solrep.records)
    print(solsum.records)


if __name__ == "__main__":
    main()
