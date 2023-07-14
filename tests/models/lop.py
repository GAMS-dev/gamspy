# flake8: noqa

import sys
from pathlib import Path
from gamspy import (
    Alias,
    Set,
    Parameter,
    Card,
    Domain,
    Sum,
    Smax,
    Variable,
    Model,
    Equation,
    Container,
    Number,
    Ord,
)
import gamspy.math as math


def main():
    m = Container(
        load_from=str(Path(__file__).parent.absolute()) + "/lop.gdx",
        system_directory="/opt/gams/gams44.0_linux_x64_64_sfx",
    )

    # Sets
    s, lf, ac, d = m.getSymbols(["s", "lf", "ac", "d"])

    # Parameters
    rt, tt, lfr, od = m.getSymbols(["rt", "tt", "lfr", "od"])

    # Scalars
    mincars, ccap, cfx, crm, trm, cmp, maxtcap = m.getSymbols(
        ["mincars", "ccap", "cfx", "crm", "trm", "cmp", "maxtcap"]
    )

    # Alias
    s1, s2, s3 = m.getSymbols(["s1", "s2", "s3"])

    # Variables
    f, spobj = m.getSymbols(["f", "spobj"])

    # Equations
    balance, defspobj = m.getSymbols(["balance", "defspobj"])

    balance[s, s1] = (
        Sum(d[s1, s2], f[s, d])
        == Sum(d[s2, s1], f[s, d]) + s.sameAs(s1) * Card(s) - 1
    )

    defspobj.definition = spobj == Sum(
        (s, d[s1, s2]), f[s, d] * math.max(rt[s1, s2], rt[s2, s1])
    )

    sp = Model(m, "sp", equations=[balance, defspobj])
    m.solve(sp, "LP", "min", objective_variable=spobj)

    tree = Set(m, "tree", domain=[s, s1, s2])
    tree[s, s1, s2] = f.l[s, s1, s2]

    r = Set(m, "r", records=[str(idx) for idx in range(1, 101)])
    k = Set(m, "k", domain=[s, s])
    v = Set(m, "v", domain=[s, r])
    unvisit = Set(m, "unvisit", domain=[s])
    visit = Set(m, "visit", domain=[s])
    from_ = Set(m, "from", domain=[s])
    to = Set(m, "to", domain=[s])
    l = Set(m, "l", domain=[s, s1, s2, s3])  # noqa: E741
    lr = Set(m, "lr", domain=[s, s1, s2, r])

    root = Alias(m, "root", s)
    r1 = Alias(m, "r1", r)

    l[root, s, s1, s2] = False
    lr[root, s, s1, r] = False

    my_idx = 0
    for root_idx, root_elem in enumerate(root):
        from_[root_elem[0]] = True
        unvisit[s] = True
        visit[s] = False

        for idx, r_elem in enumerate(r):
            if idx > 0 and len(unvisit) > 0:
                unvisit[from_] = False
                visit[from_] = True
                to[unvisit] = Sum(tree[root_elem[0], from_, unvisit], True)

                for f_idx, f_elem in enumerate(from_):
                    k[s2, s3].where[l[root_elem[0], f_elem[0], s2, s3]] = True
                    v[s2, r1].where[lr[root_elem[0], f_elem[0], s2, r1]] = True
                    v[f_elem[0], "1"].where[Card(k) == 0] = True

                    l[root_elem[0], to, k].where[
                        tree[root_elem[0], f_elem[0], to]
                    ] = True
                    lr[root_elem[0], to, v].where[
                        tree[root_elem[0], f_elem[0], to]
                    ] = True
                    l[root_elem[0], to, f_elem[0], to].where[
                        tree[root_elem[0], f_elem[0], to]
                    ] = True
                    lr[root_elem[0], to, to, r_elem[0]].where[
                        tree[root_elem[0], f_elem[0], to]
                    ] = True

                    k[s2, s3] = False
                    v[s2, r1] = False

                from_[s] = False
                from_[to] = True
                to[s] = False
                my_idx += 1

        from_[s] = False

    print(from_.records)
    print(to.records)
    print(unvisit.records)
    print(visit.records)
    exit()
    error02 = Set(m, "error02", domain=[s1, s2])
    error02[s1, s2] = lfr[s1, s2] & Sum(l[root, s, s1, s2], 1) == 0

    if not len(error02) > 0:
        sys.exit(f"There is an error {error02.records}")

    ll = Set(m, "ll", domain=[s, s])
    ll[s1, s2] = Ord(s1) < Ord(s2)

    l[root, s, s1, s2].where[~ll[root, s]] = True
    lr[root, s, s1, r].where[~ll[root, s]] = False

    l[root, s, s1, s2].where[l[root, s, s2, s1] & rt[s1, s2]] = True
    l[root, s, s1, s2].where[~rt[s1, s2]] = False

    rp = Parameter(m, "rp", domain=[s, s, s])
    lastrp = Parameter(m, "lastrp", domain=[s, s])

    rp[ll, s] = Sum(r.where[lr[ll, s, r]], Ord(r))
    lastrp[ll] = Smax(s, rp[ll, s])

    load = Parameter(m, "load", domain=[s1, s2])
    load[s1, s2].where[rt[s1, s2]] = Sum(
        l[root, s, s1, s2].where[od[root, s]], od[root, s]
    )

    dt = Variable(m, "dt", domain=[s1, s2])
    freq = Variable(m, "freq", domain=[s1, s2])
    phi = Variable(m, "phi", domain=[s1, s2], type="integer")
    obj = Variable(m, "obj")

    deffreqlop = Equation(m, "deffreqlop", type="eq", domain=[s1, s2])
    dtlimit = Equation(m, "dtlimit", type="leq", domain=[s1, s2])
    defobjdtlop = Equation(m, "defobjdtlop", type="eq")

    deffreqlop[s1, s2].where[rt[s1, s2]] = freq[s1, s2] == Sum(
        l[ll, s1, s2], phi[ll]
    )

    dtlimit[s1, s2].where[od[s1, s2]] = dt[s1, s2] <= math.min(
        od[s1, s2], maxtcap
    ) * Sum(ll.where[rp[ll, s1] & rp[ll, s2]], phi[ll])

    defobjdtlop.definition = obj == Sum(
        Domain(s1, s2).where[od[s1, s2]], dt[s1, s2]
    )

    lopdt = Model(m, "lopdt", equations=[deffreqlop, dtlimit, defobjdtlop])

    freq.lo[s1, s2].where[rt[s1, s2]] = math.max(
        lfr[s1, s2], math.ceil(load[s1, s2] / maxtcap)
    )
    freq.up[s1, s2].where[rt[s1, s2]] = freq.lo[s1, s2]
    dt.up[s1, s2].where[od[s1, s2]] = od[s1, s2]

    m.solve(lopdt, problem="mip", sense="max", objective_variable=obj)

    solrep = Parameter(m, "solrep")
    solsum = Parameter(m, "solsum")

    solrep["DT", ll, "freq"] = phi.l[ll]
    solrep["DT", ll, "cars"].where[phi.l[ll]] = mincars + Card(ac) - 1

    xcost = Parameter(m, "xcost", domain=[root, s, lf])
    ycost = Parameter(m, "ycost", domain=[root, s, lf])
    length = Parameter(m, "len", domain=[s, s])
    sigma = Parameter(m, "sigma", domain=[s, s])

    length[ll] = Sum(l[ll, s1, s2], rt[s1, s2])
    sigma[ll] = (
        length[ll]
        + Sum(s.where[lr[ll, s, "1"]], tt[s])
        + Sum(s.where[rp[ll, s] == lastrp[ll]], tt[s])
    ) / 60

    xcost[ll, lf] = (
        Ord(lf) * length[ll] * (trm + mincars * crm)
        + mincars * math.ceil(sigma[ll] * Ord(lf)) * cfx
    )
    ycost[ll, lf] = (
        Ord(lf) * length[ll] * crm + math.ceil(sigma[ll] * Ord(lf)) * cfx
    )

    x = Variable(m, "x", type="binary", domain=[s1, s2, lf])
    y = Variable(m, "y", type="integer", domain=[s1, s2, lf])

    deffreqilp = Equation(m, "deffreqilp", domain=[s, s], type="eq")
    defloadilp = Equation(m, "defloadilp", domain=[s, s], type="leq")
    oneilp = Equation(m, "oneilp", domain=[s, s], type="leq")
    couplexy = Equation(m, "couplexy", domain=[s, s, lf], type="leq")
    defobjilp = Equation(m, "defobjilp", type="eq")

    deffreqilp[s1, s2].where[rt[s1, s2]] = freq[s1, s2] == Sum(
        (l[ll, s1, s2], lf), Ord(lf) * x[ll, lf]
    )

    defloadilp[s1, s2].where[rt[s1, s2]] = math.ceil(
        load[s1, s2] / ccap
    ) <= Sum((l[ll, s1, s2], lf), Ord(lf) * (mincars * x[ll, lf] + y[ll, lf]))

    oneilp[ll] = Sum(lf, x[ll, lf]) <= 1

    couplexy[ll, lf] = y[ll, lf] <= y.up[ll, lf] * x[ll, lf]

    defobjilp.definition = obj == Sum(
        (ll, lf), xcost[ll, lf] * x[ll, lf] + ycost[ll, lf] * y[ll, lf]
    )

    ilp = Model(
        m,
        "ilp",
        equations=[defobjilp, deffreqilp, defloadilp, oneilp, couplexy],
    )

    y.up[ll, lf] = Card(ac) - 1
    freq.up[s1, s2].where[rt[s1, s2]] = 100

    m.addOptions({"optCr": 0, "resLim": 100})

    m.solve(ilp, problem="mip", sense="min", objective_variable=obj)

    # solrep["ILP", ll, "freq"] = Sum(lf.where[x.l[ll, lf]], Ord(lf))
    # solrep["ILP", ll, "cars"] = Sum(
    #     lf.where[x.l[ll, lf]], mincars + y.l[ll, lf]
    # )
    # solsum["ILP", "cost"] = obj.l

    # cap = Parameter(m, "cap", domain=[s, s])
    # sol = Set(m, "sol", domain=[s, s])

    # dtr = Variable(m, "dtr", domain=[s, s, s, s], type="positive")

    # dtllimit = Equation(m, "dtllimit", domain=[s, s1, s2, s3], type="leq")
    # sumbound = Equation(m, "sumbound", domain=[s, s], type="eq")

    # dtllimit[l[sol, s, s1]] = (
    #     Sum(
    #         Domain(s2, s3).where[od[s2, s3]]
    #         & rp[sol, s2]
    #         & rp[sol, s3]
    #         & (
    #             math.min(rp[sol, s], rp[sol, s1])
    #             >= rp[sol, s2] & math.max(rp[sol, s], rp[sol, s1])
    #             <= rp[sol, s3] | math.min(rp[sol, s], rp[sol, s1])
    #             >= rp[sol, s3] & math.max(rp[sol, s], rp[sol, s1])
    #             <= rp[sol, s2]
    #         ),
    #         dtr[sol, s2, s3],
    #     )
    #     <= cap[sol]
    # )

    # sumbound[s2, s3].where[od[s2, s3]] = (
    #     Sum(sol.where[rp[sol, s2] & rp[sol, s3]], dtr[sol, s2, s3])
    #     == dt[s2, s3]
    # )

    # evaldt = Model(m, "evaldt", equations=[dtllimit, sumbound, defobjdtlop])

    # sol[ll] = solrep["DT", ll, "freq"]
    # cap[sol] = solrep["DT", sol, "freq"] * solrep["DT", sol, "cars"] * ccap

    # m.solve(evaldt, problem="lp", sense="max", objective_variable=obj)

    # solsum["DT", "dtrav"] = obj.l
    # solsum["DT", "cost"] = Sum(
    #     sol,
    #     solrep["DT", sol, "freq"] * length[sol] * trm
    #     + (
    #         solrep["DT", sol, "freq"] * length[sol] * crm
    #         + math.ceil(sigma[sol] * solrep["DT", sol, "freq"]) * cfx
    #     )
    #     * solrep["DT", sol, "cars"],
    # )

    # sol[ll] = solrep["ILP", ll, "freq"]
    # cap[sol] = solrep["DT", sol, "freq"] * solrep["DT", sol, "cars"] * ccap

    # m.solve(evaldt, problem="lp", sense="max", objective_variable=obj)
    # solsum["ILP", "dtrav"] = obj.l

    # print(solrep.records)
    # print(solsum.records)


if __name__ == "__main__":
    main()
