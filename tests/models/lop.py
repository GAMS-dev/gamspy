from pathlib import Path
from gamspy import (
    Alias,
    Set,
    Card,
    Sum,
    Model,
    Container,
    Number,
)
import gamspy.math as math


def main():
    m = Container(load_from=str(Path(__file__).parent.absolute()) + "/lop.gdx")

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

    l[root, s, s1, s2] = Number(0)
    lr[root, s, s1, r] = Number(0)

    for elem in root:
        from_[elem[0]] = Number(1)
        unvisit[s] = Number(1)
        visit[s] = Number(0)

        for idx, r_elem in enumerate(r):
            if idx > 0 and len(unvisit):
                unvisit[from_] = Number(0)
                visit[from_] = Number(1)
                to[unvisit] = Sum(tree[elem[0], from_, unvisit], Number(1))

                for f_elem in from_:
                    k[s2, s3].where[l[elem[0], f_elem[0], s2, s3]] = Number(1)
                    v[s2, r1].where[lr[elem[0], f_elem[0], s2, r1]] = Number(1)
                    v[f_elem[0], "1"].where[Card(k) == 0] = Number(1)

                    l[elem[0], to, k].where[
                        tree[elem[0], f_elem[0], to]
                    ] = Number(1)
                    lr[elem[0], to, v].where[
                        tree[elem[0], f_elem[0], to]
                    ] = Number(1)

                    l[elem[0], to, f_elem[0], to].where[
                        tree[elem[0], f_elem[0], to]
                    ] = Number(1)
                    lr[elem[0], to, to, r_elem[0]].where[
                        tree[elem[0], f_elem[0], to]
                    ] = Number(1)

                    k[s2, s3] = Number(0)
                    v[s2, r1] = Number(0)

                from_[s] = Number(0)
                from_[to] = Number(1)
                to[s] = Number(0)

        from_[s] = Number(0)


if __name__ == "__main__":
    main()
