from pathlib import Path
from gamspy import (
    Set,
    Card,
    Sum,
    Model,
    Container,
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


if __name__ == "__main__":
    main()
