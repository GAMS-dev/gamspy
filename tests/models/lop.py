from pathlib import Path
from gamspy import Container


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


if __name__ == "__main__":
    main()
