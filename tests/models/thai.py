from pathlib import Path
from gamspy import Sum, Domain
from gamspy import Model, Container


def main():
    m = Container(
        load_from=str(Path(__file__).parent.absolute()) + "/thai.gdx"
    )

    # Sets
    i, j, k, sc, vc = m.getSymbols(["i", "j", "k", "sc", "vc"])

    # Parameters
    (d, shipcap, n, a, w1, w2, w3) = m.getSymbols(
        ["d", "shipcap", "n", "a", "w1", "w2", "w3"]
    )

    # Variables
    z, y, obj = m.getSymbols(["z", "y", "obj"])

    # Equations
    (
        objdef,
        demand,
        voycap,
        shiplim,
    ) = m.getSymbols(
        [
            "objdef",
            "demand",
            "voycap",
            "shiplim",
        ]
    )

    objdef.definition = obj == w1 * Sum(
        Domain(j, k).where[vc[j, k]], z[j, k]
    ) + w2 * Sum(
        Domain(j, k).where[vc[j, k]], a[j, "dist"] * z[j, k]
    ) + w3 * Sum(
        Domain(j, k, i).where[a[j, i].where[vc[j, k]]],
        a[j, "dist"] * y[j, k, i],
    )

    demand[i] = (
        Sum(Domain(j, k).where[a[j, i].where[vc[j, k]]], y[j, k, i]) >= d[i]
    )
    voycap[j, k].where[vc[j, k]] = (
        Sum(i.where[a[j, i]], y[j, k, i]) <= shipcap[k] * z[j, k]
    )
    shiplim[k] = Sum(j.where[vc[j, k]], z[j, k]) <= n[k]

    thainavy = Model(m, name="thainavy", equations="all")
    z.up[j, k].where[vc[j, k]] = n[k]
    m.solve(thainavy, problem="MIP", sense="min", objective_variable=obj)


if __name__ == "__main__":
    main()
