"""
*** Transmission Expansion Planning

For more details please refer to Chapter 9 (Gcode9.1), of the following book:
Soroudi, Alireza. Power System Optimization Modeling in GAMS. Springer, 2017.
--------------------------------------------------------------------------------
Model type: MIP
--------------------------------------------------------------------------------
Contributed by
Dr. Alireza Soroudi
IEEE Senior Member
email: alireza.soroudi@gmail.com
We do request that publications derived from the use of the developed GAMS code
explicitly acknowledge that fact by citing
Soroudi, Alireza. Power System Optimization Modeling in GAMS. Springer, 2017.
DOI: doi.org/10.1007/978-3-319-62350-4
"""
import numpy as np
import pandas as pd

from gamspy import Alias
from gamspy import Container
from gamspy import Domain
from gamspy import Equation
from gamspy import Model
from gamspy import Ord
from gamspy import Parameter
from gamspy import Set
from gamspy import Smax
from gamspy import Sum
from gamspy import Variable


def reformat_df(dataframe):
    return dataframe.reset_index().melt(
        id_vars="index", var_name="Category", value_name="Value"
    )


def data_records():
    # GenD records table
    cols = ["b", "Pmin", "Pmax"]
    inds = [f"g{g}" for g in range(1, 4)]
    data = [[20, 0, 400], [30, 0, 400], [10, 0, 600]]
    GenD_recs = reformat_df(pd.DataFrame(data, columns=cols, index=inds))

    # BD records table
    cols = ["Pd"]
    inds = ["1", "2", "3", "4", "5"]
    data = [[80], [240], [40], [160], [240]]
    BD_recs = reformat_df(pd.DataFrame(data, columns=cols, index=inds))

    # Branch records table
    cols = ["X", "LIMIT", "Cost", "stat"]
    inds = [
        ("1", "2"),
        ("1", "4"),
        ("1", "5"),
        ("2", "3"),
        ("2", "4"),
        ("2", "6"),
        ("3", "5"),
        ("4", "6"),
    ]
    data = [
        [0.4, 100, 40, 1],
        [0.6, 80, 60, 1],
        [0.2, 100, 20, 1],
        [0.2, 100, 20, 1],
        [0.4, 100, 40, 1],
        [0.3, 100, 30, 0],
        [0.2, 100, 20, 1],
        [0.3, 100, 30, 0],
    ]
    inds = pd.MultiIndex.from_tuples(inds, names=["Index1", "Index2"])
    B_recs = pd.DataFrame(data, columns=cols, index=inds)
    B_recs.reset_index(inplace=True)
    B_recs = B_recs.melt(
        id_vars=["Index1", "Index2"], value_vars=["X", "LIMIT", "Cost", "stat"]
    )

    return (
        GenD_recs,
        BD_recs,
        B_recs,
    )


def main():
    m = Container()

    # SETS #
    bus = Set(m, name="bus", records=[str(buses) for buses in range(1, 7)])
    slack = Set(m, name="slack", domain=bus, records=["1"])
    Gen = Set(m, name="Gen", records=[f"g{g}" for g in range(1, 4)])
    k = Set(m, name="k", records=[f"k{k}" for k in range(1, 5)])
    GBconect = Set(
        m,
        name="GBconect",
        domain=[bus, Gen],
        records=[("1", "g1"), ("3", "g2"), ("6", "g3")],
        description="connectivity index of each generating unit to each bus",
    )
    conex = Set(
        m,
        name="conex",
        domain=[bus, bus],
        description="bus connectivity matrix",
    )

    # ALIAS #
    node = Alias(m, name="node", alias_with=bus)

    # SCALARS #
    Sbase = Parameter(m, name="Sbase", records=100)
    M = Parameter(m, name="M")

    # PARAMETERS #
    GenData = Parameter(
        m,
        name="GenData",
        domain=[Gen, "*"],
        records=data_records()[0],
        description="generating units characteristics",
    )
    BusData = Parameter(
        m,
        name="BusData",
        domain=[bus, "*"],
        records=data_records()[1],
        description="demands of each bus in MW",
    )
    branch = Parameter(
        m,
        name="branch",
        domain=[bus, node, "*"],
        records=data_records()[2],
        description="network technical characteristics",
    )

    conex[bus, node].where[branch[bus, node, "x"]] = True
    conex[bus, node].where[conex[node, bus]] = True

    branch[bus, node, "x"].where[branch[node, bus, "x"]] = branch[
        node, bus, "x"
    ]
    branch[bus, node, "cost"].where[branch[node, bus, "cost"]] = branch[
        node, bus, "cost"
    ]
    branch[bus, node, "stat"].where[branch[node, bus, "stat"]] = branch[
        node, bus, "stat"
    ]
    branch[bus, node, "Limit"].where[branch[bus, node, "Limit"] == 0] = branch[
        node, bus, "Limit"
    ]
    branch[bus, node, "bij"].where[conex[bus, node]] = (
        1 / branch[bus, node, "x"]
    )
    M.assign = Smax(
        Domain(bus, node).where[conex[bus, node]],
        branch[bus, node, "bij"] * 3.14 * 2,
    )

    # VARIABLES #
    OF = Variable(m, name="OF", type="free")
    Pij = Variable(m, name="Pij", type="free", domain=[bus, node, k])
    Pg = Variable(m, name="Pg", type="free", domain=[Gen])
    delta = Variable(m, name="delta", type="free", domain=[bus])
    LS = Variable(m, name="LS", type="free", domain=[bus])
    alpha = Variable(m, name="alpha", type="binary", domain=[bus, node, k])

    alpha.l[bus, node, k] = 1
    alpha.fx[bus, node, k].where[
        (conex[bus, node]) & (Ord(k) == 1) & (branch[node, bus, "stat"])
    ] = 1

    # EQUATIONS #
    const1A = Equation(
        m, name="const1A", type="regular", domain=[bus, node, k]
    )
    const1B = Equation(
        m, name="const1B", type="regular", domain=[bus, node, k]
    )
    const1C = Equation(
        m, name="const1C", type="regular", domain=[bus, node, k]
    )
    const1D = Equation(
        m, name="const1D", type="regular", domain=[bus, node, k]
    )
    const1E = Equation(
        m, name="const1E", type="regular", domain=[bus, node, k]
    )
    const2 = Equation(m, name="const2", type="regular", domain=[bus])
    const3 = Equation(m, name="const3", type="regular")

    const1A[bus, node, k].where[conex[node, bus]] = Pij[bus, node, k] - branch[
        bus, node, "bij"
    ] * (delta[bus] - delta[node]) <= M * (1 - alpha[bus, node, k])

    const1B[bus, node, k].where[conex[node, bus]] = Pij[bus, node, k] - branch[
        bus, node, "bij"
    ] * (delta[bus] - delta[node]) >= -M * (1 - alpha[bus, node, k])

    const1C[bus, node, k].where[conex[node, bus]] = (
        Pij[bus, node, k]
        <= alpha[bus, node, k] * branch[bus, node, "Limit"] / Sbase
    )

    const1D[bus, node, k].where[conex[node, bus]] = (
        Pij[bus, node, k]
        >= -alpha[bus, node, k] * branch[bus, node, "Limit"] / Sbase
    )

    const1E[bus, node, k].where[conex[node, bus]] = (
        alpha[bus, node, k] == alpha[node, bus, k]
    )

    const2[bus] = LS[bus] + Sum(
        Gen.where[GBconect[bus, Gen]], Pg[Gen]
    ) - BusData[bus, "pd"] / Sbase == Sum(
        Domain(k, node).where[conex[node, bus]], Pij[bus, node, k]
    )

    const3.expr = OF >= 10 * 8760 * (
        Sum(Gen, Pg[Gen] * GenData[Gen, "b"] * Sbase)
        + 100000 * Sum(bus, LS[bus])
    ) + 1e6 * Sum(
        Domain(bus, node, k).where[conex[node, bus]],
        0.5
        * branch[bus, node, "cost"]
        * alpha[bus, node, k].where[
            (Ord(k) > 1) | (branch[node, bus, "stat"] == 0)
        ],
    )

    loadflow = Model(
        m,
        name="loadflow",
        equations=m.getEquations(),
        problem="mip",
        sense="min",
        objective=OF,
    )

    LS.up[bus] = BusData[bus, "pd"] / Sbase
    LS.lo[bus] = 0
    Pg.lo[Gen] = GenData[Gen, "Pmin"] / Sbase
    Pg.up[Gen] = GenData[Gen, "Pmax"] / Sbase

    delta.up[bus] = np.pi / 3
    delta.lo[bus] = -np.pi / 3
    delta.fx[slack] = 0
    Pij.up[bus, node, k].where[conex[bus, node]] = (
        1 * branch[bus, node, "Limit"] / Sbase
    )
    Pij.lo[bus, node, k].where[conex[bus, node]] = (
        -1 * branch[bus, node, "Limit"] / Sbase
    )

    m.addOptions({"optCr": 0, "mip": "HIGHS"})
    loadflow.solve()

    print("Objective Function Value:  ", round(OF.toValue(), 3))


if __name__ == "__main__":
    main()
