"""
*** Optimal power flow for a Five-bus system

For more details please refer to Chapter 6 (Gcode6.3), of the following book:
Soroudi, Alireza. Power System Optimization Modeling in GAMS. Springer, 2017.
--------------------------------------------------------------------------------
Model type: LP
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
from gamspy import Equation
from gamspy import Model
from gamspy import Parameter
from gamspy import Set
from gamspy import Sum
from gamspy import Variable


def reformat_df(dataframe):
    return dataframe.reset_index().melt(
        id_vars="index", var_name="Category", value_name="Value"
    )


def data_records():
    # GenD records table
    cols = ["b", "Pmin", "Pmax"]
    inds = [f"g{gg}" for gg in range(1, 6)]
    data = [
        [14, 0, 40],
        [15, 0, 170],
        [30, 0, 520],
        [40, 0, 200],
        [10, 0, 600],
    ]
    GenD_recs = reformat_df(pd.DataFrame(data, columns=cols, index=inds))

    # BD records table
    cols = ["Pd"]
    inds = ["2", "3", "4"]
    data = [[300], [300], [400]]
    BD_recs = reformat_df(pd.DataFrame(data, columns=cols, index=inds))

    # Branch records table
    cols = ["x", "limit"]
    inds = [
        ("1", "2"),
        ("1", "4"),
        ("1", "5"),
        ("2", "3"),
        ("3", "4"),
        ("4", "5"),
    ]
    data = [
        [0.0281, 400],
        [0.0304, 400],
        [0.0064, 400],
        [0.0108, 400],
        [0.0297, 400],
        [0.0297, 240],
    ]
    inds = pd.MultiIndex.from_tuples(inds, names=["Index1", "Index2"])
    B_recs = pd.DataFrame(data, columns=cols, index=inds)
    B_recs.reset_index(inplace=True)
    B_recs = B_recs.melt(
        id_vars=["Index1", "Index2"], value_vars=["x", "limit"]
    )

    return (
        GenD_recs,
        BD_recs,
        B_recs,
    )


def main():
    m = Container()

    # SETS #
    bus = Set(m, name="bus", records=[str(buses) for buses in range(1, 6)])
    slack = Set(m, name="slack", domain=bus, records=["1"])
    Gen = Set(m, name="Gen", records=[f"g{gg}" for gg in range(1, 6)])
    conex = Set(
        m,
        name="conex",
        domain=[bus, bus],
        records=[
            ("1", "2"),
            ("2", "3"),
            ("3", "4"),
            ("4", "1"),
            ("4", "5"),
            ("5", "1"),
        ],
        description="bus connectivity matrix",
    )
    GBconect = Set(
        m,
        name="GBconect",
        domain=[bus, Gen],
        records=[
            ("1", "g1"),
            ("1", "g2"),
            ("3", "g3"),
            ("4", "g4"),
            ("5", "g5"),
        ],
        description="connectivity index of each generating unit to each bus",
    )

    # ALIASES #
    node = Alias(m, name="node", alias_with=bus)

    conex[bus, node].where[conex[node, bus]] = 1

    # SCALARS #
    Sbase = Parameter(m, name="Sbase", records=100)

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

    branch[bus, node, "x"].where[branch[bus, node, "x"] == 0] = branch[
        node, bus, "x"
    ]
    branch[bus, node, "Limit"].where[branch[bus, node, "Limit"] == 0] = branch[
        node, bus, "Limit"
    ]
    branch[bus, node, "bij"].where[conex[bus, node]] = (
        1 / branch[bus, node, "x"]
    )

    # VARIABLES #
    OF = Variable(m, name="OF")
    Pij = Variable(m, name="Pij", domain=[bus, node])
    Pg = Variable(m, name="Pg", domain=[Gen])
    delta = Variable(m, name="delta", domain=[bus])

    # EQUATIONS #
    const1 = Equation(m, name="const1", type="regular", domain=[bus, node])
    const2 = Equation(m, name="const2", type="regular", domain=[bus])
    const3 = Equation(m, name="const3", type="regular")

    const1[bus, node].where[conex[bus, node]] = Pij[bus, node] == branch[
        bus, node, "bij"
    ] * (delta[bus] - delta[node])

    const2[bus] = Sum(Gen.where[GBconect[bus, Gen]], Pg[Gen]) - BusData[
        bus, "pd"
    ] / Sbase == Sum(node.where[conex[node, bus]], Pij[bus, node])

    const3.expr = OF >= Sum(Gen, Pg[Gen] * GenData[Gen, "b"] * Sbase)

    loadflow = Model(
        m,
        name="loadflow",
        equations=[const1, const2, const3],
        problem="lp",
        sense="min",
        objective=OF,
    )

    Pg.lo[Gen] = GenData[Gen, "Pmin"] / Sbase
    Pg.up[Gen] = GenData[Gen, "Pmax"] / Sbase
    delta.up[bus] = np.pi
    delta.lo[bus] = -np.pi
    delta.fx[slack] = 0
    Pij.up[bus, node].where[conex[bus, node]] = (
        1 * branch[bus, node, "Limit"] / Sbase
    )
    Pij.lo[bus, node].where[conex[bus, node]] = (
        -1 * branch[bus, node, "Limit"] / Sbase
    )

    loadflow.solve()

    #  REPORTING PARAMETERS
    report = Parameter(m, name="report", domain=[bus, "*"])
    Congestioncost = Parameter(m, name="Congestioncost")
    report[bus, "Gen(MW)"] = (
        Sum(Gen.where[GBconect[bus, Gen]], Pg.l[Gen]) * Sbase
    )
    report[bus, "Angle"] = delta.l[bus]
    report[bus, "load(MW)"] = BusData[bus, "pd"]
    report[bus, "LMP($/MWh)"] = const2.m[bus] / Sbase
    Congestioncost.assign = (
        Sum([bus, node], Pij.l[bus, node] * (-const2.m[bus] + const2.m[node]))
        / 2
    )

    print("report:  \n", report.pivot().round(4), "\n")
    print("Pij:  \n", Pij.pivot().round(4), "\n")
    print("Congestioncost:  ", Congestioncost.toValue())
    print("Objective Function Value:  ", round(OF.toValue(), 4))


if __name__ == "__main__":
    main()
