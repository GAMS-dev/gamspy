"""
## LICENSETYPE: Demo
## MODELTYPE: MIP
## KEYWORDS: mixed integer linear programming, absolute value, facility layout, disjunctive constraints, formulations


Single Row Facility Layout Problem (SRFLP)

Machines of different lengths are placed side by side along a straight
corridor. Material flows between the machines, and the cost of a flow is
proportional to the distance between the centers of the machines. The
problem finds the positions that minimize the total transport cost:

  min sum((i,j), flow(i,j) * |x(i) - x(j)|)

The machines must not overlap, i.e. the distance between two centers must
be at least half of the sum of their lengths:

  |x(i) - x(j)| >= (length(i) + length(j)) / 2

The non-overlap constraint makes the problem non-convex, because it
requires one machine to be either left or right of the other.

gamspy.formulations.absolute makes the distance equal to the absolute value
by selecting the sign of the difference with a binary variable, so the
machines do not overlap.

Simmons, D M, One-Dimensional Space Allocation: An Ordering Algorithm.
Operations Research 17, 5 (1969), 812-826.
"""

from __future__ import annotations

import pandas as pd

import gamspy.formulations as formulations
from gamspy import (
    Alias,
    Container,
    Equation,
    Model,
    Ord,
    Parameter,
    Sense,
    Set,
    Sum,
    Variable,
)


def report(i: Set, x: Variable, length: Parameter) -> pd.DataFrame:
    layout = pd.DataFrame(
        {"center": x.toDense(), "length": length.toDense()}, index=i.toList()
    )
    layout["start"] = layout["center"] - layout["length"] / 2
    layout["end"] = layout["center"] + layout["length"] / 2
    return layout.sort_values("center").round(3)


def overlap(layout: pd.DataFrame) -> float:
    # largest overlap of two neighboring machines
    return (layout["end"].iloc[:-1].values - layout["start"].iloc[1:].values).max()


def main():
    m = Container()

    # Sets
    i = Set(
        m,
        name="i",
        records=["saw", "lathe", "drill", "mill", "grinder", "paint"],
        description="machines",
    )
    j = Alias(m, name="j", alias_with=i)
    ij = Set(m, name="ij", domain=[i, j], description="pairs of machines")
    ij[i, j] = Ord(i) < Ord(j)

    # Parameters
    length = Parameter(
        m,
        name="length",
        domain=i,
        records=[
            ("saw", 3),
            ("lathe", 5),
            ("drill", 2),
            ("mill", 6),
            ("grinder", 4),
            ("paint", 7),
        ],
        description="length of machine i in meters",
    )
    flow = Parameter(
        m,
        name="flow",
        domain=[i, j],
        records=[
            ("saw", "lathe", 10),
            ("saw", "drill", 4),
            ("saw", "mill", 6),
            ("lathe", "drill", 8),
            ("lathe", "grinder", 7),
            ("drill", "mill", 9),
            ("drill", "paint", 2),
            ("mill", "grinder", 5),
            ("mill", "paint", 3),
            ("grinder", "paint", 12),
        ],
        description="transports between machine i and j per day",
    )

    # Variables
    x = Variable(
        m, name="x", domain=i, description="position of the center of machine i"
    )
    x.lo[i] = length[i] / 2
    x.up[i] = Sum(j, length[j]) - length[i] / 2

    delta = Variable(
        m, name="delta", domain=[i, j], description="difference of the centers"
    )
    defdelta = Equation(
        m,
        name="defdelta",
        domain=[i, j],
        description="define the difference of the centers",
    )
    defdelta[ij[i, j]] = delta[i, j] == x[i] - x[j]

    # dist[i, j] == abs(x[i] - x[j])
    dist, eqs = formulations.absolute(delta[ij])

    no_overlap = Equation(
        m, name="no_overlap", domain=[i, j], description="machines do not overlap"
    )
    no_overlap[ij[i, j]] = dist[i, j] >= (length[i] + length[j]) / 2

    srflp = Model(
        m,
        name="srflp",
        equations=[defdelta, no_overlap, *eqs],
        problem="mip",
        sense=Sense.MIN,
        objective=Sum(ij[i, j], (flow[i, j] + flow[j, i]) * dist[i, j]),
    )
    srflp.solve()

    layout = report(i, x, length)
    print("Objective Function Value: ", round(srflp.objective_value, 3))
    print(layout)

    # The machines do not overlap
    if overlap(layout) > 1e-6:
        raise Exception("machines overlap")


if __name__ == "__main__":
    main()
