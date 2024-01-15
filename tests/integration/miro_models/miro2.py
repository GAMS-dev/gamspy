"""
A Transportation Problem (TRNSPORT)

This problem finds a least cost shipping schedule that meets
requirements at markets and supplies at factories.


Dantzig, G B, Chapter 3.3. In Linear Programming and Extensions.
Princeton University Press, Princeton, New Jersey, 1963.

This formulation is described in detail in:
Rosenthal, R E, Chapter 2: A GAMS Tutorial. In GAMS: A User's Guide.
The Scientific Press, Redwood City, California, 1988.

The line numbers will not match those in the book because of these
comments.

Keywords: linear programming, transportation problem, scheduling
"""
from __future__ import annotations

import sys

from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Parameter
from gamspy import Sense
from gamspy import Set
from gamspy import Smax
from gamspy import Sum
from gamspy import Variable
from gamspy.math import Min


def main():
    m = Container(delayed_execution=True)

    # Prepare data
    distances = [
        ["seattle", "new-york", 2.5],
        ["seattle", "chicago", 1.7],
        ["seattle", "topeka", 1.8],
        ["san-diego", "new-york", 2.5],
        ["san-diego", "chicago", 1.8],
        ["san-diego", "topeka", 1.4],
    ]

    loc_data = {
        "i": [
            ["Seattle", "lat", 47.608013],
            ["Seattle", "lnG", -122.335167],
            ["San-Diego", "lat", 32.715736],
            ["San-Diego", "lnG", -117.161087],
        ],
        "j": [
            ["New-York", "lat", 40.730610],
            ["New-York", "lnG", -73.935242],
            ["Chicago", "lat", 41.881832],
            ["Chicago", "lnG", -87.623177],
            ["Topeka", "lat", 39.056198],
            ["Topeka", "lnG", -95.695312],
        ],
    }

    capacities = [["seattle", 350], ["san-diego", 600]]
    demands = [["new-york", 325], ["chicago", 300], ["topeka", 275]]

    # Set
    i = Set(
        m,
        name="I",
        records=["seattle", "san-diego"],
        description="canning plants",
    )
    j = Set(
        m,
        name="j",
        records=["new-york", "chicago", "topeka"],
        description="markets",
    )
    loc_hdr = Set(
        m,
        name="loc_hdr",
        records=["lat", "lnG"],
        description="location data header",
    )
    model_type = Set(
        m,
        name="type",
        records=["lp"],
        is_singleton=True,
        is_miro_input=True,
        description="selected model type",
    )

    # Data
    a = Parameter(
        m,
        name="a",
        domain=[i],
        records=capacities,
        domain_forwarding=True,
        is_miro_input=True,
        description="capacity of plant i in cases",
    )
    b = Parameter(
        m,
        name="b",
        domain=[j],
        records=demands,
        domain_forwarding=True,
        is_miro_input=True,
        description="demand at market j in cases",
    )
    d = Parameter(
        m,
        name="d",
        domain=[i, j],
        records=distances,
        is_miro_input=True,
        description="distance in thousands of miles",
    )
    f = Parameter(
        m,
        name="f",
        records=90,
        is_miro_input=True,
        description="freight in dollars per case per thousand miles",
    )
    min_s = Parameter(
        m,
        name="mins",
        records=100,
        is_miro_input=True,
        description="minimum shipment (MIP- and MINLP-only)",
    )
    beta = Parameter(
        m,
        name="beta",
        records=0.95,
        is_miro_input=True,
        description="beta (MINLP-only)",
    )
    i_loc_data = Parameter(
        m,
        name="ilocdAta",
        records=loc_data["i"],
        domain=[i, loc_hdr],
        is_miro_input=True,
        is_miro_table=True,
        description="Plant location information",
    )
    j_loc_data = Parameter(
        m,
        name="jlocdata",
        records=loc_data["j"],
        domain=[j, loc_hdr],
        is_miro_input=True,
        is_miro_table=True,
        description="Market location information",
    )

    c = Parameter(
        m,
        name="c",
        domain=[i, j],
        description="transport cost in thousands of dollars per case",
    )
    c[i, j] = f * d[i, j] / 1000

    with open("miro.log", "w") as f:
        f.writelines(
            [
                "------------------------------------\n",
                "        Validating data\n",
                "------------------------------------\n",
            ]
        )
        if a.records.value.sum() < b.records.value.sum():
            f.writelines(["a:: Capacity insufficient to meet demand"])
        else:
            f.writelines(["OK"])

    # Variable
    x = Variable(
        m,
        name="x",
        domain=[i, j],
        type="Positive",
        description="shipment quantities in cases",
    )
    z = Variable(
        m,
        name="z",
        description="total transportation costs in thousands of dollars",
    )

    # Equation
    supply = Equation(
        m,
        name="supply",
        domain=[i],
        description="observe supply limit at plant i",
    )
    demand = Equation(
        m, name="demand", domain=[j], description="satisfy demand at market j"
    )
    cost = Equation(m, name="cost", description="define objective function")

    supply[i] = Sum(j, x[i, j]) <= a[i]
    demand[j] = Sum(i, x[i, j]) >= b[j]
    cost[...] = z == Sum((i, j), c[i, j] * x[i, j])

    transport_lp = Model(
        m,
        name="transport_lp",
        equations=[supply, demand, cost],
        problem="LP",
        sense=Sense.MIN,
        objective=z,
    )

    big_m = Parameter(m, name="bigm", description="big M")

    big_m[...] = Min(Smax(i, a[i]), Smax(j, b[j]))

    ship = Variable(
        m,
        name="ship",
        domain=[i, j],
        type="binary",
        description="1 if we ship from i to j, otherwise 0",
    )

    minship = Equation(
        m, name="minship", domain=[i, j], description="minimum shipment"
    )
    maxship = Equation(
        m, name="maxship", domain=[i, j], description="maimum shipment"
    )

    minship[i, j] = x[i, j] >= min_s * ship[i, j]
    maxship[i, j] = x[i, j] <= big_m * ship[i, j]

    transport_mip = Model(
        m,
        name="transport_mip",
        equations=[supply, demand, minship, maxship, cost],
        problem="MIP",
        sense=Sense.MIN,
        objective=z,
    )

    costnlp = Equation(
        m, name="costnlp", description="define non-linear objective function"
    )
    costnlp[...] = z == Sum([i, j], c[i, j] * x[i, j] ** beta)

    transport_nlp = Model(
        m,
        name="transport_nlp",
        equations=[supply, demand, minship, maxship, costnlp],
        problem="MIP",
        sense=Sense.MIN,
        objective=z,
    )

    # some starting point
    x.l[i, j] = 1

    schedule_hdr = Set(
        m,
        name="schedule_hdr",
        records=[
            "lngP",
            "latP",
            "lngM",
            "latM",
            "cap",
            "demand",
            "quantities",
        ],
        description="schedule header",
    )
    schedule = Parameter(
        m,
        name="schedule",
        domain=[i, j, schedule_hdr],
        is_miro_output=True,
        is_miro_table=True,
        description="shipment quantities in cases",
    )
    total_cost = Parameter(
        m,
        name="total_cost",
        is_miro_output=True,
        description="total transportation costs in thousands of dollars",
    )

    total_cost[...] = z.l
    schedule[i, j, "lngP"] = i_loc_data[i, "lnG"]
    schedule[i, j, "latP"] = i_loc_data[i, "lat"]
    schedule[i, j, "lngM"] = j_loc_data[j, "lnG"]
    schedule[i, j, "latM"] = j_loc_data[j, "lat"]
    schedule[i, j, "cap"] = a[i]
    schedule[i, j, "demand"] = b[j]
    schedule[i, j, "quantities"] = x.l[i, j]

    model_type_str = model_type.records.loc[0, "uni"]
    if model_type_str == "lp":
        transport_lp.solve(output=sys.stdout)
    elif model_type_str == "mip":
        transport_mip.solve(output=sys.stdout)
    elif model_type_str == "nlp":
        transport_nlp.solve(output=sys.stdout)

    if transport_lp.status.value > 2 and transport_lp.solver_status != 8:
        raise Exception("No feasible solution found")


if __name__ == "__main__":
    main()
