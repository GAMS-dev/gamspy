"""
## GAMSSOURCE: https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_spbenders1.html
## LICENSETYPE: Demo
## MODELTYPE: LP
## KEYWORDS: linear programming, stochastic Benders algorithm, transportation problem


Stochastic Benders - Sequential GAMS Loop (SPBENDERS1)

This example demonstrates a stochastic Benders implementation for the
simple transport example.

This is the first example of a sequence of stochastic Benders
implementations using various methods to solve the master and
subproblem.

This first example implements the stochastic Benders algorithm using
sequential solves of the master and subproblems in a GAMS loop.
"""

from __future__ import annotations

import pandas as pd

import gamspy.math as gams_math
from gamspy import (
    Container,
    Equation,
    Model,
    Parameter,
    Sense,
    Set,
    Sum,
    Variable,
)


def main():
    m = Container()

    # Prepare data
    cost = pd.DataFrame(
        [
            ["f1", "d1", 2.49],
            ["f1", "d2", 5.21],
            ["f1", "d3", 3.76],
            ["f1", "d4", 4.85],
            ["f1", "d5", 2.07],
            ["f2", "d1", 1.46],
            ["f2", "d2", 2.54],
            ["f2", "d3", 1.83],
            ["f2", "d4", 1.86],
            ["f2", "d5", 4.76],
            ["f3", "d1", 3.26],
            ["f3", "d2", 3.08],
            ["f3", "d3", 2.60],
            ["f3", "d4", 3.76],
            ["f3", "d5", 4.45],
        ]
    )

    scenarios = pd.DataFrame(
        [
            ["lo", "d1", 150],
            ["lo", "d2", 100],
            ["lo", "d3", 250],
            ["lo", "d4", 300],
            ["lo", "d5", 600],
            ["lo", "prob", 0.25],
            ["mid", "d1", 160],
            ["mid", "d2", 120],
            ["mid", "d3", 270],
            ["mid", "d4", 325],
            ["mid", "d5", 700],
            ["mid", "prob", 0.50],
            ["hi", "d1", 170],
            ["hi", "d2", 135],
            ["hi", "d3", 300],
            ["hi", "d4", 350],
            ["hi", "d5", 800],
            ["hi", "prob", 0.25],
        ]
    )

    cut_coefficients = pd.DataFrame(
        [
            [idx, f"d{center}", 0]
            for idx in range(1, 26)
            for center in range(1, 6)
        ]
    )

    # Set
    i = Set(m, name="i", records=["f1", "f2", "f3"], description="factories")
    j = Set(
        m,
        name="j",
        records=["d1", "d2", "d3", "d4", "d5"],
        description="distribution centers",
    )
    s = Set(m, name="s", records=["lo", "mid", "hi"], description="scenarios")

    # Data
    capacity = Parameter(
        m,
        name="capacity",
        domain=i,
        records=pd.DataFrame([["f1", 500], ["f2", 450], ["f3", 650]]),
        description="unit capacity at factories",
    )
    demand = Parameter(
        m,
        name="demand",
        domain=j,
        records=pd.DataFrame(
            [["d1", 160], ["d2", 120], ["d3", 270], ["d4", 325], ["d5", 700]]
        ),
        description="unit demand at distribution centers",
    )
    prodcost = Parameter(
        m, name="prodcost", records=14, description="unit production cost"
    )
    price = Parameter(m, name="price", records=24, description="sales price")
    wastecost = Parameter(
        m,
        name="wastecost",
        records=4,
        description="cost of removal of overstocked products",
    )
    transcost = Parameter(
        m,
        name="transcost",
        domain=[i, j],
        records=cost,
        description="unit transportation cost",
    )
    ScenarioData = Parameter(
        m,
        name="scenariodata",
        domain=[s, "*"],
        records=scenarios,
        description="possible outcomes for demand plus probabilities",
    )

    # Set
    iter = Set(
        m,
        name="iter",
        records=[f"{idx}" for idx in range(1, 26)],
        description="max Benders iterations",
    )
    itActive = Set(
        m, name="itActive", domain=iter, description="active Benders cuts"
    )

    # Parameter
    cutconst = Parameter(
        m,
        name="cutconst",
        domain=iter,
        records=pd.DataFrame([[f"{idx}", 0] for idx in range(1, 26)]),
        description="constants in optimality cuts",
    )
    cutcoeff = Parameter(
        m,
        name="cutcoeff",
        domain=[iter, j],
        records=cut_coefficients,
        description="coefficients in optimality cuts",
    )

    # Variable
    ship = Variable(
        m, name="ship", domain=[i, j], type="Positive", description="shipments"
    )
    product = Variable(m, name="product", domain=i, description="production")
    received = Variable(
        m, name="received", domain=j, description="quantity sent to market"
    )
    zmaster = Variable(
        m, name="zmaster", description="objective variable of master problem"
    )
    theta = Variable(m, name="theta", description="future profit")

    # Equation
    masterobj = Equation(
        m, name="masterobj", description="master objective function"
    )
    production = Equation(
        m,
        name="production",
        domain=i,
        description="calculate production in each factory",
    )
    receive = Equation(
        m,
        name="receive",
        domain=j,
        description="calculate quantity to be send to markets",
    )
    optcut = Equation(
        m, name="optcut", domain=iter, description="Benders optimality cuts"
    )

    masterobj[...] = zmaster == theta - Sum(
        (i, j), transcost[i, j] * ship[i, j]
    ) - Sum(i, prodcost * product[i])
    receive[j] = received[j] == Sum(i, ship[i, j])
    production[i] = product[i] == Sum(j, ship[i, j])
    optcut[itActive] = theta <= cutconst[itActive] + Sum(
        j, cutcoeff[itActive, j] * received[j]
    )
    product.up[i] = capacity[i]

    masterproblem = Model(
        m,
        name="masterproblem",
        equations=m.getEquations(),
        problem="LP",
        sense=Sense.MAX,
        objective=zmaster,
    )

    # Variable
    sales = Variable(
        m,
        name="sales",
        domain=j,
        type="Positive",
        description="sales (actually sold)",
    )
    waste = Variable(
        m,
        name="waste",
        domain=j,
        type="Positive",
        description="overstocked products",
    )
    zsub = Variable(
        m, name="zsub", description="objective variable of sub problem"
    )

    # Equation
    subobj = Equation(
        m, name="subobj", description="subproblem objective function"
    )
    selling = Equation(
        m, name="selling", domain=j, description="part of received is sold"
    )
    market = Equation(
        m, name="market", domain=j, description="upperbound on sales"
    )

    subobj[...] = zsub == Sum(j, price * sales[j]) - Sum(
        j, wastecost * waste[j]
    )
    selling[j] = sales[j] + waste[j] == received.l[j]
    market[j] = sales[j] <= demand[j]

    subproblem = Model(
        m,
        name="subproblem",
        equations=[subobj, selling, market],
        problem="LP",
        sense=Sense.MAX,
        objective=zsub,
    )

    # Scalar
    rgap = Parameter(m, name="rgap", records=0, description="relative gap")
    lowerBound = Parameter(
        m,
        name="lowerBound",
        records=float("-inf"),
        description="global lower bound",
    )
    upperBound = Parameter(
        m,
        name="upperBound",
        records=float("inf"),
        description="global upper bound",
    )
    objMaster = Parameter(m, name="objMaster", records=0)
    objSub = Parameter(m, name="objSub", records=0)
    rtol = 0.001

    received.l[j] = 0

    for iteration, _ in iter.records.itertuples(index=False):
        objSub[...] = 0

        for scenario, _ in s.records.itertuples(index=False):
            demand[j] = ScenarioData[scenario, j]
            subproblem.solve()
            objSub[...] = objSub + ScenarioData[scenario, "prob"] * zsub.l
            cutconst[iteration] = cutconst[iteration] + ScenarioData[
                scenario, "prob"
            ] * Sum(j, market.m[j] * demand[j])
            cutcoeff[iteration, j] = (
                cutcoeff[iteration, j]
                + ScenarioData[scenario, "prob"] * selling.m[j]
            )

        itActive[iteration] = True

        if (
            lowerBound.records.values[0][0]
            < objMaster.records.values[0][0] + objSub.records.values[0][0]
        ):
            lowerBound[...] = objMaster + objSub

        rgap[...] = (upperBound - lowerBound) / (1 + gams_math.abs(upperBound))
        if rgap.records.values[0][0] < rtol:
            break

        masterproblem.solve()
        upperBound.setRecords(zmaster.records["level"])
        objMaster.setRecords(zmaster.records["level"] - theta.records["level"])

    print(
        "Master problem Objective Function Value: ",
        masterproblem.objective_value,
    )
    print("Subproblem Objective Function Value: ", subproblem.objective_value)


if __name__ == "__main__":
    main()
