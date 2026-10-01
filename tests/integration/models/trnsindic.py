"""
## GAMSSOURCE: https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_trnsindic.html
## LICENSETYPE: Demo
## MODELTYPE: MIP
## KEYWORDS: mixed integer linear programming, transportation problem, indicator constraints, scheduling


Fixed Charge Transportation Problem with Indicator Constraints (TRNSINDIC)

This problem finds a least cost shipping schedule that meets
requirements at markets and supplies at factories and has a fixed charge
if something flows on an network arc.

The model demonstrates how to formulate some implication constraints
"binaryVariable = 1 => equation eq holds" via indicator constraints.
Usually a bigM formulation can be used to express this logic, but indicator
constraint can do this without having an a priori finite bigM. This model shows
a couple of variants using the bigM and indicator constraints.


Dantzig, G B, Chapter 3.3. In Linear Programming and Extensions.
Princeton University Press, Princeton, New Jersey, 1963.
"""

from __future__ import annotations

import os

import gamspy.formulations as formulations
from gamspy import (
    Container,
    Equation,
    Model,
    ModelStatus,
    Options,
    Parameter,
    Sense,
    Set,
    Smax,
    SolveStatus,
    Sum,
    Variable,
)


def report(model: Model) -> tuple[ModelStatus, SolveStatus, float]:
    return model.status, model.solve_status, model.objective_value


def main():
    m = Container()

    # Sets
    i = Set(
        m,
        name="i",
        records=["seattle", "san-diego"],
        description="canning plants",
    )
    j = Set(
        m,
        name="j",
        records=["new-york", "chicago", "topeka"],
        description="markets",
    )

    # Data
    a = Parameter(
        m,
        name="a",
        domain=i,
        records=[["seattle", 350], ["san-diego", 600]],
        description="capacity of plant i in cases",
    )
    b = Parameter(
        m,
        name="b",
        domain=j,
        records=[["new-york", 325], ["chicago", 300], ["topeka", 275]],
        description="demand at market j in cases",
    )
    d = Parameter(
        m,
        name="d",
        domain=[i, j],
        records=[
            ["seattle", "new-york", 2.5],
            ["seattle", "chicago", 1.7],
            ["seattle", "topeka", 1.8],
            ["san-diego", "new-york", 2.5],
            ["san-diego", "chicago", 1.8],
            ["san-diego", "topeka", 1.4],
        ],
        description="distance in thousands of miles",
    )
    f = Parameter(
        m,
        name="f",
        records=90,
        description="freight in dollars per case per thousand miles",
    )
    c = Parameter(
        m,
        name="c",
        domain=[i, j],
        description="transport cost in thousands of dollars per case",
    )
    fixcost = Parameter(
        m,
        name="fixcost",
        domain=[i, j],
        description="fixed cost in thousands of dollars",
    )
    c[i, j] = f * d[i, j] / 1000
    fixcost[i, j] = 10 * d[i, j] / 1000

    minshipping = Parameter(
        m,
        name="minshipping",
        records=100,
        description="minimum shipping of cases",
    )
    bigM = Parameter(m, name="bigM", description="sufficiently large number")
    bigM[...] = Smax(i, a[i])

    # Variables
    x = Variable(
        m,
        name="x",
        type="positive",
        domain=[i, j],
        description="shipment quantities in cases",
    )
    use = Variable(
        m,
        name="use",
        type="binary",
        domain=[i, j],
        description="is 1 if arc is used in solution",
    )

    # Equations
    supply = Equation(
        m,
        name="supply",
        domain=i,
        description="observe supply limit at plant i",
    )
    demand = Equation(
        m,
        name="demand",
        domain=j,
        description="satisfy demand at market j",
    )
    minship = Equation(
        m,
        name="minship",
        domain=[i, j],
        description="ensure minimum shipping",
    )
    maxship = Equation(
        m,
        name="maxship",
        domain=[i, j],
        description="ensure zero shipping if use variable is 0",
    )

    supply[i] = Sum(j, x[i, j]) <= a[i]
    demand[j] = Sum(i, x[i, j]) >= b[j]
    minship[i, j] = x[i, j] >= minshipping * use[i, j]
    maxship[i, j] = x[i, j] <= bigM * use[i, j]

    cost = Sum([i, j], c[i, j] * x[i, j] + fixcost[i, j] * use[i, j])
    options = Options(relative_optimality_gap=0)
    rep = {}

    bigMModel = Model(
        m,
        name="bigMModel",
        equations=[supply, demand, minship, maxship],
        problem="mip",
        sense=Sense.MIN,
        objective=cost,
    )
    bigMModel.solve(solver="cplex", options=options)
    rep["bigMModel", 1] = report(bigMModel)

    # Now let's build a model for the same problem using indicator constraints.
    # With native=True, the indicator constraints are passed to the solver
    # through the solver options file instead of being reformulated.
    _, iminship = formulations.indicator(use, 1, x[i, j] >= minshipping, native=True)
    _, imaxship = formulations.indicator(use, 0, x[i, j] == 0, native=True)

    indicatorModel = Model(
        m,
        name="indicatorModel",
        equations=[supply, demand, *iminship, *imaxship],
        problem="mip",
        sense=Sense.MIN,
        objective=cost,
    )
    indicatorModel.solve(solver="cplex", options=options)
    rep["indicatorModel", 1] = report(indicatorModel)

    # Let's do the same by writing the indicator options with labels
    iminship = Equation(
        m,
        name="iminship",
        domain=[i, j],
        description="ensure minimum shipping using indicator constraints",
    )
    imaxship = Equation(
        m,
        name="imaxship",
        domain=[i, j],
        description=(
            "ensure zero shipping if use variable is 0 using indicator constraints"
        ),
    )
    iminship[i, j] = x[i, j] >= minshipping
    imaxship[i, j] = x[i, j] == 0

    indicator_file = os.path.join(m.working_directory, "labels.opt")
    with open(indicator_file, "w", encoding="utf-8") as file:
        for ii in i.toList():
            for jj in j.toList():
                label = f"('{ii}','{jj}')"
                file.write(f"indic iminship{label}$use{label} 1\n")
                file.write(f"indic imaxship{label}$use{label} 0\n")

    labelModel = Model(
        m,
        name="labelModel",
        equations=[supply, demand, iminship, imaxship],
        problem="mip",
        sense=Sense.MIN,
        objective=cost,
    )
    labelModel.solve(solver="cplex", options=options, solver_options=indicator_file)
    rep["indicatorModel", 2] = report(labelModel)

    # Now let's build a model for the same problem that can be used with
    # and without indicator constraints. This can become handy when
    # debugging a model with indicator constraints
    minslack = Variable(m, name="minslack", type="positive", domain=[i, j])
    maxslack = Variable(m, name="maxslack", type="positive", domain=[i, j])

    xminship = Equation(
        m,
        name="xminship",
        domain=[i, j],
        description=("ensure minimum shipping using indicator constraints and bigM"),
    )
    xmaxship = Equation(
        m,
        name="xmaxship",
        domain=[i, j],
        description=(
            "ensure zero shipping if use variable is 0 using indicator"
            " constraints and bigM"
        ),
    )
    bndminslack = Equation(
        m,
        name="bndminslack",
        domain=[i, j],
        description="ensure minslack is zero if use variable is 1",
    )
    bndmaxslack = Equation(
        m,
        name="bndmaxslack",
        domain=[i, j],
        description="ensure maxslack is zero if use variable is 0",
    )

    xminship_expr = x[i, j] >= minshipping - minslack[i, j]
    xmaxship_expr = x[i, j] == 0 + maxslack[i, j]
    xminship[i, j] = xminship_expr
    xmaxship[i, j] = xmaxship_expr
    bndminslack[i, j] = minslack[i, j] <= bigM * (1 - use[i, j])
    bndmaxslack[i, j] = maxslack[i, j] <= bigM * use[i, j]

    # Let's first solve this without use of indicators
    indicatorbigMModel = Model(
        m,
        name="indicatorbigMModel",
        equations=[supply, demand, xminship, xmaxship, bndminslack, bndmaxslack],
        problem="mip",
        sense=Sense.MIN,
        objective=cost,
    )
    indicatorbigMModel.solve(solver="cplex", options=options)
    rep["indicatorbigMModel", 1] = report(indicatorbigMModel)

    # Now we will use indicators and therefore we don't need the slacks
    _, nxminship = formulations.indicator(use, 1, xminship_expr, native=True)
    _, nxmaxship = formulations.indicator(use, 0, xmaxship_expr, native=True)

    minslack.fx[i, j] = 0
    maxslack.fx[i, j] = 0
    nativeModel = Model(
        m,
        name="nativeModel",
        equations=[
            supply,
            demand,
            *nxminship,
            *nxmaxship,
            bndminslack,
            bndmaxslack,
        ],
        problem="mip",
        sense=Sense.MIN,
        objective=cost,
    )
    nativeModel.solve(solver="cplex", options=options)
    rep["indicatorbigMModel", 2] = report(nativeModel)

    # We can also mix and match bigM with indicator constraints
    maxslack.up[i, j] = float("inf")
    mixedModel = Model(
        m,
        name="mixedModel",
        equations=[
            supply,
            demand,
            *nxminship,
            xmaxship,
            bndminslack,
            bndmaxslack,
        ],
        problem="mip",
        sense=Sense.MIN,
        objective=cost,
    )
    mixedModel.solve(solver="cplex", options=options)
    rep["indicatorbigMModel", 3] = report(mixedModel)

    for (name, run), (status, solve_status, _) in rep.items():
        if status != ModelStatus.OptimalGlobal:
            raise Exception(f"{name}.{run} is not solved to global optimality")
        if solve_status != SolveStatus.NormalCompletion:
            raise Exception(f"{name}.{run} has solvestat not Normal Completion")

    objectives = [obj for _, _, obj in rep.values()]
    for (name, run), (_, _, obj) in rep.items():
        print(f"{name}.{run}: {obj:.3f}")

    if max(objectives) - min(objectives) > 1e-3:
        raise Exception("We get different objective values")


if __name__ == "__main__":
    main()
