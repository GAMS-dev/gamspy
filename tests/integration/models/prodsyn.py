"""
## LICENSETYPE: Demo
## MODELTYPE: MIP
## KEYWORDS: mixed integer linear programming, logical and, product portfolio, cannibalization, formulations


Launching a Product Portfolio with Synergies (PRODSYN)

A consumer electronics firm chooses which new products to launch next year
under a development budget. Every product earns a standalone profit, and
some pairs of products interact when both are launched:

  - complements earn more together, e.g. a smartwatch sells better next to
    a smartphone of the same brand,
  - substitutes cannibalize each other, e.g. a budget phone takes sales
    away from the flagship smartphone.

The profit of the portfolio is

  sum(i, profit(i) * launch(i))
    + sum((i,j), synergy(i,j) * (launch(i) and launch(j)))

gamspy.formulations.logical_and makes both(i,j) equal to
launch(i) and launch(j). The synergies have both signs, so both directions
of the AND matter: without them, the firm would collect the synergies of
products it does not launch, or ignore the cannibalization of products it
does launch.
"""

from __future__ import annotations

import numpy as np

import gamspy.formulations as formulations
from gamspy import (
    Alias,
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

    # Sets
    i = Set(
        m,
        name="i",
        records=[
            "smartphone",
            "budget-phone",
            "tablet",
            "mini-tablet",
            "laptop",
            "smartwatch",
            "earbuds",
            "smart-speaker",
        ],
        description="candidate products",
    )
    j = Alias(m, name="j", alias_with=i)

    # Parameters
    cost = Parameter(
        m,
        name="cost",
        domain=i,
        records=np.array([40, 18, 30, 20, 45, 15, 8, 10]),
        description="development cost of product i in millions of dollars",
    )
    profit = Parameter(
        m,
        name="profit",
        domain=i,
        records=np.array([70, 32, 45, 30, 60, 18, 12, 11]),
        description="standalone profit of product i in millions of dollars",
    )
    synergy = Parameter(
        m,
        name="synergy",
        domain=[i, j],
        records=[
            ("smartphone", "budget-phone", -30),
            ("smartphone", "smartwatch", 16),
            ("smartphone", "earbuds", 10),
            ("smartphone", "smart-speaker", 6),
            ("budget-phone", "earbuds", 5),
            ("tablet", "mini-tablet", -24),
            ("tablet", "laptop", -18),
            ("laptop", "earbuds", 4),
        ],
        description="extra profit if products i and j are both launched",
    )
    budget = Parameter(
        m, name="budget", records=130, description="budget in millions of dollars"
    )

    # Variables
    launch = Variable(
        m,
        name="launch",
        type="binary",
        domain=i,
        description="product i is launched",
    )

    # Equations
    spend = Equation(m, name="spend", description="observe the budget")
    spend[...] = Sum(i, cost[i] * launch[i]) <= budget

    # both[i, j] == launch[i] and launch[j]
    both, eqs = formulations.logical_and(launch[i], launch[j])

    portfolio = Model(
        m,
        name="portfolio",
        equations=[spend, *eqs],
        problem="mip",
        sense=Sense.MAX,
        objective=Sum(i, profit[i] * launch[i])
        + Sum([i, j], synergy[i, j] * both[i, j]),
    )
    portfolio.solve()

    print("Objective Function Value: ", round(portfolio.objective_value, 3))
    print(launch.toDense())

    # both equals the AND of the launch decisions for every pair
    launched = launch.toDense().round()
    if np.abs(both.toDense() - np.outer(launched, launched)).max() > 1e-6:
        raise Exception("both does not equal the logical AND")


if __name__ == "__main__":
    main()
