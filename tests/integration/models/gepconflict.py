"""
## LICENSETYPE: Demo
## MODELTYPE: LP
## KEYWORDS: linear programming, generation expansion planning, energy economics, infeasibility analysis, feasibility relaxation


Generation Expansion with Conflicting Policy Targets (GEPCONFLICT)

A utility plans the capacity of coal, gas, wind and solar plants for one year
that is represented by four time slices. The plan minimizes the annualized
investment and operating cost and has to meet three targets:

  - reserve margin: the firm capacity exceeds the peak demand by 15%,
  - renewable share: wind and solar produce half of the energy,
  - investment budget: the annualized investment stays within the budget.

The targets conflict, so the model is infeasible. Model.feasibilityRelaxation
allows the targets to be violated and finds the smallest violation. The
targets are measured in MW, MWh and thousands of dollars, so each violation
is weighted with the reciprocal of its target to compare relative violations.

The first relaxation shows that lowering the renewable share is the smallest
relative change. The second relaxation keeps the reserve margin and the
renewable share and shows how much the budget has to grow instead. Both
answers are verified by changing the target and solving the original model.
"""

from __future__ import annotations

import math

from gamspy import (
    Container,
    Equation,
    Model,
    ModelStatus,
    Parameter,
    Sense,
    Set,
    Smax,
    Sum,
    Variable,
)


def main():
    m = Container()

    # Sets
    k = Set(
        m,
        name="k",
        records=["coal", "gas", "wind", "solar"],
        description="technologies",
    )
    r = Set(
        m,
        name="r",
        domain=k,
        records=["wind", "solar"],
        description="renewable technologies",
    )
    h = Set(
        m,
        name="h",
        records=["winter-day", "winter-night", "summer-day", "summer-night"],
        description="time slices",
    )

    # Parameters
    hours = Parameter(
        m,
        name="hours",
        domain=h,
        records=[(slice_, 2190) for slice_ in h.toList()],
        description="hours per year in time slice h",
    )
    demand = Parameter(
        m,
        name="demand",
        domain=h,
        records=[
            ("winter-day", 900),
            ("winter-night", 700),
            ("summer-day", 1000),
            ("summer-night", 600),
        ],
        description="demand in time slice h in MW",
    )
    existing = Parameter(
        m,
        name="existing",
        domain=k,
        records=[("coal", 500), ("gas", 300), ("wind", 100), ("solar", 50)],
        description="existing capacity of technology k in MW",
    )
    availability = Parameter(
        m,
        name="availability",
        domain=[k, h],
        records=[
            ("coal", "winter-day", 0.9),
            ("coal", "winter-night", 0.9),
            ("coal", "summer-day", 0.9),
            ("coal", "summer-night", 0.9),
            ("gas", "winter-day", 0.95),
            ("gas", "winter-night", 0.95),
            ("gas", "summer-day", 0.95),
            ("gas", "summer-night", 0.95),
            ("wind", "winter-day", 0.35),
            ("wind", "winter-night", 0.45),
            ("wind", "summer-day", 0.2),
            ("wind", "summer-night", 0.3),
            ("solar", "winter-day", 0.25),
            ("solar", "summer-day", 0.45),
        ],
        description="available share of the capacity of technology k in slice h",
    )
    credit = Parameter(
        m,
        name="credit",
        domain=k,
        records=[("coal", 0.9), ("gas", 0.9), ("wind", 0.1), ("solar", 0.05)],
        description="share of the capacity of technology k that counts as firm",
    )
    invcost = Parameter(
        m,
        name="invcost",
        domain=k,
        records=[("coal", 250), ("gas", 100), ("wind", 150), ("solar", 90)],
        description="annualized investment cost in thousands of dollars per MW",
    )
    varcost = Parameter(
        m,
        name="varcost",
        domain=k,
        records=[("coal", 30), ("gas", 60)],
        description="operating cost in dollars per MWh",
    )
    margin = Parameter(m, name="margin", records=0.15, description="reserve margin")
    target = Parameter(
        m, name="target", records=0.5, description="renewable share of the energy"
    )
    budget = Parameter(
        m,
        name="budget",
        records=120000,
        description="investment budget in thousands of dollars per year",
    )

    # Variables
    build = Variable(
        m,
        name="build",
        type="positive",
        domain=k,
        description="new capacity of technology k in MW",
    )
    gen = Variable(
        m,
        name="gen",
        type="positive",
        domain=[k, h],
        description="generation of technology k in slice h in MW",
    )

    # Equations
    balance = Equation(
        m, name="balance", domain=h, description="cover the demand in slice h"
    )
    balance[h] = Sum(k, gen[k, h]) == demand[h]
    available = Equation(
        m,
        name="available",
        domain=[k, h],
        description="generate within the available capacity",
    )
    available[k, h] = gen[k, h] <= availability[k, h] * (existing[k] + build[k])
    reserve = Equation(m, name="reserve", description="keep the reserve margin")
    reserve[...] = Sum(k, credit[k] * (existing[k] + build[k])) >= (1 + margin) * Smax(
        h, demand[h]
    )
    share = Equation(m, name="share", description="reach the renewable share")
    share[...] = Sum((r, h), hours[h] * gen[r, h]) >= target * Sum(
        h, hours[h] * demand[h]
    )
    spend = Equation(m, name="spend", description="observe the investment budget")
    spend[...] = Sum(k, invcost[k] * build[k]) <= budget

    plan = Model(
        m,
        name="plan",
        equations=[balance, available, reserve, share, spend],
        problem="lp",
        sense=Sense.MIN,
        objective=Sum(k, 1000 * invcost[k] * build[k])
        + Sum((k, h), hours[h] * varcost[k] * gen[k, h]),
    )
    plan.solve()
    original = plan.status
    print("Original plan:", original)
    if original == ModelStatus.OptimalGlobal:
        raise Exception("the targets should conflict")

    # Measure each violation relative to its target
    energy = Parameter(m, name="energy", description="energy demand in MWh")
    energy[...] = Sum(h, hours[h] * demand[h])
    w_reserve = Parameter(m, name="w_reserve", description="weight per MW of reserve")
    w_reserve[...] = 1 / ((1 + margin) * Smax(h, demand[h]))
    w_share = Parameter(m, name="w_share", description="weight per MWh of renewables")
    w_share[...] = 1 / (target * energy)
    w_spend = Parameter(m, name="w_spend", description="weight per thousand dollars")
    w_spend[...] = 1 / budget

    relax = plan.feasibilityRelaxation(
        equations=[reserve, share, spend],
        weights={reserve: w_reserve, share: w_share, spend: w_spend},
    )
    relax.solve()
    violations = relax.violations()
    print("Smallest relative violation:", round(relax.violation.toValue(), 4))
    print("Violated targets:", list(violations))

    shortfall = -violations["share"]["violation"].item()
    reachable = target.toValue() - shortfall / energy.toValue()
    print("Reachable renewable share:", round(reachable, 4))

    # Keep the reserve margin and the renewable share, raise the budget instead
    relax_budget = plan.feasibilityRelaxation(equations=[spend])
    relax_budget.solve()
    extra = relax_budget.violations()["spend"]["violation"].item()
    print("Additional budget needed:", round(extra, 2))

    relax_count = plan.feasibilityRelaxation(
        equations=[reserve, share, spend], measure="count"
    )
    relax_count.solve()
    print("Targets that must change:", relax_count.violation.toValue())

    # Verify both answers with the original model
    target[...] = reachable
    plan.solve()
    with_share = plan.status
    print("Plan with the reachable share:", with_share)

    target[...] = 0.5
    budget[...] = budget.toValue() + extra
    plan.solve()
    with_budget = plan.status
    print("Plan with the additional budget:", with_budget)
    print("Objective Function Value: ", round(plan.objective_value, 3))
    print("New capacity in MW:", [(tech, round(mw, 1)) for tech, mw in build.toList()])

    if with_share != ModelStatus.OptimalGlobal:
        raise Exception("the reachable share should make the plan feasible")

    if with_budget != ModelStatus.OptimalGlobal:
        raise Exception("the additional budget should make the plan feasible")

    if not math.isclose(relax.violation.toValue(), 0.46625, rel_tol=1e-4):
        raise Exception("unexpected relative violation")

    if not math.isclose(extra, 79700.85, rel_tol=1e-4):
        raise Exception("unexpected additional budget")


if __name__ == "__main__":
    main()
