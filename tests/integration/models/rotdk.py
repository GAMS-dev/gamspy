"""
## GAMSSOURCE: https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_rotdk.html
## LICENSETYPE: Requires license
## MODELTYPE: MIP
## KEYWORDS: mixed integer linear programming, robust optimization, capacity expansion, time-dependent knapsack problem


Robust Optimization (ROTDK)

Robust Optimization.


Laguna, M, Applying Robust Optimization to Capacity Expansion of
One Location in Telecommunications with Demand Uncertainty.
Management Science 44, 11 (1998), 101-110.
"""

from __future__ import annotations

from gamspy import (
    Alias,
    Card,
    Container,
    Equation,
    Model,
    Options,
    Ord,
    Parameter,
    Sense,
    Set,
    Sum,
    Variable,
)
from gamspy.math import Round, normal, power, uniform


def main():
    m = Container()

    # Set
    s = Set(
        m,
        name="s",
        records=[str(i) for i in range(1, 1001)],
        description="scenarios",
    )
    t = Set(
        m,
        name="t",
        records=[f"t{i}" for i in range(1, 13)],
        description="time periods",
    )
    j = Set(
        m,
        name="j",
        records=[f"C{i:03}" for i in range(1, 11)],
        description="components",
    )

    tt = Alias(m, name="tt", alias_with=t)

    # Parameter
    di = Parameter(m, name="di", domain=[s, t], description="increment")
    d = Parameter(m, name="D", domain=[t, s], description="demand")
    c = Parameter(m, name="c", domain=j, description="capacity size")
    p = Parameter(m, name="p", domain=j, description="capacity cost")
    mu = Parameter(m, name="mu", description="mean capacity parameter")
    sigma = Parameter(m, name="sigma", description="std capacity parameter")

    mu_value = 100
    sigma_value = 10
    mu[...] = mu_value
    sigma[...] = sigma_value

    c[j] = Round(uniform(1, mu_value))
    p[j] = Round(mu_value + c[j] + uniform(-sigma_value, sigma_value))

    di[s, t].where[(Ord(s)) <= (0.25 * Card(s))] = Round(normal(50, 10))
    di[s, t].where[(Ord(s) > 0.25 * Card(s)) & (Ord(s) <= 0.75 * Card(s))] = (
        Round(normal(100, 20))
    )
    di[s, t].where[Ord(s) > 0.75 * Card(s)] = Round(normal(150, 40))

    d[t, s] = Sum(tt.where[Ord(tt) <= Ord(t)], di[s, tt])

    # Parameter
    dis = Parameter(m, name="dis", domain=t, description="discount factor")
    w = Parameter(m, name="w", description="shortage penalty")

    dis[t] = power(0.86, Ord(t) - 1)
    w[...] = 5

    # Variable
    x = Variable(
        m, name="x", type="integer", domain=[j, t], description="expansion"
    )
    z = Variable(
        m,
        name="z",
        type="positive",
        domain=s,
        description="max capacity shortage",
    )
    cap = Variable(
        m,
        name="cap",
        type="free",
        domain=t,
        description="installed capacity",
    )

    # Equation
    capbal = Equation(
        m,
        name="capbal",
        domain=t,
        description="capacity balance",
    )
    dembal = Equation(
        m,
        name="dembal",
        domain=[t, s],
        description="demand balance",
    )

    # Objective Function
    objdef = Sum((j, t), dis[t] * p[j] * x[j, t]) + w / Card(s) * Sum(s, z[s])

    capbal[t] = cap[t] == cap[t - 1] + Sum(j, c[j] * x[j, t])

    dembal[t, s] = cap[t] + z[s] >= d[t, s]

    rotdk = Model(
        m,
        name="rotdk",
        equations=m.getEquations(),
        problem="mip",
        sense=Sense.MIN,
        objective=objdef,
    )

    rotdk.solve(
        options=Options(variable_listing_limit=0, equation_listing_limit=0)
    )

    print("Objective Function Value: ", round(rotdk.objective_value, 2))


if __name__ == "__main__":
    main()
