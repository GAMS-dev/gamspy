"""
## LICENSETYPE: Demo
## MODELTYPE: MIP


Capacitated Lot-Sizing Problem (CLSP)
"""

from __future__ import annotations

import os
from itertools import product

import pandas as pd
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


def model_data():
    # Sets
    products = ["Product_A", "Product_B", "Product_C"]
    time_periods = [1, 2, 3, 4]
    resources = ["Resource_A", "Resource_B"]

    kj = pd.DataFrame(
        product(products, resources)
    )  # product-resource combinations -->  ("Product_A", "Resource_A"), ("Product_B", "Resource_A"), ...

    # Parameters
    demand_data = pd.DataFrame(
        {
            "Product_A": {1: 100, 2: 150, 3: 120, 4: 180},
            "Product_B": {1: 80, 2: 100, 3: 90, 4: 120},
            "Product_C": {1: 50, 2: 60, 3: 70, 4: 80},
        }
    ).unstack()

    setup_cost_data = pd.DataFrame(
        [("Product_A", 100), ("Product_B", 200), ("Product_C", 300)]
    )
    holding_cost_data = pd.DataFrame(
        [("Product_A", 0.2), ("Product_B", 0.1), ("Product_C", 0.6)]
    )

    capacity_data = pd.DataFrame(
        [
            ("Resource_A", 1, 340),
            ("Resource_B", 1, 340),
            ("Resource_A", 2, 330),
            ("Resource_B", 2, 330),
            ("Resource_A", 3, 300),
            ("Resource_B", 3, 300),
            ("Resource_A", 4, 380),
            ("Resource_B", 4, 380),
        ]
    )

    return {
        "products": products,
        "time_periods": time_periods,
        "resources": resources,
        "kj": kj,
        "demand_data": demand_data,
        "setup_cost_data": setup_cost_data,
        "holding_cost_data": holding_cost_data,
        "capacity_data": capacity_data,
    }


def main():
    m = Container(
        system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None),
    )
    data = model_data()

    # SETS
    k = Set(
        m,
        name="k",
        description="products",
        records=data["products"],
    )
    j = Set(m, name="j", description="resources", records=data["resources"])
    t = Set(
        m, name="t", description="time periods", records=data["time_periods"]
    )

    KJ = Set(
        m,
        name="KJ",
        domain=[k, j],
        description="products k that can be handled by resource j",
        records=data["kj"],
    )

    # ALIAS
    tau = Alias(m, name="tau", alias_with=t)

    # PARAMETERS
    d = Parameter(
        m,
        name="d",
        domain=[k, t],
        description="demand of product k in period t",
        records=data["demand_data"],
    )
    s = Parameter(
        m,
        name="s",
        domain=k,
        description="fixed setup cost for product k",
        records=data["setup_cost_data"],
    )
    h = Parameter(
        m,
        name="h",
        domain=k,
        description="holding cost for product k",
        records=data["holding_cost_data"],
    )
    c = Parameter(
        m,
        name="c",
        domain=[j, t],
        description="production capacity of resource j in period t",
        records=data["capacity_data"],
    )

    # VARIABLES
    X = Variable(
        m,
        name="X",
        domain=[k, t],
        type="positive",
        description="lot size of product k in period t",
    )
    Y = Variable(
        m,
        name="Y",
        domain=[k, t],
        type="binary",
        description="indicates if product k is manufactured in period t",
    )
    Z = Variable(
        m,
        name="Z",
        domain=[k, t],
        type="positive",
        description="stock of product k in period t",
    )

    # EQUATIONS
    objective = Sum((k, t), s[k] * Y[k, t] + h[k] * Z[k, t])

    stock = Equation(
        m, name="stock", domain=[k, t], description="Stock balance equation"
    )
    stock[...] = Z[k, t] == Z[k, t.lag(1)] + X[k, t] - d[k, t]

    production = Equation(
        m, name="production", domain=[k, t], description="Ensure production"
    )
    production[...] = X[k, t] <= Y[k, t] * Sum(tau, d[k, tau])

    capacity = Equation(
        m, name="capacity", domain=[j, t], description="Capacity restriction"
    )
    capacity[...] = Sum(KJ[k, j], X[k, t]) <= c[j, t]

    Z.fx[k, t].where[t.last] = 0

    # Model definition
    clsp = Model(
        m,
        name="CLSP",
        problem="MIP",
        equations=m.getEquations(),
        sense=Sense.MIN,
        objective=objective,
    )

    clsp.solve()

    import math

    assert math.isclose(clsp.objective_value, 1694, rel_tol=0.001)

    print("Objective function value:", clsp.objective_value)

    print("X: \n", X.pivot(index="t", columns="k"))

    print("Z: \n", Z.pivot(index="t", columns="k"))


if __name__ == "__main__":
    main()
