"""
## GAMSSOURCE: https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_whouse.html
## LICENSETYPE: Demo
## MODELTYPE: LP
## KEYWORDS: linear programming, warehouse management, inventory


Simple Warehouse Problem (WHOUSE)

A warehouse can store limited units of a commodity. Given an
initial stock, the manager has to decide when to buy or sell in
order to minimize total cost.


Dantzig, G B, Chapter 3.6. In Linear Programming and Extensions.
Princeton University Press, Princeton, New Jersey, 1963.
"""

from __future__ import annotations

import os

import numpy as np

from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Parameter
from gamspy import Sense
from gamspy import Set
from gamspy import Sum
from gamspy import Variable


def main():
    m = Container(
        system_directory=os.getenv("SYSTEM_DIRECTORY", None),
    )

    # Sets
    t = Set(
        m,
        name="t",
        records=[f"q-{i}" for i in range(1, 5)],
        description="time in quarters",
    )

    # Parameters
    price = Parameter(
        m,
        name="price",
        domain=t,
        records=np.array([10, 12, 8, 9]),
        description="selling price ($ per unit)",
    )
    istock = Parameter(
        m,
        name="istock",
        domain=t,
        records=np.array([50, 0, 0, 0]),
        description="initial stock      (units)",
    )  # OR records=pd.DataFrame([["q-1", 50]])

    # Scalars
    storecost = Parameter(
        m,
        name="storecost",
        records=1,
        description="storage cost  ($ per quarter per unit)",
    )
    storecap = Parameter(
        m,
        name="storecap",
        records=100,
        description="stocking capacity of warehouse (units)",
    )

    # Variables
    stock = Variable(
        m,
        name="stock",
        domain=t,
        type="Positive",
        description="stock stored at time t (units)",
    )
    sell = Variable(
        m,
        name="sell",
        domain=t,
        type="Positive",
        description="stock sold at time t   (units)",
    )
    buy = Variable(
        m,
        name="buy",
        domain=t,
        type="Positive",
        description="stock bought at time t (units)",
    )

    # Equations
    sb = Equation(
        m, name="sb", domain=t, description="stock balance at time t (units)"
    )

    sb[t] = (
        stock[t] == stock[t.lag(1, "linear")] + buy[t] - sell[t] + istock[t]
    )

    # ObjectFunction; accounting: total cost      ($)
    at = Sum(t, price[t] * (buy[t] - sell[t]) + storecost * stock[t])

    stock.up[t] = storecap

    swp = Model(
        m,
        name="swp",
        equations=m.getEquations(),
        problem="LP",
        sense=Sense.MIN,
        objective=at,
    )
    swp.solve()

    import math

    assert math.isclose(swp.objective_value, -600, rel_tol=0.001)

    print("Objective function value: ", swp.objective_value)


if __name__ == "__main__":
    main()
