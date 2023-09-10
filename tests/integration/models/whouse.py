"""
Simple Warehouse Problem (WHOUSE)

A warehouse can store limited units of a commodity. Given an
initial stock, the manager has to decide when to buy or sell in
order to minimize total cost.


Dantzig, G B, Chapter 3.6. In Linear Programming and Extensions.
Princeton University Press, Princeton, New Jersey, 1963.

Keywords: linear programming, warehouse management, inventory
"""

from gamspy import (
    Set,
    Parameter,
    Variable,
    Equation,
    Container,
    Sum,
    Model,
    Sense,
)
import numpy as np


def main():
    m = Container()

    # Sets
    t = Set(m, name="t", records=[f"q-{i}" for i in range(1, 5)])

    # Parameters
    price = Parameter(
        m, name="price", domain=[t], records=np.array([10, 12, 8, 9])
    )
    istock = Parameter(
        m, name="istock", domain=[t], records=np.array([50, 0, 0, 0])
    )  # OR records=pd.DataFrame([["q-1", 50]])

    # Scalars
    storecost = Parameter(m, name="storecost", records=1)
    storecap = Parameter(m, name="storecap", records=100)

    # Variables
    stock = Variable(m, name="stock", domain=[t], type="Positive")
    sell = Variable(m, name="sell", domain=[t], type="Positive")
    buy = Variable(m, name="buy", domain=[t], type="Positive")
    cost = Variable(m, name="cost")

    # Equations
    sb = Equation(m, name="sb", domain=[t])
    at = Equation(m, name="at")

    sb[t] = (
        stock[t] == stock[t.lag(1, "linear")] + buy[t] - sell[t] + istock[t]
    )
    at.expr = cost == Sum(
        t, price[t] * (buy[t] - sell[t]) + storecost * stock[t]
    )

    stock.up[t] = storecap

    swp = Model(
        m,
        name="swp",
        equations=m.getEquations(),
        problem="LP",
        sense=Sense.MIN,
        objective=cost,
    )
    swp.solve()

    print("Objective function value: ", cost.records.level[0])


if __name__ == "__main__":
    main()
