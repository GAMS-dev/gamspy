"""
A Production Mix Problem (PRODMIX)

A furniture company wants to maximize its profits from the
manufacture of different types of desks.


Dantzig, G B, Chapter 3.5. In Linear Programming and Extensions.
Princeton University Press, Princeton, New Jersey, 1963.

Keywords: linear programming, production planning, manufacturing, furniture
production
"""
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
    m = Container(delayed_execution=True)

    # Set
    desk = Set(m, name="desk", records=["d1", "d2", "d3", "d4"])
    shop = Set(m, name="shop", records=["carpentry", "finishing"])

    # Data
    labor = Parameter(
        m,
        name="labor",
        domain=[shop, desk],
        records=np.array([[4, 9, 7, 10], [1, 1, 3, 40]]),
    )
    caplim = Parameter(
        m, name="caplim", domain=[shop], records=np.array([6000, 4000])
    )
    price = Parameter(
        m, name="price", domain=[desk], records=np.array([12, 20, 18, 40])
    )

    # Variable
    mix = Variable(m, name="mix", domain=[desk], type="Positive")
    profit = Variable(m, name="profit")

    # Equation
    cap = Equation(m, name="cap", domain=[shop])
    ap = Equation(m, name="ap")

    cap[shop] = Sum(desk, labor[shop, desk] * mix[desk]) <= caplim[shop]
    ap[...] = profit == Sum(desk, price[desk] * mix[desk])

    pmp = Model(
        m,
        name="pmp",
        equations=m.getEquations(),
        problem="LP",
        sense=Sense.MAX,
        objective=profit,
    )

    pmp.solve()


if __name__ == "__main__":
    main()
