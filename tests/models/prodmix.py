from gamspy import Set, Parameter, Variable, Equation, Model, Container
from gamspy import Sum
import numpy as np


def main():
    m = Container()

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
    caplim = Parameter(m, name="caplim", domain=[shop], records=np.array([6000, 4000]))
    price = Parameter(
        m, name="price", domain=[desk], records=np.array([12, 20, 18, 40])
    )

    # Variable
    mix = Variable(m, name="mix", domain=[desk], type="Positive")
    profit = Variable(m, name="profit")

    # Equation
    cap = Equation(m, name="cap", domain=[shop], type="leq")
    ap = Equation(m, name="ap", type="eq")

    cap[shop] = Sum(desk, labor[shop, desk] * mix[desk]) <= caplim[shop]
    ap.definition = profit == Sum(desk, price[desk] * mix[desk])

    pmp = Model(m, name="pmp", equations="all")

    m.solve(pmp, problem="LP", sense="max", objective_variable=profit)


if __name__ == "__main__":
    main()
