from gamspy import Set, Parameter, Variable, Equation, Model, Container
from gamspy import Sum
import numpy as np


def main():
    m = Container()

    # Set
    alloy = Set(m, name="alloy", records=["a", "b", "c", "d", "e", "f", "g", "h", "i"])
    elem = Set(m, name="elem", records=["lead", "zinc", "tin"])

    # Data
    compdat = Parameter(
        m,
        name="compdat",
        domain=[elem, alloy],
        records=np.array(
            [
                [10, 10, 40, 60, 30, 30, 30, 50, 20],
                [10, 30, 50, 30, 30, 40, 20, 40, 30],
                [80, 60, 10, 10, 40, 30, 50, 10, 50],
            ]
        ),
    )
    price = Parameter(
        m,
        name="price",
        domain=[alloy],
        records=np.array([4.1, 4.3, 5.8, 6.0, 7.6, 7.5, 7.3, 6.9, 7.3]),
    )
    rb = Parameter(m, name="rb", domain=[elem], records=np.array([30, 30, 40]))

    # Variable
    v = Variable(m, name="v", domain=[alloy], type="Positive")
    phi = Variable(m, name="phi")

    # Equation
    pc = Equation(m, name="pc", domain=[elem], type="eq")
    mb = Equation(m, name="mb", type="eq")
    ac = Equation(m, name="ac", type="eq")

    pc[elem] = Sum(alloy, compdat[elem, alloy] * v[alloy]) == rb[elem]
    mb.definition = Sum(alloy, v[alloy]) == 1
    ac.definition = phi == Sum(alloy, price[alloy] * v[alloy])

    b1 = Model(m, name="b1", equations=[pc, ac])
    b2 = Model(m, name="b2", equations="all")

    report = Parameter(m, name="report", domain=[alloy, "*"])

    m.solve(b1, problem="LP", sense="min", objective_variable=phi)

    report[alloy, "blend-1"] = v.l[alloy]
    m.solve(b2, problem="LP", sense="min", objective_variable=phi)
    report[alloy, "blend-2"] = v.l[alloy]
    # Can be removed after devel/gams-transfer-python#69 gets fixed
    report.records.columns = ["alloy", "uni", "value"]
    print(report.pivot())


if __name__ == "__main__":
    main()
