"""
Blending Problem I (BLEND)

A company wishes to produce a lead-zinc-tin alloy at minimal cost.
The problem is to blend a new alloy from other purchased alloys.


Dantzig, G B, Chapter 3.4. In Linear Programming and Extensions.
Princeton University Press, Princeton, New Jersey, 1963.

Keywords: linear programming, blending problem, manufacturing, alloy blending
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
    m = Container(delayed_execution=int(os.getenv("DELAYED_EXECUTION", False)))

    # Set
    alloy = Set(
        m, name="alloy", records=["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    )
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

    # Equation
    pc = Equation(m, name="pc", domain=[elem])
    mb = Equation(m, name="mb")

    pc[elem] = Sum(alloy, compdat[elem, alloy] * v[alloy]) == rb[elem]
    mb[...] = Sum(alloy, v[alloy]) == 1

    b1 = Model(
        m,
        name="b1",
        equations=[pc],
        problem="LP",
        sense=Sense.MIN,
        objective=Sum(alloy, price[alloy] * v[alloy]),
    )
    b2 = Model(
        m,
        name="b2",
        equations=[pc, mb],
        problem="LP",
        sense=Sense.MIN,
        objective=Sum(alloy, price[alloy] * v[alloy]),
    )

    report = Parameter(m, name="report", domain=[alloy, "*"])

    b1.solve()

    report[alloy, "blend-1"] = v.l[alloy]
    b2.solve()
    report[alloy, "blend-2"] = v.l[alloy]

    print(report.pivot())

    import math

    assert math.isclose(b2.objective_value, 4.980000, rel_tol=0.001)


if __name__ == "__main__":
    main()
