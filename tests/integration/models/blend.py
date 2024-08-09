"""
## GAMSSOURCE: https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_blend.html
## LICENSETYPE: Demo
## MODELTYPE: LP
## KEYWORDS: linear programming, blending problem, manufacturing, alloy blending


Blending Problem I (BLEND)

A company wishes to produce a lead-zinc-tin alloy at minimal cost.
The problem is to blend a new alloy from other purchased alloys.


Dantzig, G B, Chapter 3.4. In Linear Programming and Extensions.
Princeton University Press, Princeton, New Jersey, 1963.
"""

from __future__ import annotations

import numpy as np
from gamspy import (
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

    # Set
    alloy = Set(
        m,
        name="alloy",
        records=["a", "b", "c", "d", "e", "f", "g", "h", "i"],
        description="products on the market",
    )
    elem = Set(
        m,
        name="elem",
        records=["lead", "zinc", "tin"],
        description="required elements",
    )

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
        description="composition data (pct)",
    )
    price = Parameter(
        m,
        name="price",
        domain=alloy,
        records=np.array([4.1, 4.3, 5.8, 6.0, 7.6, 7.5, 7.3, 6.9, 7.3]),
        description="composition data (price)",
    )
    rb = Parameter(
        m,
        name="rb",
        domain=elem,
        records=np.array([30, 30, 40]),
        description="required blend",
    )

    # Variable
    v = Variable(
        m,
        name="v",
        domain=alloy,
        type="Positive",
        description="purchase of alloy (pounds)",
    )

    # Equation
    objective = Sum(alloy, price[alloy] * v[alloy])
    pc = Equation(m, name="pc", domain=elem, description="purchase constraint")
    mb = Equation(m, name="mb", description="material balance")

    pc[elem] = Sum(alloy, compdat[elem, alloy] * v[alloy]) == rb[elem]
    mb[...] = Sum(alloy, v[alloy]) == 1

    b1 = Model(
        m,
        name="b1",
        equations=[pc],
        problem="LP",
        sense=Sense.MIN,
        objective=objective,
    )
    b2 = Model(
        m,
        name="b2",
        equations=[pc, mb],
        problem="LP",
        sense=Sense.MIN,
        objective=objective,
    )

    report = Parameter(
        m,
        name="report",
        domain=[alloy, "*"],
        description="comparison of model 1 and 2",
    )

    b1.solve()

    report[alloy, "blend-1"] = v.l[alloy]
    b2.solve()
    report[alloy, "blend-2"] = v.l[alloy]

    print(report.pivot())

    import math

    assert math.isclose(b2.objective_value, 4.980000, rel_tol=0.001)


if __name__ == "__main__":
    main()
