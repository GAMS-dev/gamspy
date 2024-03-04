"""
## LICENSETYPE: Demo
## MODELTYPE: NLP


Design of a disc flywheel

Schittkowski, K., More test examples for nonlinear programming codes.
Lecture Notes in Economics and Mathematical Systems, Vol.282, Springer-Verlag,
Berlin, 1987. (Problem 346, page 167)
"""

from __future__ import annotations

import os

from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Variable


def main():
    m = Container(
        system_directory=os.getenv("SYSTEM_DIRECTORY", None),
    )

    # VARIABLES #
    x1 = Variable(m, name="x1")
    x2 = Variable(m, name="x2")
    x3 = Variable(m, name="x3")

    # EQUATIONS #
    e1 = Equation(m, name="e1", type="regular")
    e2 = Equation(m, name="e2", type="regular")

    # Objective function:
    obj = -0.0201 * (x1**4) * x2 * (x3**2) / 10000000

    # Constraints:
    e1[...] = 675 - (x1**2) * x2 >= 0
    e2[...] = 0.419 - (x1**2) * (x3**2) / 10000000 >= 0

    # Bounds on variables:
    x1.lo[...] = 0
    x1.up[...] = 36
    x2.lo[...] = 0
    x2.up[...] = 5
    x3.lo[...] = 0
    x3.up[...] = 125

    # Initial point:
    x1.l[...] = 22.3
    x2.l[...] = 0.5
    x3.l[...] = 125

    flywheel = Model(
        m,
        name="flywheel",
        equations=m.getEquations(),
        problem="nlp",
        sense="MIN",
        objective=obj,
    )

    flywheel.solve()

    print("Objective Function Value:  ", round(flywheel.objective_value, 4))
    print("x1:  ", round(x1.toValue(), 3))
    print("x2:  ", round(x2.toValue(), 3))
    print("x3:  ", round(x3.toValue(), 3))

    # End flywheel


if __name__ == "__main__":
    main()
