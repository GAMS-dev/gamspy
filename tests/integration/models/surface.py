"""
## GAMSSOURCE: https://www.gams.com/latest/noalib_ml/libhtml/noalib_surface.html
## LICENSETYPE: Demo
## MODELTYPE: NLP


Minimal surface problem

Find a function f that minimizes the array of its graph subject to some
constraints on the boundary of the domain of f.

Boyd, S., Vandenberghe, L., Convex Optimization, Cambridge University Press,
Cambridge, 2004.
"""

from __future__ import annotations

import gamspy.math as gams_math
from gamspy import (
    Card,
    Container,
    Domain,
    Model,
    Ord,
    Parameter,
    Set,
    Sum,
    Variable,
)
from gamspy.math import sqr


def main():
    m = Container()

    # SETS #
    X = Set(m, name="X", records=[f"I{i}" for i in range(1, 22)])
    Y = Set(m, name="Y", records=[f"J{j}" for j in range(1, 22)])
    inside = Set(m, name="inside", domain=[X, Y])

    # Exclude i1 and i21 from inside
    inside[X, Y].where[~((Ord(X) == 1) & (Ord(X) == Card(X)))] = True

    #   display inside

    # SCALAR #
    K = Parameter(m, name="K", records=10)

    # VARIABLES #
    f = Variable(m, name="f", domain=[X, Y], type="positive")

    # Bounds on variables, initial conditions, fixing conditions:
    f.up[X, Y] = 1
    f.l[X, Y] = 1.0
    f.fx[X, Y].where[(Ord(X) == 1) | (Ord(X) == Card(X))] = 1

    # EQUATION #
    objfun = (1 / sqr(K)) * Sum(
        Domain(X, Y).where[inside[X, Y]],
        gams_math.sqrt(
            sqr((f[X + 1, Y] - f[X, Y]) / K)
            + sqr((f[X, Y + 1] - f[X, Y]) / K)
            + 1
        ),
    )

    surface = Model(
        m,
        name="surface",
        equations=m.getEquations(),
        problem="nlp",
        sense="MIN",
        objective=objfun,
    )
    surface.solve()

    print("Objective Function Value:  ", round(surface.objective_value, 4))
    print("f(X,Y):  \n", f.pivot())

    # End surface


if __name__ == "__main__":
    main()
