"""
## GAMSSOURCE: https://www.gams.com/latest/noalib_ml/libhtml/noalib_control3.html
## LICENSETYPE: Demo
## MODELTYPE: NLP


Control3 : Optimal Control Problem with a Nonlinear Dynamic Constraint and Boundary Conditions Solved as a General Nonlinear Programming Problem

Optimal control problem with a nonlinear dynamic constraint and boundary
conditions solved as a General Nonlinear Programming Problem.

Divya Garg, et al., Direct trajectory optimization and costate estimation of
finite-horizon and infinite-horizon optimal control problems using a
Radau pseudospectral method. Computational optimization and Applications,
vol.49, nr. 2, June 2011, pp. 335-358.
"""

from __future__ import annotations

import gamspy.math as gams_math
from gamspy import (
    Card,
    Container,
    Equation,
    Model,
    Number,
    Ord,
    Parameter,
    Set,
    Sum,
    Variable,
)


def main():
    m = Container()

    # SETS #
    n = Set(m, name="n", records=["state1"], description="states")
    k = Set(m, name="k", records=[f"t{t}" for t in range(1, 101)])
    ku = Set(m, name="ku", domain=k, description="control horizon")
    ki = Set(m, name="ki", domain=k, description="initial ")
    kt = Set(m, name="kt", domain=k, description="terminal period")

    ku[k] = Number(1).where[Ord(k) < Card(k)]
    ki[k] = Number(1).where[Ord(k) == 1]
    kt[k] = ~ku[k]

    # PARAMETERS #
    rk = Parameter(m, name="rk", records=0.01, description="penalty control")
    _ = Parameter(
        m,
        name="xinit",
        domain=n,
        records=[("state1", 2)],
        description="initial value",
    )

    # VARIABLES #
    x = Variable(m, name="x", domain=[n, k], description="state variable")
    u = Variable(m, name="u", domain=k, description="control variable")

    # EQUATIONS #
    stateq = Equation(
        m,
        name="stateq",
        type="regular",
        domain=[n, k],
        description="state equation",
    )
    stateq[n, k.lead(1)] = x[n, k.lead(1)] == 2 * x[n, k] + 2 * u[
        k
    ] * gams_math.sqrt(x[n, k])

    # OBJECTIVE #
    j = 0.5 * Sum([k, n], (x[n, k]) + 0.5 * Sum(ku, (u[ku]) * rk * (u[ku])))

    control3 = Model(
        m,
        name="control3",
        equations=m.getEquations(),
        problem="nlp",
        sense="MIN",
        objective=j,
    )

    control3.solve(solver="conopt")

    import math

    assert math.isclose(control3.objective_value, 0)


if __name__ == "__main__":
    main()
