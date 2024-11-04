"""
## GAMSSOURCE: https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_chain.html
## LICENSETYPE: Demo
## MODELTYPE: NLP


Hanging Chain COPS 2.0 #3

Find the chain (of uniform density) of length L suspended between two
points with minimal potential energy.

This model is from the COPS benchmarking suite.
See http://www-unix.mcs.anl.gov/~more/cops/.

The number of intervals for the discretization can be specified using
the command line parameter --nh. COPS performance tests have been
reported for nh = 50, 100, 200, 400

Tested with nh=3000, 4000, 5000     May 26, 2005

References:
Neculai Andrei, "Models, Test Problems and Applications for
Mathematical Programming". Technical Press, Bucharest, 2003.
Application A7, page 350.

Dolan, E D, and More, J J, Benchmarking Optimization Software with COPS.
Tech. rep., Mathematics and Computer Science Division, 2000.

Cesari, L, Optimization - Theory and Applications. Springer Verlag, 1983.
"""

from __future__ import annotations

import sys

import gamspy.math as gams_math
from gamspy import (
    Alias,
    Card,
    Container,
    Equation,
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

    n_rec = int(sys.argv[1]) if len(sys.argv) > 1 else 400

    # SETS #
    nh = Set(m, name="nh", records=[f"i{i}" for i in range(n_rec + 1)])

    # ALIASES #
    i = Alias(m, name="i", alias_with=nh)

    # SCALARS #
    L = Parameter(
        m, name="L", records=4, description="length of the suspended chain"
    )
    a = Parameter(
        m, name="a", records=1, description="height of the chain at t=0 (left)"
    )
    b = Parameter(
        m, name="b", records=3, description="height of the chain at t=1 (left)"
    )
    tf = Parameter(
        m, name="tf", records=1, description="ODEs defined in [0 tf]"
    )
    h = Parameter(m, name="h", description="uniform interval length")
    n = Parameter(m, name="n", description="number of subintervals")
    tmin = Parameter(m, name="tmin")

    if b.toValue() > a.toValue():
        tmin[...] = 0.25
    else:
        tmin[...] = 0.75

    n[...] = Card(nh) - 1
    h[...] = tf / n

    # VARIABLES #
    x = Variable(m, name="x", domain=i, description="height of the chain")
    u = Variable(m, name="u", domain=i, description="derivative of x")

    x.fx["i0"] = a
    x.fx[f"i{n_rec}"] = b

    x.l[i] = (
        4
        * gams_math.abs(b - a)
        * ((Ord(i) - 1) / n)
        * (0.5 * ((Ord(i) - 1) / n) - tmin)
        + a
    )
    u.l[i] = 4 * gams_math.abs(b - a) * (((Ord(i) - 1) / n) - tmin)

    # EQUATIONS #
    x_eqn = Equation(m, name="x_eqn", type="regular", domain=i)
    length_eqn = Equation(m, name="length_eqn", type="regular")

    energy = (
        0.5
        * h
        * Sum(
            nh[i + 1],
            x[i] * gams_math.sqrt(1 + sqr(u[i]))
            + x[i + 1] * gams_math.sqrt(1 + sqr(u[i + 1])),
        )
    )

    x_eqn[i + 1] = x[i + 1] == x[i] + 0.5 * h * (u[i] + u[i + 1])

    length_eqn[...] = (
        0.5
        * h
        * Sum(
            nh[i + 1],
            gams_math.sqrt(1 + sqr(u[i])) + gams_math.sqrt(1 + sqr(u[i + 1])),
        )
        == L
    )

    chain = Model(
        m,
        name="chain",
        equations=m.getEquations(),
        problem="nlp",
        sense="min",
        objective=energy,
    )

    chain.solve()
    print("Objective Function Value:  ", round(chain.objective_value, 4))

    import math

    assert math.isclose(chain.objective_value, 5.0723, rel_tol=0.001)


if __name__ == "__main__":
    main()
