"""
Inscribed Square Problem (INSCRIBEDSQUARE)

The inscribed square problem, also known as the square peg problem or
the Toeplitz' conjecture, is an unsolved question in geometry:

  Does every plane simple closed curve contain all four vertices of
  some square?


This is true if the curve is convex or piecewise smooth and in other
special cases. The problem was proposed by Otto Toeplitz in 1911.
See also https://en.wikipedia.org/wiki/Inscribed_square_problem

This model computes a square of maximal area for a given curve.

Use options --fx and --fy to specify the x and y coordinates of a closed
curve as function in variable t, where t ranges from -pi to pi.
Use option --gnuplot 1 to enable plotting the curve and computed square
with gnuplot (if available and a feasible solution has been found).

Contributor: Benjamin Mueller and Felipe Serrano
"""
from __future__ import annotations

import math
import os

from gamspy import Card
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Ord
from gamspy import Sense
from gamspy import Set
from gamspy import Variable
from gamspy.math import cos
from gamspy.math import sin


def fx(t):
    return sin(t) * cos(t - t * t)


def fy(t):
    return t * sin(t)


def main():
    m = Container(delayed_execution=int(os.getenv("DELAYED_EXECUTION", False)))

    # Set
    i = Set(m, name="i", records=["1", "2", "3", "4"])

    # Variable
    z = Variable(m, name="z")
    t = Variable(m, name="t", domain=[i])
    x = Variable(m, name="x")
    y = Variable(m, name="y")
    a = Variable(m, name="a", type="Positive")
    b = Variable(m, name="b", type="Positive")

    t.lo[i] = -math.pi
    t.up[i] = math.pi

    # Equation
    obj = Equation(m, name="obj")
    e1x = Equation(m, name="e1x")
    e1y = Equation(m, name="e1y")
    e2x = Equation(m, name="e2x")
    e2y = Equation(m, name="e2y")
    e3x = Equation(m, name="e3x")
    e3y = Equation(m, name="e3y")
    e4x = Equation(m, name="e4x")
    e4y = Equation(m, name="e4y")

    obj[...] = z == a**2 + b**2
    e1x[...] = fx(t["1"]) == x
    e1y[...] = fy(t["1"]) == y
    e2x[...] = fx(t["2"]) == x + a
    e2y[...] = fy(t["2"]) == y + b
    e3x[...] = fx(t["3"]) == x - b
    e3y[...] = fy(t["3"]) == y + a
    e4x[...] = fx(t["4"]) == x + a - b
    e4y[...] = fy(t["4"]) == y + a + b

    square = Model(
        m,
        name="square",
        equations=m.getEquations(),
        problem="DNLP",
        sense=Sense.MAX,
        objective=z,
    )

    t.l[i] = -math.pi + (Ord(i) - 1) * 2 * math.pi / Card(i)
    x.l[...] = fx(t.l["1"])
    y.l[...] = fy(t.l["1"])
    a.l[...] = 1
    b.l[...] = 1

    square.solve()

    assert math.isclose(square.objective_value, 1.6009, rel_tol=0.001)


if __name__ == "__main__":
    main()
