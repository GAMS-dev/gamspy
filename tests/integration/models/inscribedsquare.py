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

from gamspy import Set, Variable, Equation, Model, Container
from gamspy import Ord, Card, Sense
import math
from gamspy.math import sin, cos


def fx(t):
    return sin(t) * cos(t - t * t)


def fy(t):
    return t * sin(t)


def main():
    m = Container()

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

    obj.expr = z == a**2 + b**2
    e1x.expr = fx(t["1"]) == x
    e1y.expr = fy(t["1"]) == y
    e2x.expr = fx(t["2"]) == x + a
    e2y.expr = fy(t["2"]) == y + b
    e3x.expr = fx(t["3"]) == x - b
    e3y.expr = fy(t["3"]) == y + a
    e4x.expr = fx(t["4"]) == x + a - b
    e4y.expr = fy(t["4"]) == y + a + b

    square = Model(
        m,
        name="square",
        equations=m.getEquations(),
        problem="DNLP",
        sense=Sense.MAX,
        objective=z,
    )

    t.l[i] = -math.pi + (Ord(i) - 1) * 2 * math.pi / Card(i)
    x.l.assign = fx(t.l["1"])
    y.l.assign = fy(t.l["1"])
    a.l.assign = 1
    b.l.assign = 1

    square.solve()


if __name__ == "__main__":
    main()
