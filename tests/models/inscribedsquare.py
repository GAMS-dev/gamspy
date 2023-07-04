from gamspy import Set, Variable, Equation, Model, Container
from gamspy import Ord, Card
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
    obj = Equation(m, name="obj", type="eq")
    e1x = Equation(m, name="e1x", type="eq")
    e1y = Equation(m, name="e1y", type="eq")
    e2x = Equation(m, name="e2x", type="eq")
    e2y = Equation(m, name="e2y", type="eq")
    e3x = Equation(m, name="e3x", type="eq")
    e3y = Equation(m, name="e3y", type="eq")
    e4x = Equation(m, name="e4x", type="eq")
    e4y = Equation(m, name="e4y", type="eq")

    obj.definition = z == a**2 + b**2
    e1x.definition = fx(t["1"]) == x
    e1y.definition = fy(t["1"]) == y
    e2x.definition = fx(t["2"]) == x + a
    e2y.definition = fy(t["2"]) == y + b
    e3x.definition = fx(t["3"]) == x - b
    e3y.definition = fy(t["3"]) == y + a
    e4x.definition = fx(t["4"]) == x + a - b
    e4y.definition = fy(t["4"]) == y + a + b

    square = Model(m, name="square", equations="all")

    t.l[i] = -math.pi + (Ord(i) - 1) * 2 * math.pi / Card(i)
    x.l.assign = fx(t.l["1"])
    y.l.assign = fy(t.l["1"])
    a.l.assign = 1
    b.l.assign = 1

    m.solve(square, problem="DNLP", sense="max", objective_variable=z)


if __name__ == "__main__":
    main()
