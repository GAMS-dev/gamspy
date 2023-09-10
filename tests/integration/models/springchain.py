"""
Equilibrium of System with Piecewise Linear Springs (SPRINGCHAIN)

This model finds the shape of a hanging chain consisting of
N springs and N-1 nodes. Each spring buckles under compression and each
node has a weight hanging from it. The springs are assumed
weightless. The goal is to minimize the potential energy of the
system.

We use rotated quadratic cone constraints to model the extension
of each spring.


M. Lobo, L. Vandenberghe, S. Boyd, and H. Lebret,
Applications of second-order cone programming, Linear Algebra and its
Applications, 284:193-228, November 1998, Special Issue on Linear Algebra
in Control, Signals and Image Processing.
"""

import math
from gamspy import Set, Parameter, Variable, Equation, Model, Container
from gamspy import Sum, Ord, Card
import pandas as pd
from gamspy import Problem, Sense


def main():
    cont = Container()

    N = 10
    NM1 = 9

    # Set
    n = Set(cont, name="n", records=[f"n{str(i)}" for i in range(N + 1)])

    # Data
    a_x = 0
    a_y = 0
    b_x = 2
    b_y = -1
    L0 = 2 * math.sqrt((a_x - b_x) ** 2 + (a_y - b_y) ** 2) / N
    g = 9.8
    k = 100

    m = Parameter(
        cont,
        name="m",
        domain=[n],
        records=pd.DataFrame([[f"n{i}", 1] for i in range(1, NM1 + 1)]),
    )

    # Variable
    obj = Variable(cont, name="obj")
    x = Variable(cont, name="x", domain=[n])
    y = Variable(cont, name="y", domain=[n])
    delta_x = Variable(cont, name="delta_x", domain=[n])
    delta_y = Variable(cont, name="delta_y", domain=[n])
    unit = Variable(cont, name="unit")
    t_L0 = Variable(cont, name="t_L0", domain=[n], type="Positive")
    t = Variable(cont, name="t", domain=[n], type="Positive")
    v = Variable(cont, name="v", type="Positive")

    # Equation
    pot_energy = Equation(cont, name="pot_energy")
    delta_x_eq = Equation(cont, name="delta_x_eq", domain=[n])
    delta_y_eq = Equation(cont, name="delta_y_eq", domain=[n])
    link_L0 = Equation(cont, name="link_L0", domain=[n])
    link_up = Equation(cont, name="link_up", domain=[n])
    cone_eq = Equation(cont, name="cone_eq")

    pot_energy.expr = (
        obj
        == Sum(n.where[Ord(n) > 1 & (Ord(n) < Card(n))], m[n] * g * y[n])
        + k * v
    )
    delta_x_eq[n] = delta_x[n] == x[n] - x[n.lag(1)]
    delta_y_eq[n] = delta_y[n] == y[n] - y[n.lag(1)]

    link_L0[n] = t_L0[n] == L0 + t[n]
    link_up[n].where[Ord(n) > 1] = (
        t_L0[n] ** 2 >= delta_x[n] ** 2 + delta_y[n] ** 2
    )

    cone_eq.expr = 2 * v * unit >= Sum(n.where[Ord(n) > 1], t[n] ** 2)

    spring = Model(
        cont,
        name="spring",
        equations=cont.getEquations(),
        problem=Problem.QCP,
        sense=Sense.MIN,
        objective=obj,
    )

    x.l[n] = ((Ord(n) - 1) / N) * b_x + (Ord(n) / N) * a_x
    y.l[n] = ((Ord(n) - 1) / N) * b_y + (Ord(n) / N) * a_y

    x.fx["n0"] = a_x
    y.fx["n0"] = a_y
    x.fx[f"n{N}"] = b_x
    y.fx[f"n{N}"] = b_y
    unit.fx.assign = 1

    spring.solve()


if __name__ == "__main__":
    main()
