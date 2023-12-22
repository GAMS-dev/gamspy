"""
A Transportation Problem (TRNSPORT)

This problem finds a least cost shipping schedule that meets
requirements at markets and supplies at factories.


Dantzig, G B, Chapter 3.3. In Linear Programming and Extensions.
Princeton University Press, Princeton, New Jersey, 1963.

This formulation is described in detail in:
Rosenthal, R E, Chapter 2: A GAMS Tutorial. In GAMS: A User's Guide.
The Scientific Press, Redwood City, California, 1988.

The line numbers will not match those in the book because of these
comments.

Keywords: linear programming, transportation problem, scheduling
"""
from __future__ import annotations

from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Parameter
from gamspy import Sense
from gamspy import Set
from gamspy import Sum
from gamspy import Variable


def main():
    m = Container(delayed_execution=True, working_directory=".")

    # Prepare data
    distances = [
        ["seattle", "new-york", 2.5],
        ["seattle", "chicago", 1.7],
        ["seattle", "topeka", 1.8],
        ["san-diego", "new-york", 2.5],
        ["san-diego", "chicago", 1.8],
        ["san-diego", "topeka", 1.4],
    ]

    capacities = [["seattle", 350], ["san-diego", 600]]
    demands = [["new-york", 325], ["chicago", 300], ["topeka", 275]]

    # Set
    i = Set(m, name="i", records=["seattle", "san-diego"])
    j = Set(m, name="j", records=["new-york", "chicago", "topeka"])
    _ = Set(
        m,
        name="model_type",
        records=["lp"],
        is_singleton=True,
        is_miro_input=True,
    )

    # Data
    a = Parameter(m, name="a", domain=[i], records=capacities)
    b = Parameter(m, name="b", domain=[j], records=demands)
    d = Parameter(
        m, name="d", domain=[i, j], records=distances, is_miro_input=True
    )
    c = Parameter(m, name="c", domain=[i, j])
    f = Parameter(m, name="f", records=90, is_miro_input=True)
    c[i, j] = f * d[i, j] / 1000

    # Variable
    x = Variable(
        m, name="x", domain=[i, j], type="Positive", is_miro_output=True
    )
    z = Variable(m, name="z", is_miro_output=True)

    # Equation
    supply = Equation(m, name="supply", domain=[i])
    demand = Equation(m, name="demand", domain=[j])
    cost = Equation(m, name="cost")

    supply[i] = Sum(j, x[i, j]) <= a[i]
    demand[j] = Sum(i, x[i, j]) >= b[j]
    cost[...] = Sum((i, j), c[i, j] * x[i, j]) == z

    transport = Model(
        m,
        name="my_model",
        equations=m.getEquations(),
        problem="LP",
        sense=Sense.MIN,
        objective=z,
    )
    transport.solve()
    print(transport.objective_value)


if __name__ == "__main__":
    main()
