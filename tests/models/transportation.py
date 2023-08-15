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

from gamspy import Set, Parameter, Variable, Equation, Model, Container
from gamspy import Sum, Sense
import sys


def main():
    m = Container()

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

    # Data
    a = Parameter(m, name="a", domain=[i], records=capacities)
    b = Parameter(m, name="b", domain=[j], records=demands)
    d = Parameter(m, name="d", domain=[i, j], records=distances)
    c = Parameter(m, name="c", domain=[i, j])
    c[i, j] = 90 * d[i, j] / 1000

    # Variable
    x = Variable(m, name="x", domain=[i, j], type="Positive")

    # Equation
    supply = Equation(m, name="supply", domain=[i])
    demand = Equation(m, name="demand", domain=[j])

    supply[i] = Sum(j, x[i, j]) <= a[i]
    demand[j] = Sum(i, x[i, j]) >= b[j]

    transport = Model(
        m,
        name="transport",
        equations=m.getEquations(),
        problem="LP",
        sense=Sense.MIN,
        objective=Sum((i, j), c[i, j] * x[i, j]),
    )
    transport.solve(output=sys.stdout)

    print(x.records)
    print(transport.objective_value)
    print(transport.status)


if __name__ == "__main__":
    main()
