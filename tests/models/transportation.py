from gamspy import Set, Parameter, Variable, Equation, Model, Container
from gamspy import Sum


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
    z = Variable(m, name="z")

    # Equation
    cost = Equation(m, name="cost", type="eq")
    supply = Equation(m, name="supply", domain=[i], type="leq")
    demand = Equation(m, name="demand", domain=[j], type="geq")

    cost.definition = Sum((i, j), c[i, j] * x[i, j]) == z
    supply[i] = Sum(j, x[i, j]) <= a[i]
    demand[j] = Sum(i, x[i, j]) >= b[j]

    transport = Model(m, name="transport", equations="all")

    m.solve(
        transport,
        problem="LP",
        sense="min",
        objective_variable=z,
    )
    print(x.records)


if __name__ == "__main__":
    main()
