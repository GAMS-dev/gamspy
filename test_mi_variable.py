from gamspy import Set, Parameter, Variable, Equation, Model, Container
from gamspy import Sum, Sense


def main():
    m = Container(
        working_directory=".",
    )

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
    bmult = Parameter(m, name="bmult", records=1)
    c[i, j] = 90 * d[i, j] / 1000

    # Variable
    x = Variable(m, name="x", domain=[i, j], type="Positive")
    z = Variable(m, name="z")

    # Equation
    cost = Equation(m, name="cost")
    supply = Equation(m, name="supply", domain=[i])
    demand = Equation(m, name="demand", domain=[j])

    cost.definition = z == Sum((i, j), c[i, j] * x[i, j])
    supply[i] = Sum(j, x[i, j]) <= a[i]
    demand[j] = Sum(i, x[i, j]) >= bmult * b[j]

    transport = Model(
        m,
        name="transport",
        equations=m.getEquations(),
        problem="LP",
        sense=Sense.MIN,
        objective=z,
    )

    transport.freeze(modifiables=[x.up])
    transport.solve()
    print(z.records)
    m["x_up"].records.at[1, "value"] = 0
    transport.solve()
    print(z.records)


if __name__ == "__main__":
    main()
