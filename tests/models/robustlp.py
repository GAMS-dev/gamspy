from gamspy import Alias, Set, Parameter, Variable, Equation, Model, Container
from gamspy import Sum
from gamspy.math import uniform


def main():
    m = Container()

    mu = 1e-2

    # Set
    i = Set(m, name="i", records=[f"{idx}" for idx in range(1, 8)])
    j = Set(m, name="j", records=[f"{idx}" for idx in range(1, 5)])

    # Data
    a = Parameter(m, name="a", domain=[i, j])
    b = Parameter(m, name="b", domain=[i])
    c = Parameter(m, name="c", domain=[j])
    b[i] = 1
    c[j] = -1

    a[i, j] = uniform(0, 1)

    # Variable
    x = Variable(m, name="x", domain=[j])
    obj = Variable(m, name="obj")

    # Equation
    defobj = Equation(m, name="defobj", type="eq")
    cons = Equation(m, name="cons", domain=[i], type="leq")

    defobj.definition = obj == Sum(j, c[j] * x[j])
    cons[i] = Sum(j, a[i, j] * x[j]) <= b[i]

    lpmod = Model(m, name="lpmod", equations=[defobj, cons])
    m.solve(lpmod, problem="LP", sense="min", objective_variable=obj)

    results = Parameter(m, name="results", domain=["*", "*"])
    results["lp", j] = x.l[j]
    results["lp", "obj"] = obj.l

    lmbda = Variable(m, name="lambda", domain=[j])
    gamma = Variable(m, name="gamma", domain=[j])

    lpcons = Equation(m, name="lpcons", domain=[i], type="leq")
    defdual = Equation(m, name="defdual", domain=[j], type="eq")

    lpcons[i] = (
        mu * Sum(j, lmbda[j] + gamma[j]) + Sum(j, a[i, j] * x[j]) <= b[i]
    )
    defdual[j] = lmbda[j] - gamma[j] == x[j]

    lproblp = Model(m, name="lproblp", equations=[defobj, lpcons, defdual])
    m.solve(lproblp, problem="LP", sense="min", objective_variable=obj)

    results["roblp", j] = x.l[j]
    results["roblp", "obj"] = obj.l

    k = Alias(m, name="k", alias_with=j)
    p = Parameter(m, name="p", domain=[i, j, k])
    p[i, j, j] = mu

    y = Variable(m, name="y", domain=[i])
    v = Variable(m, name="v", domain=[i, k])

    defrhs = Equation(m, name="defrhs", domain=[i], type="eq")
    defv = Equation(m, name="defv", domain=[i, k], type="eq")
    socpqcpcons = Equation(m, name="socpqcpcons", domain=[i], type="geq")

    defrhs[i] = y[i] == b[i] - Sum(j, a[i, j] * x[j])
    defv[i, k] = v[i, k] == Sum(j, p[i, j, k] * x[j])
    socpqcpcons[i] = y[i] ** 2 >= Sum(k, v[i, k] ** 2)

    roblpqcp = Model(
        m, name="roblpqcp", equations=[defobj, socpqcpcons, defrhs, defv]
    )

    y.lo[i] = 0

    m.solve(roblpqcp, problem="QCP", sense="min", objective_variable=obj)
    results["qcp", j] = x.l[j]
    results["qcp", "obj"] = obj.l

    print(results.records)


if __name__ == "__main__":
    main()
