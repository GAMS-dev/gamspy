from gamspy import Set, Parameter, Variable, Equation, Model, Container
from gamspy import Sum
import pandas as pd
from gamspy.math import uniform


def main():
    m = Container()

    # Prepare data
    forces = pd.DataFrame(
        [
            ["j1", "k1", 0.0008],
            ["j1", "k2", 1.0668],
            ["j1", "k3", 0.2944],
            ["j2", "k1", 0.0003],
            ["j2", "k2", 0.0593],
            ["j2", "k3", -1.3362],
            ["j3", "k1", -0.0006],
            ["j3", "k2", -0.0956],
            ["j3", "k3", 0.7143],
            ["j4", "k1", -1.0003],
            ["j4", "k2", -0.8323],
            ["j4", "k3", 1.6236],
        ]
    )

    stiff_data = pd.DataFrame(
        [
            ["j1", "i1", 1.0],
            ["j1", "i2", 0],
            ["j1", "i3", 0.5],
            ["j1", "i4", 0],
            ["j1", "i5", 0],
            ["j2", "i1", 0],
            ["j2", "i2", 0],
            ["j2", "i3", -0.5],
            ["j2", "i4", -1.0],
            ["j2", "i5", 0],
            ["j3", "i1", 0],
            ["j3", "i2", 0.5],
            ["j3", "i3", 0],
            ["j3", "i4", 0],
            ["j3", "i5", 1.0],
            ["j4", "i1", 0],
            ["j4", "i2", 0.5],
            ["j4", "i3", 0],
            ["j4", "i4", 1.0],
            ["j4", "i5", 0],
        ]
    )

    # Set
    i = Set(m, name="i", records=[f"i{idx}" for idx in range(1, 6)])
    j = Set(m, name="j", records=[f"j{idx}" for idx in range(1, 5)])
    k = Set(m, name="k", records=[f"k{idx}" for idx in range(1, 4)])

    # Data
    f = Parameter(m, name="f", domain=[j, k], records=forces)
    b = Parameter(m, name="b", domain=[j, i], records=stiff_data)

    max_volume = 10

    # Variable
    tau = Variable(m, name="tau")
    s = Variable(m, name="s", domain=[i, k])
    tk = Variable(m, name="tk", domain=[i, k], type="Positive")
    t = Variable(m, name="t", domain=[i], type="Positive")
    sigma = Variable(m, name="sigma", domain=[i, k], type="Positive")

    # Equation
    volumeeq = Equation(m, type="geq", name="volumeeq", domain=[i, k])
    deftk = Equation(m, type="eq", name="deftk", domain=[i, k])
    reseq = Equation(m, type="leq", name="reseq", domain=[k])
    trusscomp = Equation(m, type="leq", name="trusscomp")
    stiffness = Equation(m, type="eq", name="stifness", domain=[j, k])

    volumeeq[i, k] = 2 * tk[i, k] * sigma[i, k] >= s[i, k] ** 2
    deftk[i, k] = tk[i, k] == t[i]
    reseq[k] = Sum(i, sigma[i, k]) <= tau
    trusscomp.definition = Sum(i, t[i]) <= max_volume
    stiffness[j, k] = Sum(i, s[i, k] * b[j, i]) == f[j, k]

    truss = Model(m, name="truss", equations="all")
    sigma.l[i, k] = uniform(0.1, 1)
    m.solve(truss, problem="QCP", sense="min", objective_variable=tau)

    f[j, "k2"] = 0
    f[j, "k3"] = 0

    m.solve(truss, problem="QCP", sense="min", objective_variable=tau)


if __name__ == "__main__":
    main()
