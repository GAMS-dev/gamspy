"""
Truss Toplogy Design with Multiple Loads (TRUSSM)

A structure of n linear elastic bars connects a set of m nodes.
The task is to size the bars, i.e. determine t(i), the volume
of the bars, that yield the stiffest truss subject to constraints
such as total weight limit and k different (nonsimultaneous) loading
scenarios to be satisfied. For example, the different load scenarios
for a bridge could include rush hour traffic, night traffic, earthquake
and side wind.

The model is given as a conic program. The cone implementation comes
from Ben-Tal and Nemirovski.

Suppose we have a truss of n bars and m nodes. Now consider a set of
k fixed externally applied nodal forces f(k)=[f1, .., fn].

Let d_i denote the small node displacement resulting from the force on
each node i. The objective is to maximize the stiffness of the truss,
which is equivalent to minimizing the elastic stored energy 0.5*f^T*d,
subject to some maximum volume restriction on the truss.

Using the formulation given in Ben-Tal and Nemirovski (2001), we can
model this as the second order cone problem:

           minimize      tau
           subject to
                         sum(i, t(i)) <= maxvolume

                         s(i,k)^2 <= 2*t(i)*sigma(i,k)
                         sum(i, sigma(i,k)) <= tau
                         sum(i,k) s(i,k)*b(i)) <= f(k)

The first constraint is the material volume limitation. The latter 3
constraints and the objective are the compliance constraints, which are
equivalent to minimization of the elastic potential energy under a given
load.


A. Ben-Tal and A. Nemirovski, Lectures on Modern Convex Optimization:
Analysis, Algorithms, and Engineering Applications, MPS/SIAM Series
on Optimization, SIAM Press, 2001.

M.S. Lobo, L. Vandenberghe, S. Boyd, and H. Lebret, "Applications of
Second-order Cone Programming", Linear Algebra and its Applications,
Special Issue on Linear Algebra in Control, Signals and Image Processing.
284 (1998) 193-228.
"""

from gamspy import Set, Parameter, Variable, Equation, Model, Container
from gamspy import Sum
import pandas as pd
from gamspy.math import uniform
from gamspy import Problem, Sense


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
    volumeeq = Equation(m, name="volumeeq", domain=[i, k])
    deftk = Equation(m, name="deftk", domain=[i, k])
    reseq = Equation(m, name="reseq", domain=[k])
    trusscomp = Equation(m, name="trusscomp")
    stiffness = Equation(m, name="stifness", domain=[j, k])

    volumeeq[i, k] = 2 * tk[i, k] * sigma[i, k] >= s[i, k] ** 2
    deftk[i, k] = tk[i, k] == t[i]
    reseq[k] = Sum(i, sigma[i, k]) <= tau
    trusscomp.expr = Sum(i, t[i]) <= max_volume
    stiffness[j, k] = Sum(i, s[i, k] * b[j, i]) == f[j, k]

    truss = Model(
        m,
        name="truss",
        equations=m.getEquations(),
        problem=Problem.QCP,
        sense=Sense.MIN,
        objective=tau,
    )
    sigma.l[i, k] = uniform(0.1, 1)
    truss.solve()

    f[j, "k2"] = 0
    f[j, "k3"] = 0

    truss.solve()


if __name__ == "__main__":
    main()
