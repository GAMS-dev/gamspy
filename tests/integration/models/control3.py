"""
Optimal control problem with a nonlinear dynamic constraint and boundary
conditions solved as a General Nonlinear Programming Problem.

Divya Garg, et al., Direct trajectory optimization and costate estimation of
finite-horizon and infinite-horizon optimal control problems using a
Radau pseudospectral method. Computational optimization and Applications,
vol.49, nr. 2, June 2011, pp. 335-358.
"""
import gamspy.math as gams_math
from gamspy import Card
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Number
from gamspy import Ord
from gamspy import Parameter
from gamspy import Set
from gamspy import Sum
from gamspy import Variable


def main():
    m = Container()

    # SETS #
    n = Set(m, name="n", records=["state1"], description="states")
    k = Set(m, name="k", records=[f"t{t}" for t in range(1, 101)])
    ku = Set(m, name="ku", domain=[k], description="control horizon")
    ki = Set(m, name="ki", domain=[k], description="initial ")
    kt = Set(m, name="kt", domain=[k], description="terminal period")

    ku[k] = Number(1).where[Ord(k) < Card(k)]
    ki[k] = Number(1).where[Ord(k) == 1]
    kt[k] = ~ku[k]

    # PARAMETERS #
    rk = Parameter(m, name="rk", records=0.01, description="penalty control")
    xinit = Parameter(
        m,
        name="xinit",
        domain=[n],
        records=[("state1", 2)],
        description="initial value",
    )

    # VARIABLES #
    x = Variable(m, name="x", domain=[n, k], description="state variable")
    u = Variable(m, name="u", domain=[k], description="control variable")
    j = Variable(m, name="j", description="criterion")

    # EQUATIONS #
    cost = Equation(
        m, name="cost", type="regular", description="criterion definition"
    )
    stateq = Equation(
        m,
        name="stateq",
        type="regular",
        domain=[n, k],
        description="state equation",
    )

    cost.expr = j == 0.5 * Sum(
        [k, n], (x[n, k]) + 0.5 * Sum(ku, (u[ku]) * rk * (u[ku]))
    )

    stateq[n, k.lead(1)] = x[n, k.lead(1)] == 2 * x[n, k] + 2 * u[
        k
    ] * gams_math.sqrt(x[n, k])

    control3 = Model(
        m,
        name="control3",
        equations=m.getEquations(),
        problem="nlp",
        sense="MIN",
        objective=j,
    )

    x.l[n, k] = xinit[n]
    x.fx[n, ki] = xinit[n]
    x.fx[n, kt] = 2

    control3.solve()

    print("x:  \n", x.pivot().round(3))
    print("u:  \n", u.records.level.round(3))

    # End Control3


if __name__ == "__main__":
    main()
