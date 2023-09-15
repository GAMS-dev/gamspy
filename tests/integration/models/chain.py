"""
   Find the chain (of uniform density) of length L suspended between two
   points with minimal potential energy.

   This model is from the COPS benchmarking suite.
   See http://www-unix.mcs.anl.gov/~more/cops/.

   The number of intervals for the discretization can be specified using
   the command line parameter --nh. COPS performance tests have been
   reported for nh = 50, 100, 200, 400

   Tested with nh=3000, 4000, 5000     May 26, 2005

   References:
   Neculai Andrei, "Models, Test Problems and Applications for
   Mathematical Programming". Technical Press, Bucharest, 2003.
   Application A7, page 350.

   Dolan, E D, and More, J J, Benchmarking Optimization Software with COPS.
   Tech. rep., Mathematics and Computer Science Division, 2000.

   Cesari, L, Optimization - Theory and Applications. Springer Verlag, 1983.
"""
import sys

import gamspy.math as gams_math
from gamspy import Alias
from gamspy import Card
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Ord
from gamspy import Parameter
from gamspy import Set
from gamspy import Sum
from gamspy import Variable


def sqr(x):
    return gams_math.power(x, 2)


def main():
    m = Container()

    n_rec = int(sys.argv[1]) if len(sys.argv) > 1 else 400

    # SETS #
    nh = Set(m, name="nh", records=[f"i{i}" for i in range(n_rec + 1)])

    # ALIASES #
    i = Alias(m, name="i", alias_with=nh)

    # SCALARS #
    L = Parameter(
        m, name="L", records=4, description="length of the suspended chain"
    )
    a = Parameter(
        m, name="a", records=1, description="height of the chain at t=0 (left)"
    )
    b = Parameter(
        m, name="b", records=3, description="height of the chain at t=1 (left)"
    )
    tf = Parameter(
        m, name="tf", records=1, description="ODEs defined in [0 tf]"
    )
    h = Parameter(m, name="h", description="uniform interval length")
    n = Parameter(m, name="n", description="number of subintervals")
    tmin = Parameter(m, name="tmin")

    if b.toValue() > a.toValue():
        tmin.assign = 0.25
    else:
        tmin.assign = 0.75

    n.assign = Card(nh) - 1
    h.assign = tf / n

    # VARIABLES #
    x = Variable(m, name="x", domain=[i], description="height of the chain")
    u = Variable(m, name="u", domain=[i], description="derivative of x")
    energy = Variable(m, name="energy", description="potential energy")

    x.fx["i0"] = a
    x.fx[f"i{n_rec}"] = b

    x.l[i] = (
        4
        * gams_math.abs(b - a)
        * ((Ord(i) - 1) / n)
        * (0.5 * ((Ord(i) - 1) / n) - tmin)
        + a
    )
    u.l[i] = 4 * gams_math.abs(b - a) * (((Ord(i) - 1) / n) - tmin)

    # EQUATIONS #
    obj = Equation(m, name="obj", type="regular")
    x_eqn = Equation(m, name="x_eqn", type="regular", domain=[i])
    length_eqn = Equation(m, name="length_eqn", type="regular")

    obj.expr = energy == 0.5 * h * Sum(
        nh[i.lead(1)],
        x[i] * gams_math.sqrt(1 + sqr(u[i]))
        + x[i.lead(1)] * gams_math.sqrt(1 + sqr(u[i.lead(1)])),
    )

    x_eqn[i.lead(1)] = x[i.lead(1)] == x[i] + 0.5 * h * (u[i] + u[i.lead(1)])

    length_eqn.expr = (
        0.5
        * h
        * Sum(
            nh[i.lead(1)],
            gams_math.sqrt(1 + sqr(u[i]))
            + gams_math.sqrt(1 + sqr(u[i.lead(1)])),
        )
        == L
    )

    chain = Model(
        m,
        name="chain",
        equations=m.getEquations(),
        problem="nlp",
        sense="min",
        objective=energy,
    )

    chain.solve()
    print("Objective Function Value:  ", round(energy.toValue(), 4))

    # End Hanging Chain


if __name__ == "__main__":
    main()
