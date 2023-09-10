"""
Linear Phase Lowpass Filter Design (FDESIGN)

This model finds the filter weights for a finite impulse response
(FIR) filter. We use rotated quadratic cones for the constraints.

This model is the minimax linear phase lowpass filter design from Lobo
et. al (Section 3.3) We model the nonlinear term 1/t in the model as
follows: introduce variables u,v, where v = 2 (and u = 1/t). Then 1/t
can be modeled as the quadratic cone

              ||[v, u-t]|| <= u+t,   u,t >=0

Contributed by Michael Ferris, University of Wisconsin, Madison


Lobo, M S, Vandenberghe, L, Boyd, S, and Lebret, H, Applications of
Second Order Cone Programming. Linear Algebra and its Applications,
Special Issue on Linear Algebra in Control, Signals and Image
Processing. 284 (November, 1998).

Keywords: quadratic constraint programming, second order cone programming,
          engineering, finite impulse response filter designment
"""

from gamspy import Set, Parameter, Variable, Equation, Model, Container
from gamspy import Sum, Card, Ord
import gamspy.math as gams_math
import math
from gamspy import Problem, Sense


def main():
    m = Container()

    # Set
    i = Set(m, name="i", records=[str(idx) for idx in range(0, 181)])
    omega_stop = Set(
        m,
        name="omega_stop",
        domain=[i],
        records=[str(idx) for idx in range(120, 181)],
    )
    omega_pass = Set(
        m,
        name="omega_pass",
        domain=[i],
        records=[str(idx) for idx in range(0, 91)],
    )
    k = Set(m, name="k", records=[str(idx) for idx in range(0, 11)])

    # Parameter
    beta = Parameter(m, name="beta", records=0.01)
    step = Parameter(m, name="step", records=math.pi / 180)
    n = Parameter(m, name="n", records=20)
    omega = Parameter(m, name="omega", domain=[i])
    omega[i] = (Ord(i) - 1) * step

    # Variable
    h = Variable(m, name="h", domain=[k])
    t = Variable(m, name="t")
    v2 = Variable(m, name="v2")
    v3 = Variable(m, name="v3", type="Positive")
    u = Variable(m, name="u", type="Positive")
    v = Variable(m, name="v", type="Positive")

    # Equation
    passband_up_bnds = Equation(m, name="passband_up_bnds", domain=[i])
    cone_lhs = Equation(m, name="cone_lhs")
    cone_rhs = Equation(m, name="cone_rhs")
    so = Equation(m, name="so")
    passband_lo_bnds = Equation(m, name="passband_lo_bnds", domain=[i])
    stopband_bnds = Equation(m, name="stopband_bnds", domain=[i])
    stopband_bnds2 = Equation(m, name="stopband_bnds2", domain=[i])

    passband_up_bnds[i].where[omega_pass[i]] = (
        2
        * Sum(
            k.where[Ord(k) < Card(k)],
            h[k] * gams_math.cos((Ord(k) - 1 - (n - 1) / 2) * omega[i]),
        )
        <= t
    )

    cone_rhs.expr = v2 == u - t
    cone_lhs.expr = v3 == u + t
    so.expr = v3**2 >= v**2 + v2**2

    passband_lo_bnds[i].where[omega_pass[i]] = u <= 2 * Sum(
        k.where[Ord(k) < Card(k)],
        h[k] * gams_math.cos((Ord(k) - 1 - (n - 1) / 2) * omega[i]),
    )

    stopband_bnds[i].where[omega_stop[i]] = -beta <= 2 * Sum(
        k.where[Ord(k) < Card(k)],
        h[k] * gams_math.cos((Ord(k) - 1 - (n - 1) / 2) * omega[i]),
    )

    stopband_bnds2[i].where[omega_stop[i]] = (
        2
        * Sum(
            k.where[Ord(k) < Card(k)],
            h[k] * gams_math.cos((Ord(k) - 1 - (n - 1) / 2) * omega[i]),
        )
        <= beta
    )

    t.lo.assign = 1
    v.fx.assign = 2

    fir_socp = Model(
        m,
        name="fir_socp",
        equations=m.getEquations(),
        problem=Problem.QCP,
        sense=Sense.MIN,
        objective=t,
    )
    fir_socp.solve()


if __name__ == "__main__":
    main()
