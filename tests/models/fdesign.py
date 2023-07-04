from gamspy import Set, Parameter, Variable, Equation, Model, Container
from gamspy import Sum, Card, Ord
import gamspy.math as gams_math
import math


def main():
    m = Container()

    # Set
    i = Set(m, name="i", records=[str(idx) for idx in range(0, 181)])
    omega_stop = Set(
        m, name="omega_stop", domain=[i], records=[str(idx) for idx in range(120, 181)]
    )
    omega_pass = Set(
        m, name="omega_pass", domain=[i], records=[str(idx) for idx in range(0, 91)]
    )
    k = Set(m, name="k", records=[str(idx) for idx in range(0, 11)])

    # Parameter
    n2 = Parameter(m, name="n2", records=10)
    beta = Parameter(m, name="beta", records=0.01)
    step = Parameter(m, name="step", records=math.pi / 180)
    n = Parameter(m, name="n", records=20)
    omega_s = Parameter(m, name="omega_s", records=2 * math.pi / 3)
    omega_p = Parameter(m, name="omega_p", records=math.pi / 2)
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
    passband_up_bnds = Equation(m, name="passband_up_bnds", domain=[i], type="eq")
    cone_lhs = Equation(m, name="cone_lhs", type="eq")
    cone_rhs = Equation(m, name="cone_rhs", type="eq")
    so = Equation(m, name="so", type="eq")
    passband_lo_bnds = Equation(m, name="passband_lo_bnds", domain=[i], type="eq")
    stopband_bnds = Equation(m, name="stopband_bnds", domain=[i], type="eq")
    stopband_bnds2 = Equation(m, name="stopband_bnds2", domain=[i], type="eq")

    passband_up_bnds[i].where[omega_pass[i]] = (
        2
        * Sum(
            k.where[Ord(k) < Card(k)],
            h[k] * gams_math.cos((Ord(k) - 1 - (n - 1) / 2) * omega[i]),
        )
        <= t
    )

    cone_rhs.definition = v2 == u - t
    cone_lhs.definition = v3 == u + t
    so.definition = v3**2 >= v**2 + v2**2

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

    fir_socp = Model(m, name="fir_socp", equations="all")
    m.solve(fir_socp, problem="QCP", sense="min", objective_variable=t)


if __name__ == "__main__":
    main()
