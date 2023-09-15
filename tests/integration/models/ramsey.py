"""
An elementary Ramsey growth model

References:
Frank P. Ramsey, A Mathematical Theory of Saving, Economics Journal,
vol.38, No. 152, December 1928.

Erwin Kalvelagen, (2003) An elementary Ramsey growth model.
http://www.gams.com/~erwin/micro/growth.gms
"""
import gamspy.math as gams_math
from gamspy import Card
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Ord
from gamspy import Parameter
from gamspy import Set
from gamspy import Sum
from gamspy import Variable


def main():
    m = Container()

    # SETS #
    t = Set(
        m,
        name="t",
        records=[f"t{t}" for t in range(1, 51)],
        description="time periods",
    )
    tfirst = Set(
        m, name="tfirst", domain=[t], description="first interval (t0)"
    )
    tlast = Set(m, name="tlast", domain=[t], description="last intervat [T]")
    tnotlast = Set(
        m, name="tnotlast", domain=[t], description="all intervals but last"
    )

    tfirst[t].where[Ord(t) == 1] = True
    tlast[t].where[Ord(t) == Card(t)] = True
    tnotlast[t] = ~tlast[t]

    # SCALARS #
    rho = Parameter(m, name="rho", records=0.04, description="discount factor")
    g = Parameter(m, name="g", records=0.03, description="labor growth rate")
    delta = Parameter(
        m,
        name="delta",
        records=0.02,
        description="capital depreciation factor",
    )
    K0 = Parameter(m, name="K0", records=3.00, description="initial capital")
    I0 = Parameter(
        m, name="I0", records=0.07, description="initial investment"
    )
    C0 = Parameter(
        m, name="C0", records=0.95, description="initial consumption"
    )
    L0 = Parameter(m, name="L0", records=1.00, description="initial labor")
    b = Parameter(
        m, name="b", records=0.25, description="Cobb Douglas coefficient"
    )
    a = Parameter(m, name="a", description="Cobb Douglas coefficient")

    # PARAMETERS #
    L = Parameter(
        m, name="L", domain=[t], description="labor (production input)"
    )
    beta = Parameter(
        m,
        name="beta",
        domain=[t],
        description="weight factor for future utilities",
    )
    tval = Parameter(
        m, name="tval", domain=[t], description="numerical value of t"
    )

    tval[t] = Ord(t) - 1

    # The terminal weight beta(tlast) computation.
    beta[tnotlast[t]] = gams_math.power(1 + rho, -tval[t])
    beta[tlast[t]] = (1 / rho) * gams_math.power(1 + rho, 1 - tval[t])
    # display beta

    # Labor is determined using an exponential growth process.
    L[t] = gams_math.power(1 + g, tval[t]) * L0

    # Cobb-Douglas coefficient a computation.
    a = (C0 + I0) / (K0**b * L0 ** (1 - b))

    # VARIABLES #
    C = Variable(m, name="C", domain=[t], description="consumption")
    Y = Variable(m, name="Y", domain=[t], description="production")
    K = Variable(m, name="K", domain=[t], description="capital")
    I = Variable(m, name="I", domain=[t], description="investment")
    W = Variable(m, name="W", description="total utility")

    # EQUATIONS #
    utility = Equation(
        m, name="utility", type="regular", description="discounted utility"
    )
    production = Equation(
        m,
        name="production",
        type="regular",
        domain=[t],
        description="Cobb-Douglas production function",
    )
    allocation = Equation(
        m,
        name="allocation",
        type="regular",
        domain=[t],
        description="household choose between consumption and saving",
    )
    accumulation = Equation(
        m,
        name="accumulation",
        type="regular",
        domain=[t],
        description="capital accumulation",
    )
    final = Equation(
        m,
        name="final",
        type="regular",
        domain=[t],
        description="minimal investment in final period",
    )

    utility.expr = W == Sum(t, beta[t] * gams_math.log(C[t]))
    production[t] = Y[t] == a * (K[t] ** b) * (L[t] ** (1 - b))
    allocation[t] = Y[t] == C[t] + I[t]
    accumulation[tnotlast[t]] = K[t.lead(1)] == (1 - delta) * K[t] + I[t]
    final[tlast] = I[tlast] >= (g + delta) * K[tlast]

    # Bounds.
    K.lo[t] = 0.001
    C.lo[t] = 0.001

    # Initial conditions
    K.fx[tfirst] = K0
    I.fx[tfirst] = I0
    C.fx[tfirst] = C0

    ramsey = Model(
        m,
        name="ramsey",
        equations=m.getEquations(),
        problem="nlp",
        sense="MAX",
        objective=W,
    )

    ramsey.solve()

    print("Objective Function Value:  ", round(W.toValue(), 4))

    # Solution visualization
    # ----------------------

    rep = Parameter(m, name="rep", domain=[t, "*"])

    rep[t, "C[t]"] = C.l[t]
    rep[t, "Y[t]"] = Y.l[t]
    rep[t, "K[t]"] = K.l[t]
    rep[t, "I[t]"] = I.l[t]

    print("Solution:\n", rep.pivot().round(3))
    # End Ramsey


if __name__ == "__main__":
    main()
