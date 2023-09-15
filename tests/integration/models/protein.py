"""
Optimal production of secreted protein in a fed-batch reactor.

Park, S., Ramirez, W.F., Optimal production of secreted protein in fed-batch
reactors. A.I.Ch.E. Journal, 34, 1988, pp.1550-1558.
"""
import gamspy.math as gams_math
from gamspy import Alias
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Parameter
from gamspy import Set
from gamspy import Variable


def main():
    m = Container()

    n = 500

    # SET #
    nh = Set(
        m,
        name="nh",
        records=[str(i) for i in range(n + 1)],
        description="Number of subintervals",
    )

    # ALIAS #
    k = Alias(m, name="k", alias_with=nh)

    # SCALARS #
    tf = Parameter(m, name="tf", records=10, description="final time")
    x1_0 = Parameter(
        m, name="x1_0", records=1, description="initial value for x1"
    )
    x2_0 = Parameter(
        m, name="x2_0", records=5, description="initial value for x2"
    )
    x3_0 = Parameter(
        m, name="x3_0", records=0, description="initial value for x3"
    )
    x4_0 = Parameter(
        m, name="x4_0", records=0, description="initial value for x4"
    )
    x5_0 = Parameter(
        m, name="x5_0", records=1, description="initial value for x5"
    )
    h = Parameter(m, name="h")
    h.assign = tf / n

    # VARIABLES #
    x1 = Variable(m, name="x1", domain=[nh])
    x2 = Variable(m, name="x2", domain=[nh])
    x3 = Variable(m, name="x3", domain=[nh])
    x4 = Variable(m, name="x4", domain=[nh])
    x5 = Variable(m, name="x5", domain=[nh])
    u = Variable(m, name="u", domain=[nh], description="control variable")
    a1 = Variable(m, name="a1", domain=[nh])
    a2 = Variable(m, name="a2", domain=[nh])
    a3 = Variable(m, name="a3", domain=[nh])
    obj = Variable(m, name="obj", description="criterion")

    # EQUATIONS #
    eobj = Equation(
        m, name="eobj", type="regular", description="criterion definition"
    )
    state1 = Equation(
        m,
        name="state1",
        type="regular",
        domain=[nh],
        description="state equation 1",
    )
    state2 = Equation(
        m,
        name="state2",
        type="regular",
        domain=[nh],
        description="state equation 2",
    )
    state3 = Equation(
        m,
        name="state3",
        type="regular",
        domain=[nh],
        description="state equation 3",
    )
    state4 = Equation(
        m,
        name="state4",
        type="regular",
        domain=[nh],
        description="state equation 4",
    )
    state5 = Equation(
        m,
        name="state5",
        type="regular",
        domain=[nh],
        description="state equation 5",
    )
    ea1 = Equation(m, name="ea1", type="regular", domain=[nh])
    ea2 = Equation(m, name="ea2", type="regular", domain=[nh])
    ea3 = Equation(m, name="ea3", type="regular", domain=[nh])

    eobj.expr = obj == x4[str(n)] * x5[str(n)]

    state1[nh[k.lead(1)]] = x1[k.lead(1)] == (
        x1[k]
        + (h / 2)
        * (
            a1[k] * x1[k]
            - u[k] * x1[k] / x5[k]
            + a1[k.lead(1)] * x1[k.lead(1)]
            - u[k.lead(1)] * x1[k.lead(1)] / x5[k.lead(1)]
        )
    )

    state2[nh[k.lead(1)]] = x2[k.lead(1)] == (
        x2[k]
        + (h / 2)
        * (
            -7.3 * a1[k] * x1[k]
            - u[k] * (x2[k] - 20) / x5[k]
            - 7.3 * a1[k.lead(1)] * x1[k.lead(1)]
            - u[k.lead(1)] * (x2[k.lead(1)] - 20) / x5[k.lead(1)]
        )
    )

    state3[nh[k.lead(1)]] = x3[k.lead(1)] == (
        x3[k]
        + (h / 2)
        * (
            a2[k] * x1[k]
            - u[k] * x3[k] / x5[k]
            + a2[k.lead(1)] * x1[k.lead(1)]
            - u[k.lead(1)] * x3[k.lead(1)] / x5[k.lead(1)]
        )
    )

    state4[nh[k.lead(1)]] = x4[k.lead(1)] == (
        x4[k]
        + (h / 2)
        * (
            a3[k] * (x3[k] - x4[k])
            - u[k] * x4[k] / x5[k]
            + a3[k.lead(1)] * (x3[k.lead(1)] - x4[k.lead(1)])
            - u[k.lead(1)] * x4[k.lead(1)] / x5[k.lead(1)]
        )
    )

    state5[nh[k.lead(1)]] = x5[k.lead(1)] == x5[k] + (h / 2) * (
        u[k] + u[k.lead(1)]
    )

    ea1[nh[k]] = a1[k] == 21.87 * x2[k] / ((x2[k] + 0.4) * (x2[k] + 62.5))
    ea2[nh[k]] = a2[k] == (x2[k] * gams_math.exp(-5 * x2[k])) / (0.1 + x2[k])
    ea3[nh[k]] = a3[k] == 4.75 * a1[k] / (0.12 + a1[k])

    # Initial point
    x1.l[nh] = 1.0
    x2.l[nh] = 5.0
    x3.l[nh] = 0.0
    x4.l[nh] = 0.0
    x5.l[nh] = 1.0
    u.l[nh] = 0.0

    x1.fx["0"] = x1_0
    x2.fx["0"] = x2_0
    x3.fx["0"] = x3_0
    x4.fx["0"] = x4_0
    x5.fx["0"] = x5_0

    # Bounds
    u.lo[nh] = 0.0
    u.up[nh] = 5

    protein = Model(
        m,
        name="protein",
        equations=m.getEquations(),
        problem="nlp",
        sense="max",
        objective=obj,
    )

    m.addOptions({"reslim": 60000, "iterlim": 80000})
    protein.solve()

    print("Objective Function Value:  ", round(obj.toValue(), 4), "\n")

    # REPORTING PARAMETER #
    rep = Parameter(m, name="rep", domain=[nh, "*"])
    rep[nh, "x1"] = x1.l[nh]
    rep[nh, "x2"] = x2.l[nh]
    rep[nh, "x3"] = x3.l[nh]
    rep[nh, "x4"] = x4.l[nh]
    rep[nh, "x5"] = x5.l[nh]
    rep[nh, "u"] = u.l[nh]

    rep.pivot().round(3).to_csv("protein_report.csv")

    # End of protein


if __name__ == "__main__":
    main()
