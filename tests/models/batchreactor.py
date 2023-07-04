from gamspy import Alias, Set, Parameter, Variable, Equation, Model, Container
import gamspy.math as gams_math


def main():
    m = Container()

    # Set
    nh = Set(m, name="nh", records=[str(idx) for idx in range(0, 101)])
    k = Alias(m, name="k", alias_with=nh)

    # Data
    tf = Parameter(m, name="tf", records=100)
    ca_0 = Parameter(m, name="ca_0", records=1.0)
    cb_0 = Parameter(m, name="cb_0", records=0.0)
    h = Parameter(m, name="h", records=1)

    # Variable
    ca = Variable(m, name="ca", domain=[nh])
    cb = Variable(m, name="cb", domain=[nh])
    t = Variable(m, name="t", domain=[nh])
    k1 = Variable(m, name="k1", domain=[nh])
    k2 = Variable(m, name="k2", domain=[nh])
    obj = Variable(m, name="obj")

    # Equation
    eobj = Equation(m, type="eq", name="eobj")
    state1 = Equation(m, type="eq", domain=[nh], name="state1")
    state2 = Equation(m, type="eq", domain=[nh], name="state2")
    ek1 = Equation(m, type="eq", domain=[nh], name="ek1")
    ek2 = Equation(m, type="eq", domain=[nh], name="ek2")

    eobj.definition = obj == cb["100"]

    ek1[nh[k]] = k1[k] == 4000 * gams_math.exp(-2500 / t[k])
    ek2[nh[k]] = k2[k] == 620000 * gams_math.exp(-5000 / t[k])

    state1[nh[k.lead(1, "linear")]] = ca[k.lead(1, "linear")] == ca[k] + (h / 2) * (
        -k1[k] * ca[k] * ca[k]
        - k1[k.lead(1, "linear")] * ca[k.lead(1, "linear")] * ca[k.lead(1, "linear")]
    )

    state2[nh[k.lead(1, "linear")]] = cb[k.lead(1, "linear")] == cb[k] + (h / 2) * (
        k1[k] * ca[k] * ca[k]
        - k2[k] * cb[k]
        + k1[k.lead(1, "linear")] * ca[k.lead(1, "linear")] * ca[k.lead(1, "linear")]
        - k2[k.lead(1, "linear")] * cb[k.lead(1, "linear")]
    )

    ca.l[nh] = 1.0
    cb.l[nh] = 0.0

    ca.fx["0"] = ca_0
    cb.fx["0"] = cb_0

    t.lo[nh] = 110
    t.up[nh] = 280

    batchReactor = Model(m, name="batchReactor", equations="all")
    m.solve(batchReactor, problem="NLP", sense="max", objective_variable=obj)


if __name__ == "__main__":
    main()
