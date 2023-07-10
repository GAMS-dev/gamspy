from gamspy import Set, Parameter, Variable, Equation, Model, Container
from gamspy import Sum, Ord, Card
import pandas as pd


def main():
    m = Container()

    # Set
    t = Set(m, name="t", records=["period-1", "period-2", "period-3"])

    # Data
    load = Parameter(
        m,
        name="load",
        domain=[t],
        records=pd.DataFrame(
            [["period-1", 400], ["period-2", 900], ["period-3", 700]]
        ),
    )
    initlev = Parameter(
        m,
        name="initlev",
        domain=[t],
        records=pd.DataFrame([["period-1", 3000]]),
    )

    # Variable
    status = Variable(m, name="status", domain=[t], type="Binary")
    poil = Variable(m, name="poil", domain=[t])
    others = Variable(m, name="others", domain=[t])
    oil = Variable(m, name="oil", domain=[t], type="Positive")
    volume = Variable(m, name="volume", domain=[t], type="Positive")
    cost = Variable(m, name="cost")

    volume.up[t] = 4000
    volume.lo[t].where[Ord(t) == Card(t)] = 2000

    others.lo[t] = 50
    others.up[t] = 700

    # Equation
    costfn = Equation(m, type="eq", name="costfn")
    lowoil = Equation(m, type="geq", name="lowoil", domain=[t])
    maxoil = Equation(m, type="leq", name="maxoil", domain=[t])
    floweq = Equation(m, type="eq", name="floweq", domain=[t])
    demcons = Equation(m, type="eq", name="demcons", domain=[t])
    oileq = Equation(m, type="geq", name="oileq", domain=[t])

    costfn.definition = cost == Sum(
        t, 300 + 6 * others[t] + 0.0025 * (others[t] ** 2)
    )
    lowoil[t] = poil[t] >= 100 * status[t]
    maxoil[t] = poil[t] <= 500 * status[t]
    floweq[t] = volume[t] == volume[t.lag(1)] + 500 - oil[t] + initlev[t]
    oileq[t] = oil[t] == 50 * status[t] + poil[t] + 0.005 * (poil[t] ** 2)
    demcons[t] = poil[t] + others[t] >= load[t]

    model = Model(m, name="ucom", equations="all")
    poil.l[t] = 100

    m.solve(model, problem="MINLP", sense="min", objective_variable=cost)


if __name__ == "__main__":
    main()
