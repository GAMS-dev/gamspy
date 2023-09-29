"""
Fuel Scheduling and Unit Commitment Problem (FUEL)


Fuel scheduling and unit commitment addresses the problem of
fuel supply to plants and determining on/off status of units
simultaneously to minimize total operating cost.
The present problem: there are two generating units to
meet a total load over a 6-hour period. One of the unit is oil-based
and has to simultaneously meet the storage requirements, flow rates
etc. There are limits on the generation levels for both the units.


Wood, A J, and Wollenberg, B F, Example Problem 4e. In Power Generation,
Operation and Control. John Wiley and Sons, 1984, pp. 85-88.

Keywords: mixed integer nonlinear programming, scheduling, engineering, power
          generation, unit commitment problem
"""
import pandas as pd

from gamspy import Card
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Ord
from gamspy import Parameter
from gamspy import Sense
from gamspy import Set
from gamspy import Sum
from gamspy import Variable


def main():
    m = Container(delayed_execution=True)

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
    costfn = Equation(m, name="costfn")
    lowoil = Equation(m, name="lowoil", domain=[t])
    maxoil = Equation(m, name="maxoil", domain=[t])
    floweq = Equation(m, name="floweq", domain=[t])
    demcons = Equation(m, name="demcons", domain=[t])
    oileq = Equation(m, name="oileq", domain=[t])

    costfn.definition = cost == Sum(
        t, 300 + 6 * others[t] + 0.0025 * (others[t] ** 2)
    )
    lowoil[t] = poil[t] >= 100 * status[t]
    maxoil[t] = poil[t] <= 500 * status[t]
    floweq[t] = volume[t] == volume[t.lag(1)] + 500 - oil[t] + initlev[t]
    oileq[t] = oil[t] == 50 * status[t] + poil[t] + 0.005 * (poil[t] ** 2)
    demcons[t] = poil[t] + others[t] >= load[t]

    model = Model(
        m,
        name="ucom",
        equations=m.getEquations(),
        problem="MINLP",
        sense=Sense.MIN,
        objective=cost,
    )
    poil.l[t] = 100

    model.solve()


if __name__ == "__main__":
    main()
