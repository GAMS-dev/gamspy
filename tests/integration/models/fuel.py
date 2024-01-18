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

from __future__ import annotations

import os

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
    m = Container(
        system_directory=os.getenv("SYSTEM_DIRECTORY", None),
        delayed_execution=int(os.getenv("DELAYED_EXECUTION", False)),
    )

    # Set
    t = Set(
        m,
        name="t",
        records=["period-1", "period-2", "period-3"],
        description="scheduling periods (2hrs)",
    )

    # Data
    load = Parameter(
        m,
        name="load",
        domain=t,
        records=pd.DataFrame(
            [["period-1", 400], ["period-2", 900], ["period-3", 700]]
        ),
        description="system load",
    )
    initlev = Parameter(
        m,
        name="initlev",
        domain=t,
        records=pd.DataFrame([["period-1", 3000]]),
        description="initial level of the oil storage tank",
    )

    # Variable
    status = Variable(
        m,
        name="status",
        domain=t,
        type="Binary",
        description="on or off status of the oil based generating unit",
    )
    poil = Variable(
        m,
        name="poil",
        domain=t,
        description="generation level of oil based unit",
    )
    others = Variable(
        m, name="others", domain=t, description="other generation"
    )
    oil = Variable(
        m, name="oil", domain=t, type="Positive", description="oil consumption"
    )
    volume = Variable(
        m,
        name="volume",
        domain=t,
        type="Positive",
        description="the volume of oil in the storage tank",
    )
    cost = Variable(m, name="cost", description="total operating cost")

    volume.up[t] = 4000
    volume.lo[t].where[Ord(t) == Card(t)] = 2000

    others.lo[t] = 50
    others.up[t] = 700

    # Equation
    costfn = Equation(
        m,
        name="costfn",
        description="total operating cost of unit 2 -- the objective fn",
    )
    lowoil = Equation(
        m,
        name="lowoil",
        domain=t,
        description="lower limit on oil generating unit",
    )
    maxoil = Equation(
        m,
        name="maxoil",
        domain=t,
        description="upper limit on oil generating unit",
    )
    floweq = Equation(
        m,
        name="floweq",
        domain=t,
        description="the oil flow balance in the storage tank",
    )
    demcons = Equation(
        m,
        name="demcons",
        domain=t,
        description="total generation must meet the load",
    )
    oileq = Equation(
        m, name="oileq", domain=t, description="calculation of oil consumption"
    )

    costfn[...] = cost == Sum(
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

    import math

    assert math.isclose(model.objective_value, 8566.1190, rel_tol=0.001)


if __name__ == "__main__":
    main()
