"""
Time dependent temperature field in a rectangular area.

Determination of the time dependent temperature field in a rectangular area
with heterogeneous thermal conductivity and a source points of heat
inside the area as well as heat flows temperature through the borders
of the solution domain.

Adapted from:
McKinney, D.C., Savitsky, A.G., Basic optimization models for water and
energy management. June 1999 (revision 6, February 2003).
http://www.ce.utexas.edu/prof/mckynney/ce385d/papers/GAMS-Tutorial.pdf
"""
import numpy as np

from gamspy import Card
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Ord
from gamspy import Parameter
from gamspy import Set
from gamspy import Sum
from gamspy import Variable


def data_records():
    # v records table
    v_rec = np.array(
        [
            [
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
            [
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
            [
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
            [
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
            [
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                1.0,
                1.0,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
            [
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                1.0,
                1.0,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
            [
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                1.0,
                1.0,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
            [
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                1.0,
                1.0,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
            [
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                1.0,
                1.0,
                1.0,
                1.0,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
            [
                0.5,
                0.5,
                0.5,
                0.5,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
            [
                0.5,
                0.5,
                0.5,
                0.5,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
            [
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
            [
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
            [
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.5,
                0.5,
                0.5,
            ],
            [
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.5,
                0.5,
            ],
            [
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                1.0,
                1.0,
                1.0,
                1.0,
                0.5,
                0.5,
            ],
            [
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                1.0,
                1.0,
                1.0,
                0.5,
                0.5,
            ],
            [
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
            [
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
            [
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
        ]
    )

    return v_rec


def main():
    m = Container()

    # SETS #
    time = Set(m, name="time", records=[f"t{t}" for t in range(1, 26)])
    x = Set(m, name="x", records=[f"i{i}" for i in range(1, 21)])
    y = Set(m, name="y", records=[f"j{j}" for j in range(1, 21)])
    inside = Set(m, name="inside", domain=[x, y])

    inside[x, y] = True
    inside[x, y].where[Ord(x) == 1] = False
    inside[x, y].where[Ord(x) == Card(x)] = False
    inside[x, y].where[Ord(y) == 1] = False
    inside[x, y].where[Ord(y) == Card(y)] = False

    # SCALARS #
    # Parameters determination
    dx = Parameter(
        m, name="dx", records=0.1, description="step for space in x direction"
    )
    dy = Parameter(
        m, name="dy", records=0.1, description="step for space in y direction"
    )
    dt = Parameter(m, name="dt", records=0.1, description="step for time")
    c = Parameter(m, name="c", records=1.0)
    rho = Parameter(m, name="rho", records=1.0)
    heat = Parameter(
        m,
        name="heat",
        records=0.0,
        description="accumulation in one time step",
    )

    # PARAMETERS #
    # Temperature supply determination
    supply = Parameter(m, name="supply", domain=[x, y])
    past_T = Parameter(m, name="past_T", domain=[x, y])
    v = Parameter(m, name="v", domain=[x, y], records=data_records())

    supply[x, y] = 0
    supply["i10", "j18"] = 100 / dx / dy
    past_T[x, y] = 0

    # VARIABLES #
    # Model description
    obj = Variable(m, name="obj", description="objective variable")
    t = Variable(
        m, name="t", domain=[x, y], description="field of temperature"
    )
    Q = Variable(m, name="Q", description="temperature on boundaries")

    # Variable bounds
    t.lo[x, y] = 0.0
    t.up[x, y] = 200.0

    # Initial values
    t.l[x, y] = 0.0
    Q.l.assign = 0.0

    # EQUATIONS #
    temp = Equation(
        m,
        name="temp",
        type="regular",
        domain=[x, y],
        description="main equation of heat transport",
    )
    f1 = Equation(
        m,
        name="f1",
        type="regular",
        domain=[x, y],
        description="boundary computation dt:dn",
    )
    f2 = Equation(
        m,
        name="f2",
        type="regular",
        domain=[x, y],
        description="boundary computation dt:dn",
    )
    f3 = Equation(
        m,
        name="f3",
        type="regular",
        domain=[x, y],
        description="boundary computation dt:dn",
    )
    f4 = Equation(
        m,
        name="f4",
        type="regular",
        domain=[x, y],
        description="boundary computation dt:dn",
    )
    fp1 = Equation(
        m,
        name="fp1",
        type="regular",
        domain=[x, y],
        description="boundary computation t",
    )
    fp2 = Equation(
        m,
        name="fp2",
        type="regular",
        domain=[x, y],
        description="boundary computation ",
    )
    fp3 = Equation(
        m,
        name="fp3",
        type="regular",
        domain=[x, y],
        description="boundary computation t",
    )
    fp4 = Equation(
        m,
        name="fp4",
        type="regular",
        domain=[x, y],
        description="boundary computation t",
    )
    eobj = Equation(
        m, name="eobj", type="regular", description="objective equation"
    )

    temp[x, y].where[inside[x, y]] = t[x, y] - past_T[x, y] == (
        dt
        * (
            supply[x, y] / c / rho
            + (
                (v[x.lead(1), y] + v[x, y]) * (t[x.lead(1), y] - t[x, y])
                - (v[x, y] + v[x.lag(1), y]) * (t[x, y] - t[x.lag(1), y])
            )
            / dx
            / dx
            / 2.0
            + (
                (v[x, y.lead(1)] + v[x, y]) * (t[x, y.lead(1)] - t[x, y])
                - (v[x, y.lag(1)] == v[x, y]) * (t[x, y] - t[x, y.lag(1)])
            )
            / dy
            / dy
            / 2.0
        )
    )

    f1[x, y].where[(Ord(x) == 1)] = t[x.lead(1), y] - t[x, y] >= 0
    f2[x, y].where[(Ord(x) == Card(x))] = t[x, y] - t[x.lag(1), y] <= 0
    f3[x, y].where[(Ord(y) == 1)] = t[x, y.lead(1)] - t[x, y] >= 0
    f4[x, y].where[(Ord(y) == Card(y))] = t[x, y] - t[x, y.lag(1)] <= 0

    fp1[x, y].where[(Ord(x) == 1)] = t[x, y] == Q
    fp2[x, y].where[(Ord(x) == Card(x))] = t[x, y] == Q
    fp3[x, y].where[(Ord(y) == 1)] = t[x, y] == Q
    fp4[x, y].where[(Ord(y) == Card(y))] = t[x, y] == Q

    eobj.expr = obj == Q

    Diffusion2 = Model(
        m,
        name="Diffusion2",
        equations=m.getEquations(),
        problem="nlp",
        sense="min",
        objective=obj,
    )

    for tu in time.toList():
        Diffusion2.solve()
        print(f"\t --- \t Time interval = {tu} \t --- \n")
        print(t.pivot().round(4))
        print("\n")
        heat.assign = heat + Sum([x, y], t.l[x, y] - past_T[x, y]) * dx * dy
        past_T[x, y] = t.l[x, y]

    # End Diffusion2


if __name__ == "__main__":
    main()
