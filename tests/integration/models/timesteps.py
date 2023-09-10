"""
Accessing previous (or next) Time Steps in an Equation fast (TIMESTEPS)

In dynamic models one often needs access to previous or next time steps. Access
to single time steps can be easly implemented via the lag and lead operator.
It gets more difficult if one needs access to a larger set of time steps.
the expression sum(tt$(ord(tt)<=ord(t) and ord(tt)>=ord(t)-n), ...) where t is
the current time step controlled from the outside can be very slow.

The following example model shows how to do this fast in GAMS using an example
from power generation modeling. We have a set of time steps and a number of
generators. A generator can only start once in a given time slice. We implement
the equation that enforces this in three different ways:

1) naive GAMS syntax via ord() calculation
2) calculate a set of time slices for any given active time step
3) fast implementation directly in the equation using the same idas to create
   the set in 2 fast

Solution 2 is actually the fastest, but it consumes a lot of memory. We will
eventually require this much memory in the model generation (we have many
non-zero entires in the equation) but we can safe the extra amount inside GAMS
data by using method 3.

Keywords: mixed integer linear programming, GAMS language features, dynamic
          modelling, time steps, power generation
"""

from gamspy import (
    Set,
    Alias,
    Parameter,
    Variable,
    Equation,
    Container,
    Model,
    Sum,
    Number,
    Ord,
    Sense,
)
from gamspy.math import uniformInt


def main(mt=2016, mg=17, mindt=10, maxdt=40):
    m = Container()

    if mindt > maxdt:
        raise Exception("minimum downtime is larger than maximum downtime")

    # Sets
    t = Set(
        m,
        name="t",
        records=[f"t{i}" for i in range(1, mt + 1)],
        description="hours",
    )
    g = Set(
        m,
        name="g",
        records=[f"g{i}" for i in range(1, mg + 1)],
        description="generators",
    )

    # Parameters
    pMinDown = Parameter(
        m, name="pMinDown", domain=[g, t], description="minimum downtime"
    )
    pMinDown[g, t] = uniformInt(mindt, maxdt)

    t1 = Alias(m, name="t1", alias_with=t)
    t2 = Alias(m, name="t2", alias_with=t)

    # Subsets
    sMinDown = Set(
        m,
        name="sMinDown",
        domain=[g, t1, t2],
        description="hours t2 g cannot start if we start g in t1",
    )
    sMinDownFast = Set(
        m,
        name="sMinDownFast",
        domain=[g, t1, t2],
        description="hours t2 g cannot start if we start g in t1",
    )
    tt = Set(
        m,
        name="tt",
        domain=[t],
        records=[f"t{i}" for i in range(1, maxdt + 1)],
        description="max downtime hours",
    )

    # Slow and fast calculation for the set of time slices t2 for a given time
    # step t1
    # Output from profile=1
    # ----     50 Assignment sMinDown      5.819  5.819 SECS 26 MB  850713
    # ----     51 Assignment sMinDownFast  0.187  6.006 SECS 48 MB  850713

    sMinDown[g, t1, t2] = (Ord(t1) >= Ord(t2)) & (
        Ord(t2) > Ord(t1) - pMinDown[g, t1]
    )
    sMinDownFast[g, t1, t.lead((Ord(t1) - pMinDown[g, t1]))].where[
        (tt[t]) & (Ord(t) <= pMinDown[g, t1])
    ] = Number(1)

    diff = Set(m, name="diff", domain=[g, t1, t2])
    diff[g, t1, t2] = sMinDown[g, t1, t2] ^ sMinDownFast[g, t1, t2]
    if diff.records is not None:
        raise Exception("sets are different")

    vStart = Variable(m, name="vStart", type="binary", domain=[g, t])
    z = Variable(m, name="z")

    # Slow, fast, and fastest (but memory intensive way because we need to
    # store sMinDownFast) way to write the equation
    # Output from profile = 1
    # ----     67 Equation   eStartNaive   6.099 12.215 SECS 106 MB 34272
    # ----     68 Equation   eStartFast    0.593 12.808 SECS 144 MB 34272
    # ----     69 Equation   eStartFaster  0.468 13.276 SECS 180 MB 34272

    # Equations
    eStartNaive = Equation(m, name="eStartNaive", domain=[g, t])
    eStartFast = Equation(m, name="eStartFast", domain=[g, t])
    eStartFaster = Equation(m, name="eStartFaster", domain=[g, t])
    defobj = Equation(m, name="defobj")

    eStartNaive[g, t1] = (
        Sum(
            t2.where[
                (Ord(t1) >= Ord(t2)) & (Ord(t2) > Ord(t1) - pMinDown[g, t1])
            ],
            vStart[g, t2],
        )
        <= 1
    )

    eStartFast[g, t1] = (
        Sum(
            tt[t].where[Ord(t) <= pMinDown[g, t1]],
            vStart[g, t.lead(Ord(t1) - pMinDown[g, t1])],
        )
        <= 1
    )

    eStartFaster[g, t1] = Sum(sMinDownFast[g, t1, t2], vStart[g, t2]) <= 1

    defobj.expr = z == Sum([g, t], vStart[g, t])

    maxStarts = Model(
        m,
        name="maxStarts",
        equations=m.getEquations(),
        problem="mip",
        sense=Sense.MAX,
        objective=z,
    )

    maxStarts.solve()
    print("Objective Function Value: ", z.records.level[0])


if __name__ == "__main__":
    main()
