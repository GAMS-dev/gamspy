"""
## GAMSSOURCE: https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_carseq.html
## LICENSETYPE: Demo
## MODELTYPE: MIP, MINLP
## KEYWORDS: mixed integer linear programming, mixed integer nonlinear programming, production planning, car manufacturing, line problem


Car Sequencing (CARSEQ)

A number of cars are to be produced; they are not identical, because
different options are available as variants on the basic model. The
assembly line has different stations which install the various options
(air-conditioning, sun-roof, etc.). These stations have been designed
to handle at most a certain percentage of the cars passing along the
assembly line. Furthermore, the cars requiring a certain option must not
be bunched together, otherwise the station will not be able to cope.
Consequently, the cars must be arranged in a sequence so that the capacity
of each station is never exceeded. For instance, if a particular station
can only cope with at most half of the cars passing along the line, the
sequence must be built so that at most 1 car in any 2 requires that option.
The problem has been shown to be NP-hard (Gent 1999).

This is the example given in Dincbas, et al. More instances can
be downloaded from CSPLib (problem prob001).


Dincbas et al., Dincbas, M., Simonis, H., and Van Hentenryck, P.
Solving the car-sequencing problem in constraint logic programming.
In 8th European Conference on Artificial Intelligence (ECAI 88) ,
Y. Kodratoff, Ed. Pitmann Publishing, London, Munich, Germany, 290-295, 1988
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import gamspy.math as gams_math
from gamspy import (
    Alias,
    Card,
    Container,
    Equation,
    Model,
    Options,
    Ord,
    Parameter,
    Sense,
    Set,
    Sum,
    Variable,
)
from gamspy.math import ifthen


def main(mip=True):
    classData_recs = np.array(
        [
            [1, 1, 0, 1, 1, 0],
            [1, 0, 0, 0, 1, 0],
            [2, 0, 1, 0, 0, 1],
            [2, 0, 1, 0, 1, 0],
            [2, 1, 0, 1, 0, 0],
            [2, 1, 1, 0, 0, 0],
        ]
    )
    classData_recs = pd.DataFrame(
        classData_recs,
        columns=["numCars", "opt1", "opt2", "opt3", "opt4", "opt5"],
        index=["class1", "class2", "class3", "class4", "class5", "class6"],
    )
    no_of_cars = classData_recs.numCars.sum()
    classData_recs = classData_recs.reset_index().melt(
        id_vars="index", var_name="Category", value_name="Value"
    )

    m = Container()

    # Sets
    p = Set(
        m,
        name="p",
        records=[f"pos{i}" for i in range(1, 11)],
        description="position",
    )
    o = Set(
        m,
        name="o",
        records=[f"opt{i}" for i in range(1, 6)],
        description="options",
    )
    c = Set(
        m,
        name="c",
        records=[f"class{i}" for i in range(1, 7)],
        description="classes",
    )

    # Parameter
    maxc = Parameter(
        m,
        name="maxc",
        domain=o,
        records=np.array([1, 2, 1, 2, 1]),
        description="maximum number of cars with that option in a block",
    )
    bs = Parameter(
        m,
        name="bs",
        domain=o,
        records=np.array([2, 3, 3, 5, 5]),
        description="block size to which the maximum number maxc refers",
    )

    classData = Parameter(
        m,
        name="classData",
        domain=[c, "*"],
        records=classData_recs,
        description="class data",
    )

    if len(p) != no_of_cars:
        raise Exception("inconsistent number of cars")

    pp = Alias(m, name="pp", alias_with=p)

    # Sets
    blk = Set(
        m,
        name="blk",
        domain=[o, p],
        description="blocks of positions to monitor",
    )
    blkc = Set(
        m,
        name="blkc",
        domain=[o, p, pp],
        description="positions in the blocks",
    )

    blkc[o, p, pp].where[Ord(p) <= Card(p) - bs[o] + 1] = (
        Ord(pp) >= Ord(p)
    ) & (Ord(pp) < Ord(p) + bs[o])
    blk[o, p] = Sum(pp.where[blkc[o, p, pp]], 1)

    # Variables
    sumc = Variable(m, name="sumc", type="free", domain=[o, p])
    cp = Variable(
        m,
        name="cp",
        type="binary",
        domain=[c, p],
        description="class k is scheduled at position p",
    )
    op = Variable(
        m,
        name="op",
        type="free",
        domain=[o, p],
        description="option o appears at position p",
    )
    v = Variable(
        m,
        name="v",
        type="free",
        domain=[o, p],
        description="violations in a block",
    )

    if mip:
        v.type = "positive"
        op.type = "binary"

    # Equations
    defnumCars = Equation(
        m,
        name="defnumCars",
        domain=c,
        description="exactly numCars of class c assigned to positions",
    )
    defoneCar = Equation(
        m,
        name="defoneCar",
        domain=p,
        description="one car assigned to each position p",
    )
    defop = Equation(
        m,
        name="defop",
        domain=[o, p],
        description="option o appears at position p",
    )
    defopLS = Equation(
        m,
        name="defopLS",
        domain=[o, p],
        description="option o appears at position p",
    )
    defviol = Equation(
        m, name="defviol", domain=[o, p], description="violations in a block"
    )
    defviolLS = Equation(
        m, name="defviolLS", domain=[o, p], description="violations in a block"
    )
    defsumc = Equation(m, name="defsumc", domain=[o, p])

    defnumCars[c] = Sum(p, cp[c, p]) == classData[c, "numCars"]
    defoneCar[p] = Sum(c, cp[c, p]) == 1
    defop[o, p] = Sum(c.where[classData[c, o]], cp[c, p]) <= op[o, p]
    defsumc[o, p] = sumc[o, p] == Sum(c.where[classData[c, o]], cp[c, p])
    defopLS[o, p] = op[o, p] == ifthen(sumc[o, p] >= 0.5, 1, 0)
    defviol[blk[o, p]] = Sum(blkc[blk, pp], op[o, pp]) <= maxc[o] + v[o, p]
    defviolLS[blk[o, p]] = v[o, p] == gams_math.Max(
        Sum(blkc[blk, pp], op[o, pp]) - maxc[o], 0
    )

    obj = Sum(blk[o, p], v[o, p])

    # Model
    carseqMIP = Model(
        m,
        name="carseqMIP",
        equations=[
            defnumCars,
            defoneCar,
            defop,
            defviol,
        ],
        problem="mip",
        sense=Sense.MIN,
        objective=obj,
    )
    carseqLS = Model(
        m,
        name="carseqLS",
        equations=[
            defnumCars,
            defoneCar,
            defsumc,
            defopLS,
            defviolLS,
        ],
        problem="minlp",
        sense=Sense.MIN,
        objective=obj,
    )

    if mip:
        v.type = "positive"
        op.type = "binary"
        carseqMIP.solve(options=Options(relative_optimality_gap=0))
    else:
        carseqLS.solve(options=Options(relative_optimality_gap=0))

    rep = Parameter(m, name="rep", domain=[p, c, o])
    rep[p, c, o].where[(cp.l[c, p] > 0.5)] = classData[c, o]

    print("Objective Function Value: ", carseqMIP.objective_value)

    import math

    assert math.isclose(carseqMIP.objective_value, -9)


if __name__ == "__main__":
    main()
