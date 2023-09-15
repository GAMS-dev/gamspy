"""
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

Keywords: mixed integer linear programming, mixed integer nonlinear
programming,
          production planning, car manufacturing, line problem
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
    Ord,
    Card,
    Number,
    Sense,
)
from gamspy.functions import ifthen
import gamspy.math as gams_math
from gamspy.functions import ifthen
import pandas as pd
import numpy as np


def main(mip=False):
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
    p = Set(m, name="p", records=[f"pos{i}" for i in range(1, 11)])
    o = Set(m, name="o", records=[f"opt{i}" for i in range(1, 6)])
    c = Set(m, name="c", records=[f"class{i}" for i in range(1, 7)])

    # Parameter
    maxc = Parameter(
        m, name="maxc", domain=[o], records=np.array([1, 2, 1, 2, 1])
    )
    bs = Parameter(m, name="bs", domain=[o], records=np.array([2, 3, 3, 5, 5]))

    classData = Parameter(
        m, name="classData", domain=[c, "*"], records=classData_recs
    )

    if len(p) != no_of_cars:
        raise Exception("inconsistent number of cars")

    pp = Alias(m, name="pp", alias_with=p)

    # Sets
    blk = Set(m, name="blk", domain=[o, p])
    blkc = Set(m, name="blkc", domain=[o, p, pp])

    blkc[o, p, pp].where[Ord(p) <= Card(p) - bs[o] + Number(1)] = (
        Ord(pp) >= Ord(p)
    ) & (Ord(pp) < Ord(p) + bs[o])
    blk[o, p] = Sum(pp.where[blkc[o, p, pp]], Number(1))

    # Variables
    sumc = Variable(m, name="sumc", type="free", domain=[o, p])
    cp = Variable(m, name="cp", type="binary", domain=[c, p])
    op = Variable(m, name="op", type="free", domain=[o, p])
    v = Variable(m, name="v", type="free", domain=[o, p])
    obj = Variable(m, name="obj", type="free")

    if mip:
        v.type = "positive"
        op.type = "binary"

    # Equations
    defnumCars = Equation(m, name="defnumCars", domain=[c])
    defoneCar = Equation(m, name="defoneCar", domain=[p])
    defop = Equation(m, name="defop", domain=[o, p])
    defopLS = Equation(m, name="defopLS", domain=[o, p])
    defviol = Equation(m, name="defviol", domain=[o, p])
    defviolLS = Equation(m, name="defviolLS", domain=[o, p])
    defobj = Equation(m, name="defobj")
    defsumc = Equation(m, name="defsumc", domain=[o, p])

    defnumCars[c] = Sum(p, cp[c, p]) == classData[c, "numCars"]
    defoneCar[p] = Sum(c, cp[c, p]) == Number(1)
    defop[o, p] = Sum(c.where[classData[c, o]], cp[c, p]) <= op[o, p]
    defsumc[o, p] = sumc[o, p] == Sum(c.where[classData[c, o]], cp[c, p])
    defopLS[o, p] = op[o, p] == ifthen(sumc[o, p] >= 0.5, 1, 0)
    defviol[blk[o, p]] = Sum(blkc[blk, pp], op[o, pp]) <= maxc[o] + v[o, p]
    defviolLS[blk[o, p]] = v[o, p] == gams_math.max(
        Sum(blkc[blk, pp], op[o, pp]) - maxc[o], Number(0)
    )

    defobj.expr = obj == Sum(blk[o, p], v[o, p])

    # Model
    carseqMIP = Model(
        m,
        name="carseqMIP",
        equations=[
            defnumCars,
            defoneCar,
            defop,
            defviol,
            defobj,
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
            defobj,
        ],
        problem="minlp",
        sense=Sense.MIN,
        objective=obj,
    )

    m.addOptions({"optCr": 0})

    if mip:
        v.type = "positive"
        op.type = "binary"
        carseqMIP.solve()
    else:
        carseqLS.solve()

    rep = Parameter(m, name="rep", domain=[p, c, o])
    rep[p, c, o].where[(cp.l[c, p] > 0.5)] = classData[c, o]

    print("Objective Function Value: ", obj.records.level[0])


if __name__ == "__main__":
    main()
