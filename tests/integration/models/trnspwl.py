"""
A Transportation Problem with discretized Economies of Scale (TRNSPWL)

This problem finds a least cost shipping schedule that meets
requirements at markets and supplies at factories. This instance
applies economies of scale which results in a non-convex
objective. This is an extension of the trnsport model in the GAMS
Model Library.

The original nonlinear term is "sum((i,j), c(i,j)*sqrt(x(i,j)))".
We use the following discretization f(x) of sqrt(x)

  For x<=50:  f(x) = 1/sqrt(50)*x,
  for x>=400: f(x) = (sqrt(600)-sqrt(400))/200*(x-400) + sqrt(400)
  in between we discretize with linear interpolation between points

This discretization has some good properties:
  0) f(x) is a continuous function
  1) f(0)=0, otherwise we would pick up a fixed cost even for unused
  connections
  2) a fine representation in the reasonable range of shipments
  (between 50 and 400)
  3) f(x) underestimates sqrt in the area of x=0 to 600. Past that is
  overestimates sqrt.

The model is organized as follows:
  1) We set a starting point for the NLP solver so it will get stuck
     in local optimum that is not the global optimum.

  2) We use three formulations for representing piecewise linear
     functions all based on the same discretization.

     a) a formulation with SOS2 variables. This formulation mainly is
        based on the convex combination of neighboring
        points. Moreover, the domain of the discretization can be
        unbounded: we can assign a slope in the (potentially
        unbounded) first and last segment.

     b) a formulation with SOS2 variables based on convex combinations
        of neighboring points. This formulation requires a bounded
        region for the discretization. Here we discretize between 0 and
        600.

     c) a formuation with binary variables. This also requires the
        domain to be bounded, but it does not rely on the convex
        combination of neighboring points. There are examples, where
        this formulation solves much faster than the formulation b).

     In this example x is clearly bounded by 0 from below and
     min(smax(i,a(i),smax(j,b(j)) from above, so formulation b and c
     are sufficient and perform better on this particular model and
     instance. We added the formulation a to demonstrate how to model
     an unbounded discretization, in case there are no derived
     bounds. The formulation a can be easily adjusted to accommodate
     problems where only one end of the discretization is unbounded.

  3) We restart the non-convex NLP from the solution of the discretized
     model and hope that the NLP solver finds the global solution.


Dantzig, G B, Chapter 3.3. In Linear Programming and Extensions.
Princeton University Press, Princeton, New Jersey, 1963.

Keywords: non linear programming, mixed integer linear programming,
          transportation problem, scheduling, economies of scale, non-convex
          objective, special ordered sets
"""

from gamspy import (
    Set,
    Parameter,
    Variable,
    Equation,
    Container,
    Model,
    Sum,
    Number,
    Smax,
    Card,
)
from gamspy.math import sqrt
import numpy as np
from gamspy import Problem, Sense


def main():
    m = Container()

    # Sets
    i = Set(
        m,
        name="i",
        records=["seattle", "san-diego"],
        description="canning plants",
    )
    j = Set(
        m,
        name="j",
        records=["new-york", "chicago", "topeka"],
        description="markets",
    )

    # Parameters
    a = Parameter(
        m,
        name="a",
        domain=[i],
        records=np.array([350, 600]),
        description="capacity of plant i in cases",
    )

    b = Parameter(
        m,
        name="b",
        domain=[j],
        records=np.array([325, 300, 275]),
        description="demand at market j in cases",
    )

    d = Parameter(
        m,
        name="d",
        domain=[i, j],
        records=np.array([[2.5, 1.7, 1.8], [2.5, 1.8, 1.4]]),
        description="distance in thousands of miles",
    )

    f = Parameter(
        m,
        name="f",
        records=90,
        description="freight in dollars per case per thousand miles",
    )

    c = Parameter(
        m,
        name="c",
        domain=[i, j],
        description="transport cost in thousands of dollars per case",
    )
    c[i, j] = f * d[i, j] / 1000

    # Variables
    x = Variable(
        m,
        name="x",
        type="positive",
        domain=[i, j],
        description="shipment quantities in cases",
    )
    z = Variable(
        m,
        name="z",
        type="free",
        description="total transportation costs in thousands of dollars",
    )

    # Equation
    cost = Equation(m, name="cost", description="define objective function")
    supply = Equation(
        m,
        name="supply",
        domain=[i],
        description="observe supply limit at plant i",
    )
    demand = Equation(
        m,
        name="demand",
        domain=[j],
        description="satisfy demand at market j",
    )

    cost.expr = z == Sum([i, j], c[i, j] * sqrt(x[i, j]))

    supply[i] = Sum(j, x[i, j]) <= a[i]

    demand[j] = Sum(i, x[i, j]) >= b[j]

    transport = Model(
        m,
        name="transport",
        equations=m.getEquations(),
        problem=Problem.NLP,
        sense=Sense.MIN,
        objective=z,
    )

    # Start the local NLP solver in a local solution that is not globally
    # optimal
    x.l["seattle  ", "chicago "] = 25
    x.l["seattle  ", "topeka  "] = 275
    x.l["san-diego", "new-york"] = 325
    x.l["san-diego", "chicago "] = 275

    localopt = Parameter(
        m,
        name="localopt",
        description="objective of local optimum that is not globally optimal",
    )

    m.addOptions({"nlp": "conopt"})

    transport.solve()
    print("Initial Objective Function Value: ", round(z.records.level[0], 3))

    localopt.assign = z.l

    # The first model (formulation a) implements a piecewise linear
    # approximation based on the convex combination of neighboring points
    # using SOS2 variables with unbounded segments at the beginning and
    # end of the discretization
    # Sets
    s = Set(
        m,
        name="s",
        records=["slope0", "slopeN"] + [f"s{i}" for i in range(1, 7)],
        description="SOS2 elements",
    )
    ss = Set(
        m,
        name="ss",
        domain=[s],
        records=[f"s{i}" for i in range(1, 7)],
        description="sample points",
    )

    # Parameters
    p = Parameter(
        m, name="p", domain=[s], description="x coordinate of sample point"
    )
    sqrtp = Parameter(
        m, name="sqrtp", domain=[s], description="y coordinate of sample point"
    )
    xlow = Parameter(m, name="xlow", records=50)
    xhigh = Parameter(m, name="xhigh", records=400)
    xmax = Parameter(m, name="xmax")

    xmax.assign = Smax(i, a[i])

    if xmax.records.value[0] < xhigh.records.value[0]:
        raise Exception("xhigh too big")

    if xlow.records.value[0] < 0:
        raise Exception("xlow less than 0")

    # Equidistant sampling of the sqrt function with slopes at the beginning
    # and end
    p["slope0"] = -1
    p[ss] = xlow + (xhigh - xlow) / (Card(ss) - 1) * ss.off
    p["slopeN"] = 1

    sqrtp["slope0"] = -1 / sqrt(xlow)
    sqrtp[ss] = sqrt(p[ss])
    sqrtp["slopeN"] = (sqrt(xmax) - sqrt(xhigh)) / (xmax - xhigh)

    xs = Variable(m, name="xs", type="sos2", domain=[i, j, s])
    sqrtx = Variable(m, name="sqrtx", type="positive", domain=[i, j])

    # Equations
    defsos1 = Equation(m, name="defsos1", domain=[i, j])
    defsos2 = Equation(m, name="defsos2", domain=[i, j])
    defsos3 = Equation(m, name="defsos3", domain=[i, j])
    defobjdisc = Equation(m, name="defobjdisc")

    defsos1[i, j] = x[i, j] == Sum(s, p[s] * xs[i, j, s])

    defsos2[i, j] = sqrtx[i, j] == Sum(s, sqrtp[s] * xs[i, j, s])

    defsos3[i, j] = Sum(ss, xs[i, j, ss]) == 1

    defobjdisc.expr = z == Sum([i, j], c[i, j] * sqrtx[i, j])

    trnsdiscA = Model(
        m,
        name="trnsdiscA",
        equations=[supply, demand, defsos1, defsos2, defsos3, defobjdisc],
        problem="mip",
        sense=Sense.MIN,
        objective=z,
    )

    m.addOptions({"optCr": 0})

    trnsdiscA.solve()

    # The next model (formulation b) uses the convex combinations of
    # neighboring points but requires the discretization to be bounded
    # (here we go from 0 to xmax).
    p["slope0"] = 0
    p[ss] = xlow + (xhigh - xlow) / (Card(ss) - 1) * ss.off
    p["slopeN"] = xmax
    sqrtp[s] = sqrt(p[s])

    # We can just use model trnsdiscA but need to include the first and
    # last segment into the set ss that builds the convex combinations.
    ss[s] = Number(1)

    trnsdiscA.solve()

    # The next model (formulation c) implements another formulation for a
    # piecewise linear function. We need to assume that the domain region
    # is bounded. We use the same discretization as in the previous
    # formulation.
    # Sets
    g = Set(
        m,
        name="g",
        domain=[s],
        records=["slope0"] + [f"s{i}" for i in range(1, 7)],
        description="Segments",
    )

    # Parameters
    nseg = Parameter(
        m,
        name="nseg",
        domain=[s],
        description="relative increase of x in segment",
    )
    ninc = Parameter(
        m,
        name="ninc",
        domain=[s],
        description="relative increase of sqrtx in segment",
    )

    nseg[s].where[g[s]] = p[s.lead(1, "linear")] - p[s]
    ninc[s].where[g[s]] = sqrtp[s.lead(1, "linear")] - sqrtp[s]

    # Variables
    seg = Variable(
        m,
        name="seg",
        type="positive",
        domain=[i, j, s],
        description="shipment in segment",
    )
    gs = Variable(
        m,
        name="gs",
        type="binary",
        domain=[i, j, s],
        description="indicator for shipment in segment",
    )

    # Equations
    defx = Equation(
        m,
        name="defx",
        domain=[i, j],
        description="definition of x",
    )
    defsqrt = Equation(
        m,
        name="defsqrt",
        domain=[i, j],
        description="definition of sqrt",
    )
    defseg = Equation(
        m,
        name="defseg",
        domain=[i, j, s],
        description="segment can only have shipment if indicator is on",
    )
    defgs = Equation(
        m,
        name="defgs",
        domain=[i, j],
        description="select at most one segment",
    )

    defx[i, j] = x[i, j] == Sum(g, p[g] * gs[i, j, g] + nseg[g] * seg[i, j, g])

    defsqrt[i, j] = sqrtx[i, j] == Sum(
        g, sqrtp[g] * gs[i, j, g] + ninc[g] * seg[i, j, g]
    )

    defseg[i, j, g] = seg[i, j, g] <= gs[i, j, g]

    defgs[i, j] = Sum(g, gs[i, j, g]) <= 1

    trnsdiscB = Model(
        m,
        name="trnsdiscB",
        equations=[supply, demand, defx, defsqrt, defseg, defgs, defobjdisc],
        problem="mip",
        sense=Sense.MIN,
        objective=z,
    )

    trnsdiscB.solve()

    # Now restart the local solver from this approximate point
    transport.solve()

    print("Improved Objective Function Value: ", round(z.records.level[0], 3))

    # Ensure that we are better off than before
    if z.records.level[0] - localopt.records.value[0] > 1e-6:
        raise Exception("we should get an improved solution")


if __name__ == "__main__":
    main()
