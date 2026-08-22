"""
## GAMSSOURCE: https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_trnspwl.html
## LICENSETYPE: Demo
## MODELTYPE: MIP, NLP
## KEYWORDS: non linear programming, mixed integer linear programming, transportation problem, scheduling, economies of scale, non-convex objective, special ordered sets


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
"""

from __future__ import annotations

import numpy as np

import gamspy.formulations as formulations
from gamspy import (
    Card,
    Container,
    Equation,
    Model,
    Options,
    Parameter,
    Problem,
    Sense,
    Set,
    Smax,
    Sum,
    Variable,
)
from gamspy.math import sqrt


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
        domain=i,
        records=np.array([350, 600]),
        description="capacity of plant i in cases",
    )

    b = Parameter(
        m,
        name="b",
        domain=j,
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

    # Equation

    # Objective Function; total transportation costs in thousands of dollars
    cost = Sum([i, j], c[i, j] * sqrt(x[i, j]))

    supply = Equation(
        m,
        name="supply",
        domain=i,
        description="observe supply limit at plant i",
    )
    demand = Equation(
        m,
        name="demand",
        domain=j,
        description="satisfy demand at market j",
    )

    supply[i] = Sum(j, x[i, j]) <= a[i]

    demand[j] = Sum(i, x[i, j]) >= b[j]

    transport = Model(
        m,
        name="transport",
        equations=m.getEquations(),
        problem=Problem.NLP,
        sense=Sense.MIN,
        objective=cost,
    )

    # Start the local NLP solver in a local solution that is not globally
    # optimal
    x.l["seattle", "chicago"] = 25
    x.l["seattle", "topeka"] = 275
    x.l["san-diego", "new-york"] = 325
    x.l["san-diego", "chicago"] = 275

    localopt = Parameter(
        m,
        name="localopt",
        description="objective of local optimum that is not globally optimal",
    )

    transport.solve(options=Options(nlp="conopt"))
    print(
        "Initial Objective Function Value: ",
        round(transport.objective_value, 3),
    )

    localopt[...] = transport.objective_value

    # The first model (formulation a) implements a piecewise linear
    # approximation based on the convex combination of neighboring points
    # using SOS2 variables with unbounded segments at the beginning and
    # end of the discretization
    # Sets
    s = Set(
        m,
        name="s",
        records=["slope0"] + [f"s{i}" for i in range(1, 7)] + ["slopeN"],
        description="SOS2 elements",
    )
    ss = Set(
        m,
        name="ss",
        domain=s,
        records=[f"s{i}" for i in range(1, 7)],
        description="sample points",
    )

    # Parameters
    p = Parameter(m, name="p", domain=s, description="x coordinate of sample point")
    sqrtp = Parameter(
        m, name="sqrtp", domain=s, description="y coordinate of sample point"
    )
    xlow = Parameter(m, name="xlow", records=50)
    xhigh = Parameter(m, name="xhigh", records=400)
    xmax = Parameter(m, name="xmax")

    xmax[...] = Smax(i, a[i])

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

    x_points = [record[-1] for record in p[ss].toList()]
    y_points = [record[-1] for record in sqrtp[ss].toList()]
    left_gradient = -sqrtp["slope0"].toList()[0]
    right_gradient = sqrtp["slopeN"].toList()[0]
    sqrtx, eqs = formulations.pwl_convexity_formulation(
        x,
        [-float("inf"), *x_points, float("inf")],
        [left_gradient, *y_points, right_gradient],
        using="sos2",
    )
    sqrtx.lo[...] = 0

    # Alternatively, use the Curve API:
    # curve = formulations.PWLCurve(
    #     list(zip(x_points, y_points, strict=True)),
    #     left_gradient=left_gradient,
    #     right_gradient=right_gradient,
    # )
    # sqrtx, eqs = formulations.pwlinear(
    #     x, curve, method="convexity", using="sos2"
    # )

    defobjdisc = Sum([i, j], c[i, j] * sqrtx[i, j])

    trnsdiscA = Model(
        m,
        name="trnsdiscA",
        equations=[supply, demand, *eqs],
        problem="mip",
        sense=Sense.MIN,
        objective=defobjdisc,
    )

    trnsdiscA.solve(options=Options(relative_optimality_gap=0))

    # The next model (formulation b) uses the convex combinations of
    # neighboring points but requires the discretization to be bounded
    # (here we go from 0 to xmax).
    p["slope0"] = 0
    p[ss] = xlow + (xhigh - xlow) / (Card(ss) - 1) * ss.off
    p["slopeN"] = xmax
    sqrtp[s] = sqrt(p[s])

    x_points = [0, *[record[-1] for record in p[ss].toList()], xmax.toValue()]
    y_points = [
        0,
        *[record[-1] for record in sqrtp[ss].toList()],
        sqrtp["slopeN"].toList()[0],
    ]
    sqrtx, eqs = formulations.pwl_convexity_formulation(
        x, x_points, y_points, using="sos2"
    )

    # Alternatively, use the Curve API:
    # curve = formulations.PWLCurve(list(zip(x_points, y_points, strict=True)))
    # sqrtx, eqs = formulations.pwlinear(
    #     x, curve, method="convexity", using="sos2"
    # )

    defobjdisc = Sum([i, j], c[i, j] * sqrtx[i, j])

    trnsdiscB = Model(
        m,
        name="trnsdiscB",
        equations=[supply, demand, *eqs],
        problem="mip",
        sense=Sense.MIN,
        objective=defobjdisc,
    )

    trnsdiscB.solve()

    # The next model (formulation c) implements another formulation for a
    # piecewise linear function. We need to assume that the domain region
    # is bounded. We use the same discretization as in the previous
    # formulation.
    sqrtx, eqs = formulations.pwl_interval_formulation(x, x_points, y_points)

    # Alternatively, use the Curve API with the curve defined above:
    # sqrtx, eqs = formulations.pwlinear(x, curve, method="interval")

    defobjdisc = Sum([i, j], c[i, j] * sqrtx[i, j])

    trnsdiscC = Model(
        m,
        name="trnsdiscC",
        equations=[supply, demand, *eqs],
        problem="mip",
        sense=Sense.MIN,
        objective=defobjdisc,
    )

    trnsdiscC.solve()

    # Now restart the local solver from this approximate point
    transport.solve(options=Options(nlp="conopt"))

    print(
        "Improved Objective Function Value: ",
        round(transport.objective_value, 3),
    )

    # Ensure that we are better off transport.objective_value
    if transport.objective_value - localopt.toValue() > 1e-6:
        raise Exception("we should get an improved transport.objective_value")


if __name__ == "__main__":
    main()
