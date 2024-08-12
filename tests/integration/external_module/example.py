from __future__ import annotations

import math
import sys

import gamspy as gp

m = gp.Container()
y1 = gp.Variable(m, "y1")
y2 = gp.Variable(m, "y2")
x1 = gp.Variable(m, "x1")
x2 = gp.Variable(m, "x2")

eq1 = gp.Equation(m, "eq1", type="external")
eq2 = gp.Equation(m, "eq2", type="external")

eq1[...] = 1 * x1 + 3 * y1 == 1
eq2[...] = 2 * x2 + 4 * y2 == 2

model = gp.Model(
    container=m,
    name="sincos",
    equations=m.getEquations(),
    problem="NLP",
    sense="min",
    objective=y1 + y2,
    external_module="libsimple_ext_module",
)

model.solve(output=sys.stdout, solver="conopt")

assert math.isclose(y1.toDense(), -1)
assert math.isclose(y2.toDense(), -1)
