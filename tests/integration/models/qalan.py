"""
## GAMSSOURCE: https://gams.com/latest/gamslib_ml/libhtml/gamslib_qalan.html
## LICENSETYPE: Demo
## MODELTYPE: MIQCP
## KEYWORDS: quadratic constraint programming, mixed integer quadratic constraint programming, portfolio optimization, complete enumeration, finance

This is a mini mean-variance portfolio selection problem described in
'GAMS/MINOS: Three examples' by Alan S. Manne, Department of Operations
Research, Stanford University, May 1986.

Integer variables have been added to restrict the number of securities
selected. The resulting MINLP problem is solved with different option
settings to demonstrate some DICOPT features. Finally, the model is
solved by complete enumeration using GAMS procedural facilities.


Manne, A S, GAMS/MINOS: Three examples. Tech. rep., Department of
Operations Research, Stanford University, 1986.
"""

import math
import sys

import numpy as np

import gamspy as gp


def main():
    with gp.Container():
        i = gp.Set(records=["hardware", "software", "show-biz", "t-bills"])
        j = gp.Alias(alias_with=i)
        target = gp.Parameter(records=10)
        mean = gp.Parameter(domain=i, records=np.array([8, 9, 12, 7]))
        v = gp.Parameter(
            domain=[i, j],
            records=[
                ("hardware", "hardware", 4),
                ("hardware", "software", 3),
                ("hardware", "show-biz", -1),
                ("hardware", "t-bills", 0),
                ("software", "hardware", 3),
                ("software", "software", 6),
                ("software", "show-biz", 1),
                ("software", "t-bills", 0),
                ("show-biz", "hardware", -1),
                ("show-biz", "software", 1),
                ("show-biz", "show-biz", 10),
                ("show-biz", "t-bills", 0),
                ("t-bills", "hardware", 0),
                ("t-bills", "software", 0),
                ("t-bills", "show-biz", 0),
                ("t-bills", "t-bills", 0),
            ],
        )
        x = gp.Variable(domain=i, type="positive")
        variance = gp.Variable()
        fsum = gp.Equation(definition=x.sum() == 1)
        dmean = gp.Equation(definition=gp.Sum(i, mean[i] * x[i]) == target)
        dvar = gp.Equation(
            definition=gp.Sum(i, x[i] * gp.Sum(j, v[i, j] * x[j])) == variance
        )
        portfolio = gp.Model(
            equations=[fsum, dmean, dvar],
            problem="qcp",
            sense="min",
            objective=variance,
        )
        portfolio.solve()
        maxassets = gp.Parameter(records=3)
        active = gp.Variable(domain=i, type="binary")
        setindic = gp.Equation(domain=i, definition=x[i] <= active[i])
        maxactive = gp.Equation(definition=active.sum() <= maxassets)
        p1 = gp.Model(
            equations=[fsum, dmean, dvar, setindic, maxactive],
            problem="MIQCP",
            sense="min",
            objective=variance,
        )
        p1.solve(
            output=sys.stdout,
            solver="shot",
            options=gp.Options(relative_optimality_gap=1e-6),
        )
        assert math.isclose(p1.objective_value, 2.925, abs_tol=0.0001)


if __name__ == "__main__":
    main()
