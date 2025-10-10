"""
## GAMSSOURCE: https://www.gams.com/latest/noalib_ml/libhtml/noalib_reservoir.html
## LICENSETYPE: Demo
## MODELTYPE: NLP


Onstream and offstream optimal reservoir management

Adapted from:
McKinney, D.C., Savitsky, A.G., Basic optimization models for water and
energy management. June 1999 (revision 6, February 2003).
http://www.ce.utexas.edu/prof/mckynney/ce385d/papers/GAMS-Tutorial.pdf

Andrei, N., Optimal management of system of two reservoirs.
Revista Romana de Informatica si Automatica, vol.16, no.1, 2006, pp.15-18.
"""

from __future__ import annotations

import pandas as pd

from gamspy import (
    Container,
    Equation,
    Model,
    Parameter,
    Problem,
    Sense,
    Set,
    Sum,
    Variable,
)


def main():
    m = Container()

    # Set
    n = Set(m, name="n", records=["res1", "res2"], description="reservoirs")
    t = Set(
        m,
        name="t",
        records=[
            "ian",
            "feb",
            "mar",
            "apr",
            "mai",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
            "enda",
        ],
        description="time",
    )
    tt = Set(m, name="tt", domain=t, records=["ian"])

    # Data
    q = Parameter(
        m,
        name="q",
        domain=[n, t],
        records=pd.DataFrame(
            [
                ["res1", "ian", 128],
                ["res1", "feb", 125],
                ["res1", "mar", 234],
                ["res1", "apr", 360],
                ["res1", "mai", 541],
                ["res1", "jun", 645],
                ["res1", "jul", 807],
                ["res1", "aug", 512],
                ["res1", "sep", 267],
                ["res1", "oct", 210],
                ["res1", "nov", 981],
                ["res1", "dec", 928],
                ["res1", "enda", 250],
            ]
        ),
        description="inflow water in the first reservoir rez1 (mil.m3)",
    )
    r = Parameter(
        m,
        name="r",
        domain=[n, t],
        records=pd.DataFrame(
            [
                ["res1", "ian", 100],
                ["res1", "feb", 150],
                ["res1", "mar", 200],
                ["res1", "apr", 500],
                ["res1", "mai", 222],
                ["res1", "jun", 700],
                ["res1", "jul", 333],
                ["res1", "aug", 333],
                ["res1", "sep", 300],
                ["res1", "oct", 250],
                ["res1", "nov", 250],
                ["res1", "dec", 250],
                ["res1", "enda", 200],
            ]
        ),
        description=("required released water from the first reservoir rez1 (mil.m3)"),
    )

    # Variable
    q2 = Variable(m, name="q2", domain=t)
    r2 = Variable(m, name="r2", domain=t)
    s = Variable(m, name="s", domain=[n, t])

    # Equation
    bal1 = Equation(
        m,
        domain=[n, t],
        name="bal1",
        description="water balance in reservoir S1",
    )
    bal2 = Equation(
        m,
        domain=[n, t],
        name="bal2",
        description="water balance in reservoir S2",
    )
    dec = Equation(
        m,
        domain=[n, t],
        name="dec",
        description="decisions of filling the reservoirs",
    )

    bal1[n, t].where[~tt[t]] = (
        s["res1", t] - s["res1", t - 1] == q["res1", t] + r2[t] - q2[t] - r["res1", t]
    )
    bal2[n, t].where[~tt[t]] = s["res2", t] - s["res2", t - 1] == q2[t] - r2[t]
    dec[n, t].where[~tt[t]] = (s["res2", t] - s["res1", t]) - (
        s["res2", t] - s["res1", t]
    ) * (1.0 - q2[t] / (q2[t] + 0.000001)) == 0.0

    # Objective Function
    objf = Sum(t.where[~tt[t]], r2[t])

    s.lo["res1", t] = 1150
    s.up["res1", t] = 4590
    s.fx["res1", "ian"] = 1200
    s.lo["res2", t] = 100
    s.up["res2", t] = 4590
    s.fx["res2", "ian"] = 1200
    r2.up[t] = 1500
    r2.lo[t] = 0.0
    q2.up[t] = 1500
    q2.lo[t] = 0.0
    q2.l[t] = 0.00001

    reservoir = Model(
        m,
        name="reservoir",
        equations=m.getEquations(),
        problem=Problem.NLP,
        sense=Sense.MIN,
        objective=objf,
    )
    reservoir.solve(solver="CONOPT")

    import math

    assert math.isclose(reservoir.objective_value, 81.0)

    print("Objective Function Value: ", reservoir.objective_value)


if __name__ == "__main__":
    main()
