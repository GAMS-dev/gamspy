"""
Onstream and offstream optimal reservoir management.

Adapted from:
McKinney, D.C., Savitsky, A.G., Basic optimization models for water and
energy management. June 1999 (revision 6, February 2003).
http://www.ce.utexas.edu/prof/mckynney/ce385d/papers/GAMS-Tutorial.pdf

Andrei, N., Optimal management of system of two reservoirs.
Revista Romana de Informatica si Automatica, vol.16, no.1, 2006, pp.15-18.
"""

from gamspy import Set, Parameter, Variable, Equation, Model, Container
from gamspy import Sum
import pandas as pd
from gamspy import Problem, Sense


def main():
    m = Container()

    # Set
    n = Set(m, name="n", records=["res1", "res2"])
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
    )
    tt = Set(m, name="tt", domain=[t], records=["ian"])

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
    )

    # Variable
    q2 = Variable(m, name="q2", domain=[t])
    r2 = Variable(m, name="r2", domain=[t])
    s = Variable(m, name="s", domain=[n, t])
    obj = Variable(m, name="obj")

    # Equation
    bal1 = Equation(m, domain=[n, t], name="bal1")
    bal2 = Equation(m, domain=[n, t], name="bal2")
    dec = Equation(m, domain=[n, t], name="dec")
    objf = Equation(m, name="objf")

    bal1[n, t].where[~tt[t]] = (
        s["res1", t] - s["res1", t.lag(1, "linear")]
        == q["res1", t] + r2[t] - q2[t] - r["res1", t]
    )
    bal2[n, t].where[~tt[t]] = (
        s["res2", t] - s["res2", t.lag(1, "linear")] == q2[t] - r2[t]
    )
    dec[n, t].where[~tt[t]] = (s["res2", t] - s["res1", t]) - (
        s["res2", t] - s["res1", t]
    ) * (1.0 - q2[t] / (q2[t] + 0.000001)) == 0.0
    objf.expr = obj == Sum(t.where[~tt[t]], r2[t])

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
        objective=obj,
    )
    reservoir.solve()


if __name__ == "__main__":
    main()
