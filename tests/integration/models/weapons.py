"""
Weapons Assignment (WEAPONS)

This model determines an assignment of weapons to targets in order
to inflict maximum damage at minimal cost. This is a classic
NLP test problem.


Bracken, J, and McCormick, G P, Chapter 2. In Selected Applications of
Nonlinear Programming. John Wiley and Sons, New York, 1968, pp. 22-27.

Keywords: nonlinear programming, assignment problem, military application,
          nlp test problem
"""

from gamspy import Set, Parameter, Variable, Equation, Model, Container
from gamspy import Sum, Product, Card
import pandas as pd
from gamspy import Problem, Sense


def main():
    m = Container()

    td_data = pd.DataFrame(
        [
            ["icbm", "2", 0.05],
            ["icbm", "6", 0.15],
            ["icbm", "7", 0.10],
            ["icbm", "8", 0.15],
            ["icbm", "9", 0.20],
            ["icbm", "18", 0.05],
            ["mrbm-1", "1", 0.16],
            ["mrbm-1", "2", 0.17],
            ["mrbm-1", "3", 0.15],
            ["mrbm-1", "4", 0.16],
            ["mrbm-1", "5", 0.15],
            ["mrbm-1", "6", 0.19],
            ["mrbm-1", "7", 0.19],
            ["mrbm-1", "8", 0.18],
            ["mrbm-1", "9", 0.20],
            ["mrbm-1", "10", 0.14],
            ["mrbm-1", "12", 0.02],
            ["mrbm-1", "14", 0.12],
            ["mrbm-1", "15", 0.13],
            ["mrbm-1", "16", 0.12],
            ["mrbm-1", "17", 0.15],
            ["mrbm-1", "18", 0.16],
            ["mrbm-1", "19", 0.15],
            ["mrbm-1", "20", 0.15],
            ["lr-bomber", "1", 0.04],
            ["lr-bomber", "2", 0.05],
            ["lr-bomber", "3", 0.04],
            ["lr-bomber", "4", 0.04],
            ["lr-bomber", "5", 0.04],
            ["lr-bomber", "6", 0.10],
            ["lr-bomber", "7", 0.08],
            ["lr-bomber", "8", 0.09],
            ["lr-bomber", "9", 0.08],
            ["lr-bomber", "10", 0.05],
            ["lr-bomber", "11", 0.01],
            ["lr-bomber", "12", 0.02],
            ["lr-bomber", "13", 0.01],
            ["lr-bomber", "14", 0.02],
            ["lr-bomber", "15", 0.03],
            ["lr-bomber", "16", 0.02],
            ["lr-bomber", "17", 0.05],
            ["lr-bomber", "18", 0.08],
            ["lr-bomber", "19", 0.07],
            ["lr-bomber", "20", 0.08],
            ["f-bomber", "10", 0.04],
            ["f-bomber", "11", 0.09],
            ["f-bomber", "12", 0.08],
            ["f-bomber", "13", 0.09],
            ["f-bomber", "14", 0.08],
            ["f-bomber", "15", 0.02],
            ["f-bomber", "16", 0.07],
            ["mrbm-2", "1", 0.08],
            ["mrbm-2", "2", 0.06],
            ["mrbm-2", "3", 0.08],
            ["mrbm-2", "4", 0.05],
            ["mrbm-2", "5", 0.05],
            ["mrbm-2", "6", 0.02],
            ["mrbm-2", "7", 0.02],
            ["mrbm-2", "10", 0.10],
            ["mrbm-2", "11", 0.05],
            ["mrbm-2", "12", 0.04],
            ["mrbm-2", "13", 0.09],
            ["mrbm-2", "14", 0.02],
            ["mrbm-2", "15", 0.01],
            ["mrbm-2", "16", 0.01],
        ]
    )

    wa_data = pd.DataFrame(
        [
            ["icbm", 200],
            ["mrbm-1", 100],
            ["lr-bomber", 300],
            ["f-bomber", 150],
            ["mrbm-2", 250],
        ]
    )

    tm_data = pd.DataFrame(
        [
            ["1", 30],
            ["6", 100],
            ["10", 40],
            ["14", 50],
            ["15", 70],
            ["16", 35],
            ["20", 10],
        ]
    )

    mv_data = pd.DataFrame(
        [
            ["1", 60],
            ["2", 50],
            ["3", 50],
            ["4", 75],
            ["5", 40],
            ["6", 60],
            ["7", 35],
            ["8", 30],
            ["9", 25],
            ["10", 150],
            ["11", 30],
            ["12", 45],
            ["13", 125],
            ["14", 200],
            ["15", 200],
            ["16", 130],
            ["17", 100],
            ["18", 100],
            ["19", 100],
            ["20", 150],
        ]
    )

    # Sets
    w = Set(
        m,
        name="w",
        records=["icbm", "mrbm-1", "lr-bomber", "f-bomber", "mrbm-2"],
    )
    t = Set(m, name="t", records=[str(i) for i in range(1, 21)])

    # Parameters
    td = Parameter(m, name="td", domain=[w, t], records=td_data)
    wa = Parameter(m, name="wa", domain=[w], records=wa_data)
    tm = Parameter(m, name="tm", domain=[t], records=tm_data)
    mv = Parameter(m, name="mv", domain=[t], records=mv_data)

    # Variables
    x = Variable(m, name="x", domain=[w, t], type="Positive")
    prob = Variable(m, name="prob", domain=[t])
    tetd = Variable(m, name="tetd")

    # Equations
    maxw = Equation(m, name="maxw", domain=[w])
    minw = Equation(m, name="minw", domain=[t])
    probe = Equation(m, name="probe", domain=[t])
    etdp = Equation(m, name="etdp")
    etd = Equation(m, name="etd")

    maxw[w] = Sum(t.where[td[w, t]], x[w, t]) <= wa[w]
    minw[t].where[tm[t]] = Sum(w.where[td[w, t]], x[w, t]) >= tm[t]
    probe[t] = prob[t] == 1 - Product(
        w.where[td[w, t]], (1 - td[w, t]) ** x[w, t]
    )
    etdp.expr = tetd == Sum(t, mv[t] * prob[t])
    etd.expr = tetd == Sum(
        t, mv[t] * (1 - Product(w.where[td[w, t]], (1 - td[w, t]) ** x[w, t]))
    )

    war = Model(
        m,
        name="war",
        equations=[maxw, minw, etd],
        problem=Problem.NLP,
        sense=Sense.MAX,
        objective=tetd,
    )

    x.l[w, t].where[td[w, t]] = wa[w] / Card(t)

    war.solve()
    print(war.objective_value)


if __name__ == "__main__":
    main()
