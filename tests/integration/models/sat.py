import string

import gamspy as gp
import numpy as np

m = gp.Container()
c = gp.Set(m, "c", description="conjunctions", records=range(5))
d = gp.Set(m, "d", description="disjunctions", records=range(2))
j = gp.Set(m, "j", records=list(string.ascii_uppercase[:5]))

# CNF Representation of (D | E) & (A | ~D) & (B | ~C | ~D) & (~B | ~E) & (C | ~D)
# using numerical data -1|0|1
cnf_data = gp.Parameter(
    m,
    domain=[c, j],
    records=np.array(
        [  # A  B  C  D  E
            [0, 0, 0, 1, 1],
            [1, 0, 0, -1, 0],
            [0, 1, -1, -1, 0],
            [0, -1, 0, 0, -1],
            [0, 0, 1, -1, 0],
        ]
    ),
)

# via sympy's to_dnf (with simplify=True): (E & ~B & ~D) | (A & B & C & D & ~E)
dnf_data = gp.Parameter(
    m,
    domain=[d, j],
    records=np.array(
        [  # A  B  C  D  E
            [0, -1, 0, -1, 1],
            [1, 1, 1, 1, -1],
        ]
    ),
)

x = gp.Variable(m, domain=j, type="binary")

xc = gp.Variable(m, domain=c, type="binary")
def_conjunction = gp.Equation(m, domain=c)
def_conjunction[c] = xc[c] == gp.Sor(
    j.where[cnf_data[c, j] > 0], x[j]
) | gp.Sor(j.where[cnf_data[c, j] < 0], ~x[j])

xd = gp.Variable(m, domain=d, type="binary")
def_disjunction = gp.Equation(m, domain=d)
def_disjunction[d] = xd[d] == gp.Sand(
    j.where[dnf_data[d, j] > 0], x[j]
) | gp.Sand(j.where[dnf_data[d, j] < 0], ~x[j])

cnf = gp.Model(
    m,
    equations=[def_conjunction],
    objective=gp.Sand(c, xc[c]),
    problem="minlp",
    sense="max",
)
cnf.solve(solver="scip")

dnf = gp.Model(
    m,
    equations=[def_conjunction],
    objective=gp.Sand(c, xc[c]),
    problem="minlp",
    sense="max",
)
dnf.solve(solver="scip")

assert cnf.objective_value == dnf.objective_value
