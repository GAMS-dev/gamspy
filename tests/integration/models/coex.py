"""
## GAMSSOURCE: https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_coex.html
## LICENSETYPE: Demo
## MODELTYPE: MIP
## KEYWORDS: mixed integer linear programming, mathematical games, combinatorial optimization, peaceably coexisting armies of queens

Peacefully Coexisting Armies of Queens (COEX)

Two armies of queens (black and white) peacefully coexist on a
chessboard when they are placed on the board in such a way that
no two queens from opposing armies can attack each other. The
problem is to find the maximum two equal-sized armies.

Bosch, R, Mind Sharpener. OPTIMA MPS Newsletter (2000).
"""

from sys import stdout

import gamspy.math as gpm
from gamspy import (
    Alias,
    Container,
    Equation,
    Model,
    Ord,
    Problem,
    Sense,
    Set,
    Sum,
    Variable,
    VariableType,
)

m = Container()

i = Set(
    m,
    "i",
    description="size of chess board",
    records=[str(i + 1) for i in range(8)],
)
j, ii, jj = (Alias(m, ident, i) for ident in ["j", "ii", "jj"])

M = Set(
    m, "M", domain=[i, j, ii, jj], description="shared positions on the board"
)
M[i, j, ii, jj] = (
    (Ord(i) == Ord(ii))
    | (Ord(j) == Ord(jj))
    | (gpm.abs(Ord(i) - Ord(ii)) == gpm.abs(Ord(j) - Ord(jj)))
)

b = Variable(
    m,
    "b",
    domain=[i, j],
    type=VariableType.BINARY,
    description="square occupied by a black queen",
)
w = Variable(
    m,
    "w",
    domain=[i, j],
    type=VariableType.BINARY,
    description="square occupied by a white queen",
)

tot = Variable(m, "tot", description="total queens in each army")

eq1 = Equation(
    m, "eq1", domain=[i, j, ii, jj], description="keep armies at peace"
)
eq1[M[i, j, ii, jj]] = b[i, j] + w[ii, jj] <= 1
eq2 = Equation(
    m,
    "eq2",
    description="add up all the black queens",
    definition=tot == Sum((i, j), b[i, j]),
)
eq3 = Equation(
    m,
    "eq3",
    description="add up all the white queens",
    definition=tot == Sum((i, j), w[i, j]),
)

armies = Model(
    m,
    "armies",
    problem=Problem.MIP,
    equations=m.getEquations(),
    sense=Sense.MAX,
    objective=tot,
)
armies.solve(output=stdout)
# Display solution
print(f"Army size: {int(armies.objective_value)}")


def queen_pos(var: Variable):
    res_dict = var.toDict()
    return [
        (int(coord[0]), int(coord[1]))
        for coord, v in res_dict.items()
        if v > 0
    ]


bpos, wpos = queen_pos(b), queen_pos(w)
print(f"Black queen positions: {bpos}\nWhite queen positions: {wpos}")


# Validate solution
def can_attack(ar, ac, br, bc):
    return ar == br or ac == bc or abs(ar - br) == abs(ac - bc)


assert int(armies.objective_value) == 9
for ar, ac in bpos:
    for br, bc in wpos:
        assert not can_attack(ar, ac, br, bc)
