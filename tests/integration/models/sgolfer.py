"""
## GAMSSOURCE: https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_sgolfer.html
## LICENSETYPE: Requires license
## MODELTYPE: MINLP, MIP
## KEYWORDS: mixed integer linear programming, mixed integer nonlinear programming, social golfer problem, combinatorial optimization


Social Golfer Problem (SGOLFER)

In a golf club, there are 32 social golfers, each of whom play golf once a
week,
and always in groups of 4. The problem is to build a schedule of play for 10
weeks with ''maximum socialisation''; that is, as few repeated meetings as
possible. More generally the problem is to schedule m groups of n golfers over
p weeks, with maximum socialisation.


Warwick, H, The Fully Social Golfer Problem. In Smith, B, and Warwick, H, Eds,
Proceedings of the Third International Workshop on Symmetry in Constraint
Satisfaction Problems (SymCon 2003). 2003, pp. 75-85.
"""

from __future__ import annotations

import os

from gamspy import Alias
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Ord
from gamspy import Sense
from gamspy import Set
from gamspy import Sum
from gamspy import Variable
from gamspy.math import Max as gams_max


def main(gr_c=8, gg_c=4, nw_c=10, mip=False):
    cont = Container(
        system_directory=os.getenv("SYSTEM_DIRECTORY", None),
        delayed_execution=int(os.getenv("DELAYED_EXECUTION", False)),
    )

    gf_c = gr_c * gg_c

    # Set
    gf = Set(
        cont,
        name="gf",
        records=[str(i) for i in range(1, gf_c + 1)],
        description="golfers",
    )
    gr = Set(
        cont,
        name="gr",
        records=[str(i) for i in range(1, gr_c + 1)],
        description="groups",
    )
    w = Set(
        cont,
        name="w",
        records=[str(i) for i in range(1, nw_c + 1)],
        description="weeks",
    )

    # Alias
    gf1 = Alias(cont, name="gf1", alias_with=gf)
    gf2 = Alias(cont, name="gf2", alias_with=gf)

    mgf = Set(
        cont, name="mgf", domain=[gf1, gf2], description="possible meeting"
    )
    mgf[gf1, gf2] = Ord(gf2) > Ord(gf1)

    # Variable
    x = Variable(
        cont,
        name="x",
        type="binary",
        domain=[w, gr, gf],
        description="golfer gf is in group gr on week w",
    )
    m = Variable(
        cont,
        name="m",
        type="free",
        domain=[w, gr, gf, gf],
        description="golfers meet in week w in some group",
    )
    numm = Variable(
        cont,
        name="numm",
        type="free",
        domain=[gf, gf],
        description="number of meetings",
    )
    redm = Variable(
        cont,
        name="redm",
        type="free",
        domain=[gf, gf],
        description="number of redundant meetings",
    )

    # Equation
    defx = Equation(
        cont,
        name="defx",
        domain=[w, gf],
        description="each golfer is assigned to exactly one group",
    )
    defgr = Equation(
        cont,
        name="defgr",
        domain=[w, gr],
        description="each group contains exactly |gg| golfers",
    )
    defm = Equation(
        cont,
        name="defm",
        domain=[w, gr, gf, gf],
        description="meet in group gr on week w",
    )
    defnumm = Equation(
        cont, name="defnumm", domain=[gf, gf], description="number of meetings"
    )
    defredm = Equation(
        cont,
        name="defredm",
        domain=[gf, gf],
        description="number of redundant meetings",
    )

    if not isinstance(mip, bool):
        raise Exception(
            f"Argument <mip> should be a boolean. Not {type(mip)}."
        )

    if mip:
        m.type = "binary"
        redm.type = "positive"

        defm2 = Equation(
            cont,
            name="defm2",
            domain=[w, gr, gf, gf],
            description="meet in group gr on week w",
        )
        defm3 = Equation(
            cont,
            name="defm3",
            domain=[w, gr, gf, gf],
            description="meet in group gr on week w",
        )

        defm[w, gr, mgf[gf1, gf2]] = m[w, gr, mgf] <= x[w, gr, gf1]

        defm2[w, gr, mgf[gf1, gf2]] = m[w, gr, mgf] <= x[w, gr, gf2]

        defm3[w, gr, mgf[gf1, gf2]] = (
            m[w, gr, mgf] >= x[w, gr, gf1] + x[w, gr, gf2] - 1
        )

        defredm[mgf] = redm[mgf] >= numm[mgf] - 1

    else:
        defm[w, gr, mgf[gf1, gf2]] = (
            m[w, gr, mgf] == x[w, gr, gf1] & x[w, gr, gf2]
        )

        defredm[mgf] = redm[mgf] == gams_max(0, numm[mgf] - 1)

    defx[w, gf] = Sum(gr, x[w, gr, gf]) == 1

    defgr[w, gr] = Sum(gf, x[w, gr, gf]) == gg_c

    defnumm[mgf] = numm[mgf] == Sum((w, gr), m[w, gr, mgf])

    x.l[w, gr, gf].where[
        ((Ord(gr) - 1) * gg_c + 1 <= Ord(gf)) & (Ord(gf) <= (Ord(gr)) * gg_c)
    ] = 1

    defobj = Sum(mgf, redm[mgf])  # minimize redundant meetings

    social_golfer_mip = Model(
        cont,
        name="social_golfer_mip",
        equations=cont.getEquations(),
        problem="mip",
        sense=Sense.MIN,
        objective=defobj,
    )

    social_golfer_minlp = Model(
        cont,
        name="social_golfer_minlp",
        equations=cont.getEquations(),
        problem="minlp",
        sense=Sense.MIN,
        objective=defobj,
    )

    if mip:
        social_golfer_mip.solve()
        obj_val = social_golfer_mip.objective_value
    else:
        social_golfer_minlp.solve()
        obj_val = social_golfer_minlp.objective_value

    print("Objective Function Variable: ", obj_val)


if __name__ == "__main__":
    main()
