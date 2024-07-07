"""
## GAMSSOURCE: https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_tsp4.html
## LICENSETYPE: Demo
## MODELTYPE: MIP
## DATAFILES: tsp4.gdx
## KEYWORDS: mixed integer linear programming, traveling salesman problem, iterative subtour elimination


Traveling Salesman Problem - Four (TSP4,SEQ=180)

This is the fourth problem in a series of traveling salesman
problems. Here we revisit TSP1 and generate smarter cuts.
The first relaxation is the same as in TSP1.


Kalvelagen, E, Model Building with GAMS. forthcoming

de Wetering, A V, private communication.
"""

from __future__ import annotations

import os
from pathlib import Path

import gamspy.math as gams_math
from gamspy import (
    Alias,
    Container,
    Equation,
    Model,
    Options,
    Ord,
    Parameter,
    Set,
    Smax,
    Sum,
    Variable,
)
from gamspy.exceptions import GamspyException


def main():
    m = Container(
        system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None),
        load_from=str(Path(__file__).parent.absolute()) + "/tsp4.gdx",
    )

    # SETS
    ii, i = m.getSymbols(["ii", "i"])

    # ALIASES
    jj, j = m.getSymbols(["jj", "j"])

    # PARAMETER
    c = m.getSymbols(["c"])[0]

    # VARIABLES
    x = Variable(
        m,
        name="x",
        type="binary",
        domain=[ii, jj],
        description="decision variables - leg of trip",
    )

    # EQUATIONS
    rowsum = Equation(
        m, name="rowsum", domain=ii, description="leave each city only once"
    )
    colsum = Equation(
        m,
        name="colsum",
        domain=jj,
        description="arrive at each city only once",
    )

    # the assignment problem is a relaxation of the TSP
    rowsum[i] = Sum(j, x[i, j]) == 1
    colsum[j] = Sum(i, x[i, j]) == 1

    # Objective Function
    objective = Sum([i, j], c[i, j] * x[i, j])

    # exclude diagonal
    x.fx[ii, ii] = 0

    # For this algorithm we can try a larger subset of 12 cities.
    # SETS
    i.setRecords([f"i{ir}" for ir in range(1, 13)])

    # options. Make sure MIP solver finds global optima.
    assign = Model(
        m,
        name="assign",
        equations=[rowsum, colsum],
        problem="mip",
        sense="min",
        objective=objective,
    )
    assign.solve(options=Options(relative_optimality_gap=0))

    # find and display tours
    t = Set(
        m,
        name="t",
        records=[f"t{t}" for t in range(1, 18)],
        description="tours",
    )

    if len(t) < len(i):
        raise GamspyException("Set t is possibly too small")

    # SETS
    tour = Set(m, name="tour", domain=[i, j, t], description="subtours")
    visited = Set(
        m,
        name="visited",
        domain=i,
        description="flag whether a city is already visited",
    )

    # SINGLETON SETS
    fromi = Set(
        m,
        name="fromi",
        domain=i,
        is_singleton=True,
        description="contains always one element: the from city",
    )
    nextj = Set(
        m,
        name="nextj",
        domain=j,
        is_singleton=True,
        description="contains always one element: the to city",
    )
    tt = Set(
        m,
        name="tt",
        domain=t,
        is_singleton=True,
        description="contains always one element: the current subtour",
    )

    # ALIASES ##
    ix = Alias(m, name="ix", alias_with=i)

    # initialize
    fromi[i].where[Ord(i) == 1] = True  # turn first element on
    tt[t].where[Ord(t) == 1] = True  # turn first element on

    # subtour elimination by adding cuts
    # Set
    cc = Set(m, name="cc", records=[f"c{c}" for c in range(1, 1001)])

    # Alias
    ccc = Alias(m, name="ccc", alias_with=cc)  # we allow up to 1000 cuts

    # Set
    curcut = Set(
        m,
        name="curcut",
        domain=cc,
        description="current cut always one element",
    )
    allcuts = Set(m, name="allcuts", domain=cc, description="total cuts")

    # Parameter
    cutcoeff = Parameter(m, name="cutcoeff", domain=[cc, i, j])
    rhs = Parameter(m, name="rhs", domain=cc)
    nosubtours = Parameter(
        m, name="nosubtours", description="number of subtours"
    )

    # Equation
    cut = Equation(m, name="cut", domain=cc, description="dynamic cuts")

    cut[allcuts] = (
        Sum([i, j], cutcoeff[allcuts, i, j] * x[i, j]) <= rhs[allcuts]
    )

    tspcut = Model(
        m,
        name="tspcut",
        equations=[rowsum, colsum, cut],
        problem="mip",
        sense="MIN",
        objective=objective,
    )

    curcut[cc].where[Ord(cc) == 1] = True

    for ccc_loop in ccc.toList():
        #  initialize
        fromi[i].where[Ord(i) == 1] = True  # turn first element on
        tt[t].where[Ord(t) == 1] = True  # turn first element on
        tour[i, j, t] = False
        visited[i] = False

        for _ in i.toList():
            nextj[j].where[x.l[fromi, j] > 0.5] = (
                True  # check x.l(fromi,j) = 1 would be dangerous
            )
            tour[fromi, nextj, tt] = True  # store in table
            visited[fromi] = True  # mark city 'fromi' as visited
            fromi[j] = nextj[j]

            if nextj.toList()[0] in visited.toList():  # if already visited...
                tt[t] = tt[t.lag(1)]
                for ix_loop in ix.toList():
                    if (
                        ix_loop in visited.toList()
                    ):  # find starting point of new subtour
                        continue
                    fromi[ix_loop] = True

        nosubtours[...] = Sum(t, gams_math.Max(0, Smax(tour[i, j, t], 1)))

        if nosubtours.toValue() == 1:  # done: no subtours
            break

        # introduce cut
        for idx, t_loop in enumerate(t.toList()):
            if idx + 1 > nosubtours.toValue():
                continue
            rhs[curcut] = -1

            for i_loop, j_loop, t_loop2, _ in tour.records.itertuples(
                index=False
            ):
                if t_loop2 != t_loop:
                    continue

                cutcoeff[curcut, i_loop, j_loop].where[
                    x.l[i_loop, j_loop] > 0.5
                ] = 1
                # not needed due to nature of assignment constraints
                #        cutcoeff(curcut, i, j)$(x.l[i,j] < 0.5) = -1
                rhs[curcut] = rhs[curcut] + 1
            allcuts[curcut] = True  # include this cut in set
            curcut[cc] = curcut[cc.lag(1)]

        tspcut.solve()
        print(
            "Cut: ",
            ccc_loop,
            "\t\t # of subtours remaining: ",
            nosubtours.toValue() - 1,
        )

    if nosubtours.toValue() != 1:
        raise GamspyException("Too many cuts needed")

    print("No subtours remaining. Solution found!!\n")
    print("x: \n", x.pivot().round(1))

    import math

    assert math.isclose(tspcut.objective_value, 39, rel_tol=0.001)


if __name__ == "__main__":
    main()
