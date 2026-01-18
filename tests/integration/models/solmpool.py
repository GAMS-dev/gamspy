"""
## GAMSSOURCE: https://gams.com/latest/gamslib_ml/libhtml/gamslib_solmpool.html
## LICENSETYPE: Demo
## MODELTYPE: MIP
## KEYWORDS: mixed integer linear programming, facility location problem, CPLEX, solution pool

A simple version of a facility location problem is used to show how the
solution pool and the tools associated with it work. This example is taken
from the Cplex 11 User's Manual (ILOG, Cplex 11 User's Manual, 2007)

A company is considering opening as many as four warehouses in order to serve
nine different regions. The goal is to minimize the sum of fixed costs
associated with opening warehouses as well as the various transportation
costs incurred to ship goods from the warehouses to the regions.

Whether or not to open a warehouse is represented by binary variable ow.
Whether or not to ship goods from warehouse i to region j is represented
by binary variable oa.

Each region needs a specified amount of goods, and each warehouse can store
only a limited quantity of goods. In addition, each region must be served
by exactly one warehouse.

The following GAMSPy program demonstrates a number of different
approaches to collecting solution pools. GAMSPy will store the solutions
in a merged GDX files which can then be further used by other programs
or the same GAMSPy run. GAMS/Cplex will have the variables with an extra
index as parameters in the merged solution file.
"""

import numpy as np

import gamspy as gp


def load_solution(m: gp.Container) -> None:
    m.loadRecordsFromGdx(
        "solnpool.gdx",
        symbol_names={
            "index": "solnpool",
            "oa": "oaX",
            "ow": "owX",
            "totcost": "totcostX",
            "tcost": "tcostX",
            "fcost": "fcostX",
        },
    )

    solnpool = m["solnpool"]
    num_solutions = len(solnpool)
    print(f"{num_solutions=}")

    xcostX = m["xcostX"]
    soln = m["soln"]
    u = m["u"]
    totcostX = m["totcostX"]
    tcostX = m["tcostX"]
    fcostX = m["fcostX"]

    xcostX[soln, u] = 0
    xcostX[solnpool, "totcost"] = totcostX[solnpool]
    xcostX[solnpool, "tcost"] = tcostX[solnpool]
    xcostX[solnpool, "fcost"] = fcostX[solnpool]
    xcostX[solnpool, "fcost^0.96"] = fcostX[solnpool] ** 0.96


def calculate_diff(oaX: gp.Parameter) -> int:
    aggdiff = 0
    oax_records = oaX.records.to_numpy()
    for row1 in oax_records:
        for row2 in oax_records:
            if row1[0] == row2[0] or row1[1] != row2[1] or row1[2] != row2[2]:
                continue

            aggdiff += np.logical_xor(row1[3], row2[3])

    return aggdiff


def main():
    m = gp.Container()
    i = gp.Set(m, records=[f"w{idx}" for idx in range(1, 5)])
    j = gp.Set(m, records=[f"r{idx}" for idx in range(1, 10)])
    f = gp.Parameter(m, domain=i, records=np.array([130, 150, 170, 180]))
    c = gp.Parameter(m, domain=i, records=np.array([90, 110, 130, 150]))
    d = gp.Parameter(
        m, domain=j, records=np.array([10, 10, 12, 15, 15, 15, 20, 20, 30])
    )
    t = gp.Parameter(
        m,
        domain=[j, i],
        records=np.array(
            [
                [10, 30, 25, 55],
                [10, 25, 25, 45],
                [20, 23, 30, 40],
                [25, 10, 26, 40],
                [28, 12, 20, 29],
                [36, 19, 16, 22],
                [40, 39, 22, 27],
                [75, 65, 55, 35],
                [34, 43, 41, 62],
            ]
        ),
    )
    totcost = gp.Variable(m)
    fcost = gp.Variable(m)
    tcost = gp.Variable(m)
    ow = gp.Variable(m, domain=i, type=gp.VariableType.BINARY)
    oa = gp.Variable(m, domain=[i, j], type=gp.VariableType.BINARY)
    _ = gp.Equation(m, name="deftotcost", definition=totcost == fcost + tcost)
    _ = gp.Equation(m, name="deffcost", definition=fcost == gp.Sum(i, f[i] * ow[i]))
    _ = gp.Equation(
        m, name="deftcost", definition=tcost == gp.Sum((i, j), t[j, i] * oa[i, j])
    )
    _ = gp.Equation(
        m, name="defwcap", domain=i, definition=gp.Sum(j, d[j] * oa[i, j]) <= c[i]
    )
    _ = gp.Equation(m, name="onew", domain=j, definition=gp.Sum(i, oa[i, j]) == 1)
    _ = gp.Equation(m, name="defow", domain=[i, j], definition=ow[i] >= oa[i, j])
    loc = gp.Model(m, equations=m.getEquations(), sense=gp.Sense.MIN, objective=totcost)

    soln = gp.Set(m, records=[f"soln_loc_p{idx}" for idx in range(1, 1001)])
    _ = gp.Set(m, name="solnpool", domain=soln)
    _ = gp.UniverseAlias(m, name="u")

    owX = gp.Parameter(m, domain=[soln, i])
    oaX = gp.Parameter(m, domain=[soln, i, j])
    _ = gp.Parameter(m, name="totcostX", domain=soln)
    _ = gp.Parameter(m, name="tcostX", domain=soln)
    _ = gp.Parameter(m, name="fcostX", domain=soln)
    xcostX = gp.Parameter(m, domain=[soln, "*"])

    # 1. Collect the incumbents found during the regular optimize procedure
    #    The Cplex option 'solnpool' triggers the collection of solutions in
    #    the GDX container solnpool.
    loc.solve(
        solver="cplex",
        options=gp.Options(relative_optimality_gap=0),
        solver_options={"solnpoolmerge": "solnpool.gdx"},
    )

    load_solution(m)

    # 2. Use the populate procedure instead of regular optimize procedure (option
    #    'solnpoolpop 2'). By default we will generate 20 solutions determined by
    #    the default of option populatelim (only valid for one thread). This is a
    #    simple model which is quickly solved with heuristics and cuts, so we need
    #    to let Cplex retain sufficient exploration space to find alternative solutions.
    #    This is done with option 'solnpoolintensity 4'. With solutions where the optimal
    #    solution can not so quickly be found, the default for this option should be suffict.
    loc.solve(
        solver="cplex",
        options=gp.Options(relative_optimality_gap=0),
        solver_options={
            "solnpoolmerge": "solnpool.gdx",
            "solnpoolintensity": 4,
            "solnpoolpop": 2,
        },
    )
    load_solution(m)
    print(xcostX.records)

    # 3. Lets look at the diversity of the solution by counting the differences
    #    between the shipment indicator variables. Lets limit the number of
    #    solutions in the pool by 10 and require solution within 10% of the
    #    optimum.
    loc.solve(
        solver="cplex",
        options=gp.Options(relative_optimality_gap=0),
        solver_options={
            "solnpoolmerge": "solnpool.gdx",
            "threads": 1,
            "solnpoolintensity": 4,
            "solnpoolpop": 2,
            "solnpoolcapacity": 10,
            "solnpoolgap": 0.1,
        },
    )
    load_solution(m)

    aggdiff = calculate_diff(oaX)

    # 4. We repeat the experiment by now setting the solution pool replacement
    #    strategy to 'diversity' and let the populate procedure find many more
    #    solutions, we should see an increase in the aggregated difference
    loc.solve(
        solver="cplex",
        options=gp.Options(relative_optimality_gap=0),
        solver_options={
            "solnpoolmerge": "solnpool.gdx",
            "threads": 1,
            "solnpoolintensity": 4,
            "solnpoolpop": 2,
            "solnpoolcapacity": 10,
            "solnpoolgap": 0.1,
            "populatelim": 10000,
            "solnpoolreplace": 2,
        },
    )
    load_solution(m)

    aggdiffX = calculate_diff(oaX)
    assert aggdiffX > aggdiff

    # 5. We can fine tune diversity by using a diversity filter. Suppose that
    #    facilities w1 and w2 are open. Let a solution keeping those two facilities
    #    open be the reference. We use a diversity filter to stipulate that any
    #    solution added to the solution pool must differ from the reference by
    #    decisions to open or close at least two other facilities. The following
    #    filter enforces this diversity by specifying a minimum diversity of 2.
    #    Note that the reference solution becomes the incumbent and is reported as
    #    the first solution in the pool.
    loc.solve(
        solver="cplex",
        options=gp.Options(relative_optimality_gap=0),
        solver_options={
            "solnpoolmerge": "solnpool.gdx",
            "threads": 1,
            "solnpoolintensity": 4,
            "solnpoolpop": 2,
            "divfltlo": 2,
            "writeflt": "ow2.flt",
            "ow.divflt('w1')": 1,
            "ow.divflt('w2')": 1,
            "ow.divflt('w3')": 0,
            "ow.divflt('w4')": 0,
        },
    )
    load_solution(m)
    print(owX.records)

    # 6. Solutions ordered best to worst.
    loc.solve(
        solver="cplex",
        options=gp.Options(relative_optimality_gap=0),
        solver_options={
            "solnpoolmerge": "solnpool.gdx",
            "threads": 1,
            "solnpoolintensity": 4,
            "solnpoolpop": 2,
        },
    )
    print(xcostX[soln, "totcost"].records)


if __name__ == "__main__":
    main()
