"""
Lagrangian Relaxation for Generalized Assignment (GAPMIN,SEQ=182)

A general assignment problem is solved via Lagrangian Relaxation
by dualizing the multiple choice constraints and solving
the remaining knapsack subproblems.

The data for this problem are taken from Martello.
The optimal value is 223 and the optimal solution is:
    1 1 4 2 3 5 1 4 3 5, where
in columns 1 and 2, the variable in the first row is equal to 1,
in column 3, the variable in the fourth row is equal to 1, etc...


Martello, S, and Toth, P, Knapsack Problems: Algorithms and Computer
Implementations. John Wiley and Sons, Chichester, 1990.

Guignard, M, and Rosenwein, M, An Improved Dual-Based Algorithm for
the Generalized Assignment Problem. Operations Research 37 (1989), 658-663.


 --- original model definition

Keywords: mixed integer linear programming, relaxed mixed integer linear
          programming, general assignment problem, lagrangian relaxation, knapsack
"""
import numpy as np

import gamspy.math as gams_math
from gamspy import Alias
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import ModelStatus
from gamspy import Number
from gamspy import Parameter
from gamspy import Set
from gamspy import Smax
from gamspy import Sum
from gamspy import Variable
from gamspy.exceptions import GamspyException


def sqr(x):
    return gams_math.power(x, 2)


def table_records():
    a_recs = np.array(
        [
            [12, 8, 25, 17, 19, 22, 6, 22, 20, 25],
            [5, 15, 15, 14, 7, 11, 14, 16, 17, 15],
            [21, 24, 13, 24, 12, 16, 23, 20, 15, 5],
            [23, 17, 10, 6, 24, 20, 15, 10, 19, 9],
            [17, 20, 15, 16, 5, 13, 7, 16, 8, 5],
        ]
    )

    f_recs = np.array(
        [
            [16, 26, 30, 47, 18, 19, 33, 37, 42, 31],
            [38, 42, 15, 21, 26, 11, 11, 50, 24, 19],
            [48, 17, 14, 22, 14, 18, 47, 32, 17, 42],
            [22, 32, 28, 39, 37, 23, 25, 12, 44, 17],
            [31, 42, 31, 40, 16, 15, 29, 31, 44, 41],
        ]
    )

    return a_recs, f_recs


def main():
    m = Container()

    # Original Model Definition

    # SETS
    i = Set(m, name="i", description="resources")
    j = Set(m, name="j", description="items")

    # VARIABLES
    x = Variable(
        m,
        name="x",
        type="binary",
        domain=[i, j],
        description="assignment of i to j",
    )
    z = Variable(
        m, name="z", type="free", description="total cost of assignment"
    )

    # EQUATIONS ##
    capacity = Equation(
        m, name="capacity", domain=[i], description="resource availability"
    )
    choice = Equation(
        m,
        name="choice",
        domain=[j],
        description="assignment constraint.. one resource per item",
    )
    defz = Equation(m, name="defz", description="definition of total cost")

    # PARAMETERS ##
    a = Parameter(
        m,
        name="a",
        domain=[i, j],
        description="utilization of resource i by item j",
    )
    f = Parameter(
        m,
        name="f",
        domain=[i, j],
        description="cost of assigning item j to resource i",
    )
    b = Parameter(m, name="b", domain=[i], description="available resources")

    capacity[i] = Sum(j, a[i, j] * x[i, j]) <= b[i]

    choice[j] = Sum(i, x[i, j]) == 1

    defz.expr = z == Sum([i, j], f[i, j] * x[i, j])

    _ = Model(
        m,
        name="assign_mip",
        equations=[capacity, choice, defz],
        problem="mip",
        sense="MIN",
        objective=z,
    )
    assign_rmip = Model(
        m,
        name="assign_rmip",
        equations=[capacity, choice, defz],
        problem="rmip",
        sense="MIN",
        objective=z,
    )

    # data for Martello model
    # SETS
    i.setRecords([f"r{r}" for r in range(1, 6)])
    j.setRecords([f"i{i}" for i in range(1, 11)])

    xopt = Set(
        m,
        name="xopt",
        domain=[i, j],
        records=[
            ("r1", "i1"),
            ("r1", "i2"),
            ("r1", "i7"),
            ("r2", "i4"),
            ("r3", "i5"),
            ("r3", "i9"),
            ("r4", "i3"),
            ("r4", "i8"),
            ("r5", "i6"),
            ("r5", "i10"),
        ],
        description="optimal assignment",
    )

    # PARAMETERS
    a.setRecords(table_records()[0])
    f.setRecords(table_records()[1])
    b.setRecords(np.array([28, 20, 27, 24, 19]))

    #############################################################
    # if one wants to check the data, one can
    # solve the MIP problem, this is just a check

    # m.addOptions({"optCr": 0})
    # assign_mip.solve()

    # check = 0
    # for r_loop, i_loop in xopt.toList():
    #     check += x.pivot().loc[r_loop, i_loop]

    # if check != 1:
    #     raise GamspyException('*** Something wrong with this solution', x.pivot(), xopt.pivot())
    #############################################################

    # Relaxed Problem Definition and Subgradient Optimization
    # Lagrangian subproblem definition
    # uses dynamic set to define WHICH knapsack to solve

    # Set
    id = Set(
        m,
        name="id",
        domain=[i],
        description="dynamic version of i used to define a subset of i",
    )
    iter = Set(
        m,
        name="iter",
        records=[f"iter{it}" for it in range(1, 21)],
        description="subgradient iteration index",
    )

    # Alias
    ii = Alias(m, name="ii", alias_with=i)

    # Parameters
    w = Parameter(
        m, name="w", domain=[j], description="Lagrangian multipliers"
    )
    improv = Parameter(
        m,
        name="improv",
        description=(
            "has the Lagrangian bound improved over the previous iterations"
        ),
    )

    # Variable
    zlrx = Variable(m, name="zlrx", description="relaxed objective")

    # Equations
    knapsack = Equation(
        m,
        name="knapsack",
        domain=[i],
        description="capacity with dynamic sets",
    )
    defzlrx = Equation(m, name="defzlrx", description="definition of zlrx")

    knapsack[id] = Sum(j, a[id, j] * x[id, j]) <= b[id]

    defzlrx.expr = zlrx == Sum([id, j], (f[id, j] - w[j]) * x[id, j])

    pknap = Model(
        m,
        name="pknap",
        equations=[knapsack, defzlrx],
        problem="mip",
        sense="MIN",
        objective=zlrx,
    )

    # Scalar
    target = Parameter(
        m, name="target", description="target objective function value"
    )
    alpha = Parameter(
        m, name="alpha", records=[1], description="step adjuster"
    )
    norm = Parameter(m, name="norm", description="norm of slacks")
    step = Parameter(
        m, name="step", records=[0], description="step size for subgradient"
    )
    zfeas = Parameter(
        m,
        name="zfeas",
        description="value for best known solution or valid upper bound",
    )
    zlr = Parameter(m, name="zlr", description="Lagrangian objective value")
    zl = Parameter(m, name="zl", description="Lagrangian objective value")
    zlbest = Parameter(
        m, name="zlbest", description="current best Lagrangian lower bound"
    )
    count = Parameter(
        m, name="count", description="count of iterations without improvement"
    )
    reset = Parameter(
        m, name="reset", records=[5], description="reset count counter"
    )
    tol = Parameter(
        m, name="tol", records=[1e-5], description="termination tolerance"
    )
    status = Parameter(
        m, name="status", records=[0], description="outer loop status"
    )

    # Parameter
    s = Parameter(m, name="s", domain=[j], description="slack variable")
    report = Parameter(
        m, name="report", domain=[iter, "*"], description="iteration log"
    )
    xrep = Parameter(
        m, name="xrep", domain=[j, i, "*"], description="x iteration report"
    )
    srep = Parameter(
        m, name="srep", domain=[iter, j], description="slack report"
    )
    wrep = Parameter(
        m, name="wrep", domain=[iter, j], description="w iteration report"
    )

    # calculate initial Lagrangian multipliers
    # There are many possible ways to find initial multipliers.
    # The choice of initial multipliers is very important for the
    # overall performance. The marginals of the relaxed problem
    # are often used to initialize the multipliers. Another choice
    # is simply to start with zero multipliers.

    # replace 'default' with solver of your choice.
    m.addOptions({"mip": "default", "rmip": "default"})

    results = open("solution", "w", encoding="UTF-8")
    results.write("solvers used: RMIP = HIGHS\n")
    results.write("               MIP = HIGHS\n")

    # solve relaxed problem to get initial multipliers
    # Note that different solvers get different dual solutions
    # which are not as good as a zero set of initial multipliers.

    assign_rmip.solve()
    results.write(f"\nRMIP objective value = {round(z.toValue(), 3)}\n")

    if assign_rmip.status == ModelStatus.OptimalGlobal:
        status.assign = 1  # everything is ok
    else:
        raise GamspyException(
            "*** relaxed MIP not optimal", "    no subgradient iterations", x
        )

    xrep[j, i, "initial"] = x.l[i, j]
    xrep[j, i, "optimal"] = Number(1).where[xopt[i, j]]

    # Parameter
    _ = Parameter(
        m,
        name="wopt",
        domain=[j],
        records=np.array([35, 40, 60, 69, 21, 49, 42, 47, 64, 46]),
        description="an optimal set of multipliers",
    )

    zlbest.assign = z.l

    # use RMIP duals
    w[j] = choice.m[j]

    # use optimal duals
    # w[j] = wopt[j]

    # use zero starting point
    # w[j]   = 0
    # zlbest.assign = 0

    results.write(
        "\n\nzlbest                    objective value  = "
        f" {round(zlbest.toValue(),3)}"
    )
    results.write("\n\nDual values on assignment constraint\n")

    for idx, j_loop in enumerate(j.toList()):
        results.write(
            f"\nw('{j_loop}') = {w.records.value.round(3).tolist()[idx]};"
        )

    # one needs a value for zfeas
    # one can compute a valid upper bound as follows:

    zfeas.assign = Sum(j, Smax(i, f[i, j]))
    results.write(
        "\n\nzfeas quick and dirty bound obj value      = "
        f" {round(zfeas.toValue(),3)}"
    )
    print("a priori upper bound", round(zfeas.toValue(), 3))

    ########################################################################
    # another alternative to compute a value for zfeas is
    # to solve gapmin by B-B and stop
    # at first 0-1 feasible solution found
    # using gapmin.optCr = 1, as follows

    # assign_mip.optCr = 1

    # assign_mip.solve()
    # zfeas.assign = gams_math.min(zfeas, z.l)
    # print('final zfeas', zfeas.toValue())
    # print('heuristic solution by B-B ', z.toValue(), "\n", x.pivot())
    # results.write(f"\nzfeas IP solution bound objective value    = {zfeas.toValue()}")
    #########################################################################

    results.write(
        "\n\n\n{:<15} {:<15} {:<20} {:<15} {:<15}\n".format(
            "Iteration", "New Bound", "Previous Bound", "norm", "abs(zl-zf)"
        )
    )

    # then keep the smaller of the two values as zfeas
    pknap.optCr = 0  # ask for global solution

    # ============================================================================ #
    #                                                                              #
    #   beginning of subgradient loop                                              #
    #                                                                              #
    # ============================================================================ #
    id[i] = False  # initially empty
    count.assign = 1
    alpha.assign = 1

    for iter_loop in iter.toList():
        if status.toValue() != 1:
            continue

        # solve Lagrangian subproblems by solving nonoverlapping knapsack
        # problems. Note the use of the dynamic set id[i] which will
        # contain the current knapsack descriptor.

        zlr.assign = 0
        for ii_loop in ii.toList():
            id[ii_loop] = True  # assume id was empty
            pknap.solve()
            zlr.assign = zlr + zlrx.l
            id[ii_loop] = False  # make set empty again

        improv.assign = 0
        zl.assign = zlr + Sum(j, w[j])
        improv.where[zl > zlbest] = 1  # is zl better than zlbest?
        zlbest.assign = gams_math.max(zlbest, zl)
        s[j] = 1 - Sum(i, x.l[i, j])  # subgradient
        norm.assign = Sum(j, sqr(s[j]))

        if norm.toValue() < tol.toValue():
            status.assign = 2

        if abs(zlbest.toValue() - zfeas.toValue()) < 1e-4:
            status.assign = 3

        if pknap.status != ModelStatus.OptimalGlobal:
            status.assign = 4

        row = [
            iter_loop,
            zl.toValue(),
            zlbest.toValue(),
            norm.toValue(),
            abs(zlbest.toValue() - zfeas.toValue()),
        ]
        results.write(
            "{:<15} {:<15.3f} {:<20.3f} {:<15.1f} {:<15.3f}\n".format(
                row[0], row[1], row[2], row[3], row[4]
            )
        )

        if status.toValue() == 2:
            results.write(
                "\n\nsubgr. method has converged, status ="
                f" {status.toValue()}\n\n"
            )
            results.write(
                "\n\nlast solution found is optimal for IP problem\n\n"
            )
        # end if
        if status.toValue() == 3:
            results.write(
                "\n\nsubgr. method has converged, status ="
                f" {status.toValue()}\n\n"
            )
            results.write("\n\nno duality gap, best sol. found is optimal\n\n")
        # end if
        if status.toValue() == 4:
            results.write("\n\nsomething wrong with last Lag. subproblem\n\n")
            results.write(f"\n\nstatus = {status.toValue()}\n\n")
        # end if

        report[iter_loop, "zlr"] = zlr
        report[iter_loop, "zl"] = zl
        report[iter_loop, "zlbest"] = zlbest
        report[iter_loop, "norm"] = norm
        report[iter_loop, "step"] = step

        wrep[iter_loop, j] = w[j]
        srep[iter_loop, j] = s[j]
        xrep[j, i, iter_loop] = x.l[i, j]

        if status.toValue() == 1:
            target.assign = (zlbest + zfeas) / 2
            step.assign = (alpha * (target - zl) / norm).where[norm > tol]
            w[j] = w[j] + step * s[j]
            if (
                count.toValue() > reset.toValue()
            ):  # too many iterations w/o improvement
                alpha.assign = alpha / 2
                count.assign = 1
            else:
                if improv.toValue():  # reset count if improvement
                    count.assign = 1
                else:
                    count.assign = count + 1  # update count if no improvement

        print(
            "iteration #: ",
            iter_loop,
            "\t\t Lagrangian bound: ",
            round(zl.toValue(), 3),
            "\t\t Best Lagrangian bound: ",
            round(zlbest.toValue(), 3),
        )

        # end loop iter

    results.write("\n\nDual values on assignment constraint\n")
    for idx, j_loop in enumerate(j.toList()):
        results.write(
            f"\nw('{j_loop}') = {w.records.value.round(3).tolist()[idx]};"
        )

    results.write(
        f"\n\nbest Lagrangian bound   =   {round(zlbest.toValue(),3)}"
    )
    results.close()

    print("\n\nreport: \n", report.pivot().round(3))
    print("\n\nwrep: \n", wrep.pivot().round(3))
    print("\n\nsrep: \n", srep.pivot().round(3))
    print("\n\nxrep: \n", xrep.pivot().round(3))


if __name__ == "__main__":
    main()
