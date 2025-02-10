# Copyright (c) 2023: Yue Yang
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

import logging
import time

import numpy as np
import pandas as pd
import pyoptinterface as poi
from pyoptinterface import ipopt

import gamspy as gp

logging.disable(logging.WARNING)


def add_ndarray_variable(m, shape, **kwargs):
    array = np.empty(shape, dtype=object)
    array_flat = array.flat
    for i in range(array.size):
        array_flat[i] = m.add_variable(**kwargs)
    return array


def solve_facility_poi(m, G, F):
    # Create variables
    y = add_ndarray_variable(m, (F, 2), lb=0.0, ub=1.0)
    s = add_ndarray_variable(m, (G + 1, G + 1, F), lb=0.0)
    z = add_ndarray_variable(m, (G + 1, G + 1, F))
    r = add_ndarray_variable(m, (G + 1, G + 1, F, 2))
    d = m.add_variable()

    # Set objective
    m.set_objective(d * 1.0)

    # Add constraints
    for i in range(G + 1):
        for j in range(G + 1):
            expr = poi.quicksum(z[i, j, :])
            m.add_linear_constraint(expr, poi.Eq, 1.0)

    M = 2 * 1.414
    for i in range(G + 1):
        for j in range(G + 1):
            for f in range(F):
                expr = s[i, j, f] - d - M * (1 - z[i, j, f])
                m.add_linear_constraint(expr, poi.Eq, 0.0)
                expr = r[i, j, f, 0] - i / G + y[f, 0]
                m.add_linear_constraint(expr, poi.Eq, 0.0)
                expr = r[i, j, f, 1] - j / G + y[f, 1]
                m.add_linear_constraint(expr, poi.Eq, 0.0)
                m.add_second_order_cone_constraint(
                    [s[i, j, f], r[i, j, f, 0], r[i, j, f, 1]]
                )

    # Optimize model
    m.set_model_attribute(poi.ModelAttribute.Silent, True)
    m.set_model_attribute(poi.ModelAttribute.TimeLimitSec, 1e-9)

    m.optimize()


def solve_facility_gamspy(G, F):
    M = 2 * 1.414
    with gp.Container() as m:
        grid = gp.Set(records=range(G + 1))
        grid2 = gp.Alias(alias_with=grid)
        facs = gp.Set(records=range(1, F + 1))
        dims = gp.Set(records=range(1, 3))
        y = gp.Variable(domain=[facs, dims])
        y.lo = 0
        y.up = 1
        s = gp.Variable(domain=[grid, grid2, facs])
        s.lo = 0
        z = gp.Variable(domain=[grid, grid2, facs])
        r = gp.Variable(domain=[grid, grid2, facs, dims])
        d = gp.Variable()

        assmt = gp.Equation(domain=[grid, grid2])
        assmt[...] = gp.Sum(facs, z[grid, grid2, facs]) == 1

        quadrhs = gp.Equation(domain=[grid, grid2, facs])
        quadrhs[...] = s[grid, grid2, facs] == d + M * (
            1 - z[grid, grid2, facs]
        )

        quaddistk1 = gp.Equation(domain=[grid, grid2, facs])
        quaddistk1[...] = (
            r[grid, grid2, facs, "1"] == (1 * grid.val) / G - y[facs, "1"]
        )

        quaddistk2 = gp.Equation(domain=[grid, grid2, facs])
        quaddistk2[...] = (
            r[grid, grid2, facs, "2"] == (1 * grid2.val) / G - y[facs, "2"]
        )

        quaddist = gp.Equation(domain=[grid, grid2, facs])
        quaddist[...] = (
            r[grid, grid2, facs, "1"] ** 2 + r[grid, grid2, facs, "2"] ** 2
            <= s[grid, grid2, facs] ** 2
        )

        model = gp.Model(
            name="facility",
            equations=m.getEquations(),
            problem=gp.Problem.QCP,
            sense=gp.Sense.MIN,
            objective=d,
        )
        m.addGamsCode("facility.justscrdir = 0")
        model.solve(solver="ipopt", options=gp.Options(time_limit=1e-9))


def main(Ns=[25, 50, 75, 100]):
    results = []
    for n in Ns:
        poi_dict = dict()
        start = time.time()
        model = ipopt.Model()
        solve_facility_poi(model, n, n)
        timing = time.time() - start
        poi_dict["poi"] = timing
        results.append(poi_dict)
        print(f"[POI] {timing=}")

    gamspy_results = []
    df = pd.DataFrame(results, index=Ns)
    for n in Ns:
        start = time.time()
        solve_facility_gamspy(n, n)
        timing = time.time() - start
        gamspy_results.append(timing)
        print(f"[GAMSPy] {timing=}")

    df["gamspy"] = gamspy_results
    df["ratio"] = df["gamspy"] / df["poi"]
    print(df)


if __name__ == "__main__":
    main()
