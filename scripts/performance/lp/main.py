import logging
import math
import time

import pandas as pd
import pyoptinterface as poi
from pyoptinterface import highs

import gamspy as gp

logging.disable(logging.WARNING)


def solve_poi_model(Model, N, time_limit):
    m = Model()
    x = m.add_m_variables((N, N))
    y = m.add_m_variables((N, N))
    for i in range(N):
        for j in range(N):
            m.add_linear_constraint(x[i, j] - y[i, j], poi.Geq, i)
            m.add_linear_constraint(x[i, j] + y[i, j], poi.Geq, 0)
    expr = poi.ExprBuilder()
    poi.quicksum_(expr, x.flat, lambda x: 2 * x)
    poi.quicksum_(expr, y.flat)
    m.set_objective(expr)
    m.set_model_attribute(poi.ModelAttribute.Silent, True)
    if time_limit is not None:
        m.set_model_attribute(poi.ModelAttribute.TimeLimitSec, time_limit)

    m.set_raw_parameter("mip_rel_gap", 0)
    m.set_raw_parameter("mip_abs_gap", 0)
    m.set_raw_parameter("presolve", "off")
    m.set_raw_parameter("random_seed", 999)

    m.optimize()
    objective_value = m.get_model_attribute(poi.ModelAttribute.ObjectiveValue)
    return objective_value


def solve_gamspy_model(N, time_limit=None):
    with gp.Container() as m:
        i = gp.Set(records=range(N))
        j = gp.Alias(alias_with=i)
        x = gp.Variable(domain=[i, j])
        y = gp.Variable(domain=[i, j])
        eq1 = gp.Equation(domain=[i, j])
        eq1[i, j] = x[i, j] - y[i, j] >= i.val
        eq2 = gp.Equation(domain=[i, j])
        eq2[i, j] = x[i, j] + y[i, j] >= 0
        obj = 2 * gp.Sum((i, j), x[i, j]) + gp.Sum((i, j), y[i, j])
        model = gp.Model(
            name="bench",
            equations=m.getEquations(),
            sense=gp.Sense.MIN,
            problem=gp.Problem.LP,
            objective=obj,
        )
        m.addGamsCode("bench.justscrdir = 0")
        model.solve(
            solver="HIGHS",
            options=gp.Options(
                time_limit=time_limit, relative_optimality_gap=0
            ),
            solver_options={"presolve": "off", "random_seed": 999},
        )

    return model.objective_value


def bench(N, time_limit):
    results = {}

    t0 = time.time()
    poi_objective = solve_poi_model(highs.Model, N, time_limit)
    t1 = time.time()
    results["poi"] = t1 - t0

    t0 = time.time()
    gamspy_objective = solve_gamspy_model(N, time_limit)
    t1 = time.time()
    results["gamspy"] = t1 - t0

    return results, poi_objective, gamspy_objective


def main():
    # sanity check with a small size
    results, poi_objective, gamspy_objective = bench(10, time_limit=1e10)
    assert math.isclose(poi_objective, gamspy_objective, abs_tol=1e-6), (
        f"{poi_objective=}, {gamspy_objective=}"
    )

    Ns = range(100, 501, 100)
    results = []
    for N in Ns:
        results.append(bench(N, 0)[0])
    # create a DataFrame
    df = pd.DataFrame(results, index=Ns)
    df["ratio"] = df["gamspy"] / df["poi"]

    # show result
    print(df)


if __name__ == "__main__":
    main()
