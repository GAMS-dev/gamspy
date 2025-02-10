import time

import pandas as pd
import pyoptinterface as poi
from pyoptinterface import highs

import gamspy as gp


def create_poi_model(Model, N):
    m = Model()
    x = m.add_variables(range(N), range(N))
    y = m.add_variables(range(N), range(N))
    for i in range(N):
        for j in range(N):
            m.add_linear_constraint(x[i, j] - y[i, j], poi.Geq, i)
            m.add_linear_constraint(x[i, j] + y[i, j], poi.Geq, 0)
    expr = poi.ExprBuilder()
    poi.quicksum_(expr, x, lambda x: 2 * x)
    poi.quicksum_(expr, y)
    m.set_objective(expr)
    m.set_model_attribute(poi.ModelAttribute.Silent, True)
    m.set_model_attribute(poi.ModelAttribute.TimeLimitSec, 0.0)
    return m


def create_gamspy_model(N):
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

    return model


def bench(N, solver_name):
    results = {}

    t0 = time.time()
    model = create_poi_model(highs.Model, N)
    model.optimize()
    t1 = time.time()
    results["n_variables"] = 2 * N * N
    results["poi"] = t1 - t0

    t0 = time.time()
    model = create_gamspy_model(N)
    model.solve(solver="HIGHS", options=gp.Options(time_limit=0))
    t1 = time.time()
    results["gamspy"] = t1 - t0

    return results


def main(solver_name="gurobi"):
    Ns = range(100, 501, 100)
    results = []
    for N in Ns:
        results.append(bench(N, solver_name))
    # create a DataFrame
    df = pd.DataFrame(results, index=Ns)
    df["ratio"] = df["poi"] / df["gamspy"]

    # show result
    print(df)


if __name__ == "__main__":
    # solver_name can be "copt", "gurobi", "highs"
    main("highs")
