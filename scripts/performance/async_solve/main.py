"""A scenario sweep benchmark of asynchronous local solves against sequential ones."""

from __future__ import annotations

import argparse
import math
import os
import random
import statistics
import sys
import time

import gamspy as gp


def build_model(
    container: gp.Container, num_rows: int, num_columns: int
) -> tuple[gp.Model, gp.Parameter, gp.Parameter]:
    """
    A market split instance, the Cornuejols-Dawande family. `num_rows` is the hardness
    knob: one more row costs roughly two orders of magnitude in solve time.
    """
    r = gp.Set(container, "r", records=[f"r{idx}" for idx in range(num_rows)])
    j = gp.Set(container, "j", records=[f"j{idx}" for idx in range(num_columns)])

    coefficient = gp.Parameter(container, "coefficient", domain=[r, j])
    target = gp.Parameter(container, "target", domain=r)

    x = gp.Variable(container, "x", domain=j, type="binary")
    over = gp.Variable(container, "over", domain=r, type="positive")
    under = gp.Variable(container, "under", domain=r, type="positive")

    split = gp.Equation(container, "split", domain=r)
    split[r] = gp.Sum(j, coefficient[r, j] * x[j]) + over[r] - under[r] == target[r]

    model = gp.Model(
        container,
        "market_split",
        equations=[split],
        problem="MIP",
        sense="MIN",
        objective=gp.Sum(r, over[r] + under[r]),
    )

    return model, coefficient, target


def make_scenarios(
    num_scenarios: int, num_rows: int, num_columns: int, seed: int
) -> list[dict]:
    rng = random.Random(seed)
    scenarios = []

    for _ in range(num_scenarios):
        coefficients = {
            (row, column): rng.randint(0, 99)
            for row in range(num_rows)
            for column in range(num_columns)
        }
        targets = [
            (
                f"r{row}",
                int(
                    0.5
                    * sum(coefficients[(row, column)] for column in range(num_columns))
                ),
            )
            for row in range(num_rows)
        ]

        scenarios.append(
            {
                "coefficient": [
                    (f"r{row}", f"j{column}", value)
                    for (row, column), value in coefficients.items()
                ],
                "target": targets,
            }
        )

    return scenarios


def apply_scenario(
    scenario: dict, coefficient: gp.Parameter, target: gp.Parameter
) -> None:
    coefficient.setRecords(scenario["coefficient"])
    target.setRecords(scenario["target"])


def run_sequential(scenarios: list[dict], args: argparse.Namespace) -> dict:
    """Solve the scenarios one after another with `Model.solve`."""
    options = gp.Options(time_limit=args.time_limit, threads=1)

    with gp.Container() as container:
        model, coefficient, target = build_model(container, args.rows, args.columns)

        objectives = []
        statuses = []
        solve_times = []

        start = time.perf_counter()
        for scenario in scenarios:
            apply_scenario(scenario, coefficient, target)

            solve_start = time.perf_counter()
            model.solve(solver=args.solver, options=options)
            solve_times.append(time.perf_counter() - solve_start)

            objectives.append(model.objective_value)
            statuses.append(model.solve_status)
        wall = time.perf_counter() - start

    return {
        "wall": wall,
        "solve_times": solve_times,
        "objectives": objectives,
        "statuses": statuses,
    }


def run_asynchronous(scenarios: list[dict], args: argparse.Namespace) -> dict:
    options = gp.Options(time_limit=args.time_limit, threads=1)

    with gp.Container() as container:
        model, coefficient, target = build_model(container, args.rows, args.columns)

        objectives = []
        statuses = []

        start = time.perf_counter()
        handles = []
        for scenario in scenarios:
            apply_scenario(scenario, coefficient, target)
            handles.append(model.solveAsync(solver=args.solver, options=options))
        submission = time.perf_counter() - start

        for handle in handles:
            handle.wait()
            objectives.append(model.objective_value)
            statuses.append(model.solve_status)
        wall = time.perf_counter() - start

    return {
        "wall": wall,
        "submission": submission,
        "collection": wall - submission,
        "objectives": objectives,
        "statuses": statuses,
    }


def report(sequential: dict, asynchronous: dict, args: argparse.Namespace) -> bool:
    solve_times = sequential["solve_times"]
    speedup = sequential["wall"] / asynchronous["wall"]

    print()
    print(
        f"{len(solve_times)} scenarios | {args.rows} rows x {args.columns} columns |"
        f" solver: {args.solver or 'default'} | threads per solve: 1 |"
        f" cores: {os.cpu_count()}"
    )
    print("=" * 74)
    print(f"{'Sequential (Model.solve)':<38}{sequential['wall']:>9.2f} s")
    print(
        f"{'  per solve (min / median / max)':<38}"
        f"{min(solve_times):>9.2f} /{statistics.median(solve_times):>7.2f} /"
        f"{max(solve_times):>7.2f} s"
    )
    print(f"{'Asynchronous (Model.solveAsync)':<38}{asynchronous['wall']:>9.2f} s")
    print(f"{'  submission (serial)':<38}{asynchronous['submission']:>9.2f} s")
    print(f"{'  collection':<38}{asynchronous['collection']:>9.2f} s")
    print("-" * 74)
    print(f"{'Speedup':<38}{speedup:>9.2f}x")

    matches = all(
        math.isclose(lhs, rhs, rel_tol=1e-6)
        for lhs, rhs in zip(sequential["objectives"], asynchronous["objectives"])
    )
    print(f"{'Objective values match':<38}{'yes' if matches else 'NO':>9}")

    interrupted = [
        status.name
        for phase in (sequential, asynchronous)
        for status in phase["statuses"]
        if status != gp.SolveStatus.NormalCompletion
    ]
    print(
        f"{'All solves completed normally':<38}{'yes' if not interrupted else 'NO':>9}"
    )

    return matches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=int, default=6)
    parser.add_argument(
        "--rows",
        type=int,
        default=4,
        help="hardness knob; 3 solves in well under a second, 5 takes minutes",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=None,
        help="defaults to 10 * (rows - 1), the standard market split shape",
    )
    parser.add_argument("--solver", default=None, help="defaults to the GAMSPy default")
    parser.add_argument(
        "--time-limit",
        type=float,
        default=300.0,
        help="per-solve time limit in seconds, as a safety net",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.columns is None:
        args.columns = 10 * (args.rows - 1)

    scenarios = make_scenarios(args.scenarios, args.rows, args.columns, args.seed)

    print(f"Solving {args.scenarios} scenarios sequentially...")
    sequential = run_sequential(scenarios, args)
    print(f"Solving {args.scenarios} scenarios asynchronously...")
    asynchronous = run_asynchronous(scenarios, args)

    return 0 if report(sequential, asynchronous, args) else 1


if __name__ == "__main__":
    sys.exit(main())
