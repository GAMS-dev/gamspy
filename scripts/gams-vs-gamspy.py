import sys
import os
import subprocess
from collections import defaultdict
from timeit import default_timer as timer

import gamspy_base

NUM_ITERS = 30

dir_path = os.path.dirname(os.path.realpath(__file__))

gams_path = gamspy_base.directory + "/" + "gams"

models = [
    "aircraft",
    "blend",
    "carseq",
    "cesam2",
    "chain",
    "chenery",
    "cpack",
    "cta",
    "cutstock",
    "dyncge",
    "fdesign",
    "flowshop",
    "food",
    "fuel",
    "gapmin",
    "hansmcp",
    "inscribedsquare",
    "iobalance",
    "korcns",
    "linear",
    "lop",
    "mexss",
    "minlphi",
    "minlphix",
    "partssupply",
    "process",
    "prodmix",
    "ps10_s_mn",
    "qdemo7",
    "qp6",
    "ramsey",
    "rcpsp",
    "robustlp",
    "rotdk",
    "sgolfer",
    "spatequ",
    "spbenders1",
    "springchain",
    "tanksize",
    "tforss",
    "thai",
    "timesteps",
    "trnsport",
    "trnspwl",
    "trussm",
    "tsp4",
    "weapons",
    "whouse",
]


def run_gams_models() -> dict:
    times = defaultdict(lambda: 0)

    for iter in range(NUM_ITERS):
        for model in models:
            start = timer()
            subprocess.run(
                [
                    gams_path,
                    f"{dir_path}/gams_models/{model}.gms",
                    "solprint=silent",
                ],
                stdout=subprocess.DEVNULL,
            )
            end = timer()
            times[model] += end - start

        print(f"[GAMS] ITERATION {iter} took: {sum(times.values())}")

    return times


def run_gamspy_models() -> dict:
    times = defaultdict(lambda: 0)

    for iter in range(NUM_ITERS):
        for model in models:
            start = timer()
            subprocess.run(
                [
                    "python",
                    f"{dir_path}/../tests/integration/models/{model}.py",
                ],
                stdout=subprocess.DEVNULL,
            )
            end = timer()
            times[model] += end - start

        print(f"[GAMSPy] ITERATION {iter} took: {sum(times.values())}")

    return times


def main():
    gams_times = run_gams_models()
    gamspy_times = run_gamspy_models()

    gams_time = sum(gams_times.values()) / NUM_ITERS
    gamspy_time = sum(gamspy_times.values()) / NUM_ITERS
    print(f"GAMS: {gams_time:.02}")
    print(f"GAMSPy: {(gamspy_time / NUM_ITERS):.02}")
    print(f"Diff: {(gamspy_time / gams_time):.02}")

    print("Biggest diffs:")

    diffs = {}
    for name, gams_time, gamspy_time in zip(models, gams_times.values(), gamspy_times.values()):
        diffs[name] = gamspy_time / gams_time

    for w in sorted(diffs, key=diffs.get, reverse=True):
        print(w, diffs[w])
    


if __name__ == "__main__":
    main()
