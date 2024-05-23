import os
import subprocess
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


def run_gams_models() -> float:
    times = []

    for iter in NUM_ITERS:
        start = timer()
        for model in models:
            subprocess.run([gams_path, f"{dir_path}/{model}.gms"])
        end = timer()
        print(f"[GAMS] ITERATION {iter} took: {end - start}")
        times.append(end - start)

    return sum(times) / NUM_ITERS


def run_gamspy_models() -> float:
    times = []

    for _ in NUM_ITERS:
        start = timer()
        for model in models:
            subprocess.run(["python", f"{dir_path}/../tests/integration/models/{model}.py"])
        end = timer()
        print(f"[GAMSPy] ITERATION {iter} took: {end - start}")
        times.append(end - start)

    return sum(times) / NUM_ITERS


def main():
    gams_time = run_gams_models()
    gamspy_time = run_gamspy_models()

    print(f"GAMS: {gams_time:.02}")
    print(f"GAMSPy: {(gamspy_time / NUM_ITERS):.02}")
    print(f"Diff: {(gamspy_time / gams_time):.02}")


if __name__ == "__main__":
    main()
