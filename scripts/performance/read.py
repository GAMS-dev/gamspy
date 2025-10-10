import argparse
import os
import timeit

import gams.transfer as gt
import gamspy_base

import gamspy as gp


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--num-iters", default=500)
    return parser.parse_args()


def write_gdx(N: int, m: gt.Container):
    for idx in range(N):
        m.addSet(f"set{idx}", records=[f"i{i}" for i in range(N)])

    for idx in range(N):
        param = m.addParameter(f"parameter{idx}", domain=m[f"set{idx}"])
        param.generateRecords()

    for idx in range(N):
        var = m.addVariable(f"variable{idx}", domain=m[f"set{idx}"])
        var.generateRecords()

    m.write("in.gdx")


def run_gtp():  # read from a gdx file
    _ = gt.Container(system_directory=gamspy_base.directory, load_from="in.gdx")


def run_gp():  # read from a gdx file
    _ = gp.Container(load_from="in.gdx")


def run_gtp2(m: gt.Container):  # read from a Container
    _ = gt.Container(system_directory=gamspy_base.directory, load_from=m)


def run_gp2(m: gt.Container):  # read from a Container
    _ = gp.Container(load_from=m)


def main():
    args = get_args()
    N = args.num_iters
    R = 10
    NUM = 1

    m = gt.Container(system_directory=gamspy_base.directory)
    write_gdx(N, m)

    setup = {"func": run_gtp}
    r = timeit.repeat("func()", repeat=R, number=NUM, globals=setup)
    gt_time = sum(r) / len(r)

    setup = {"func": run_gp}
    r = timeit.repeat("func()", repeat=R, number=NUM, globals=setup)
    gp_time = sum(r) / len(r)

    print(f"{gt_time=}, {gp_time=}, ratio: {gp_time / gt_time: .2f}")

    setup = {"m": m, "func": run_gtp2}
    r = timeit.repeat("func(m)", repeat=R, number=NUM, globals=setup)
    gt_time = sum(r) / len(r)

    setup = {"m": m, "func": run_gp2}
    r = timeit.repeat("func(m)", repeat=R, number=NUM, globals=setup)
    gp_time = sum(r) / len(r)

    print(f"{gt_time=}, {gp_time=}, ratio: {gp_time / gt_time: .2f}")

    os.remove("in.gdx")


if __name__ == "__main__":
    main()
