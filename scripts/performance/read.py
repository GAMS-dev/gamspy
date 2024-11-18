import argparse
import os
import timeit

import gams.transfer as gt

import gamspy as gp


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--num-iters")


def write_gdx(N):
    m = gt.Container()
    for idx in range(N):
        m.addSet(f"set{idx}", records=[f"i{i}" for i in range(N)])

    for idx in range(N):
        param = m.addParameter(f"parameter{idx}", domain=m[f"set{idx}"])
        param.generateRecords()

    for idx in range(N):
        var = m.addVariable(f"variable{idx}", domain=m[f"set{idx}"])
        var.generateRecords()

    m.write("in.gdx")


def run_gtp():
    _ = gt.Container(load_from="in.gdx")


def run_gp():
    _ = gp.Container(load_from="in.gdx")


def main():
    N = 500
    R = 10
    NUM = 1

    write_gdx(N)

    setup = {"func": run_gtp}
    r = timeit.repeat("func()", repeat=R, number=NUM, globals=setup)
    gt_time = sum(r) / len(r)

    setup = {"func": run_gp}
    r = timeit.repeat("func()", repeat=R, number=NUM, globals=setup)
    gp_time = sum(r) / len(r)

    print(f"{gt_time=}, {gp_time=}, ratio: {gp_time / gt_time: .2f}")

    os.remove("in.gdx")


if __name__ == "__main__":
    main()
