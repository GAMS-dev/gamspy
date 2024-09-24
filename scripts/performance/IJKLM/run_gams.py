import os
import subprocess

import gams.transfer as gt
import gamspy as gp
import gamspy_base
import numpy as np
import pandas as pd


########## GAMS ##########
def data_to_gdx(I, J, K, L, M, ijk, jkl, klm):
    c = gt.Container(system_directory=gamspy_base.directory)

    # create sets
    i = c.addSet("i", records=I)
    j = c.addSet("j", records=J)
    k = c.addSet("k", records=K)
    l = c.addSet("l", records=L)
    m = c.addSet("m", records=M)

    c.addSet("IJK", [i, j, k], records=ijk[ijk["value"] == 1])
    c.addSet("JKL", [j, k, l], records=jkl[jkl["value"] == 1])
    c.addSet("KLM", [k, l, m], records=klm[klm["value"] == 1])

    # create parameter
    c.addParameter("time")

    # create variables
    c.addVariable("z")
    c.addVariable("x", domain=[i, j, k, l, m])

    c.write(os.path.join(os.path.dirname(__file__), "data", "data.gdx"))


def run_gams(N, repeats, number):
    process = subprocess.run(
        [
            os.path.join(gamspy_base.directory, "gams"),
            os.path.join(os.path.dirname(__file__), "IJKLM.gms"),
            f"--R={repeats}",
            f"--N={number}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert process.returncode == 0

    c = gp.Container()
    c.read(os.path.join(os.path.dirname(__file__), "results", "result.gdx"))
    r = c["t"].records["value"]
    c.close()

    result = pd.DataFrame(
        {
            "I": [N],
            "Language": ["GAMS"],
            "MinTime": [np.min(r)],
            "MeanTime": [np.mean(r)],
            "MedianTime": [np.median(r)],
        }
    )

    return result
