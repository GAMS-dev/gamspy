import logging
import os
import sys
import timeit

import gams.transfer as gt
import gamspy as gp
import gamspy_base
import numpy as np
import pandas as pd

logging.disable(sys.maxsize)
import warnings  # noqa: E402

warnings.filterwarnings("ignore")


def run_gamspy(I, repeats, number):
    m = gt.Container(
        system_directory=gamspy_base.directory,
        load_from=os.path.join(os.path.dirname(__file__), "data", "data.gdx"),
    )
    gamspy_container = gp.Container(
        options=gp.Options(write_listing_file=False)
    )
    gamspy_container.addSet("i", records=m["i"].records)
    gamspy_container.addSet("j", records=m["j"].records)
    gamspy_container.addSet("k", records=m["k"].records)
    gamspy_container.addSet("l", records=m["l"].records)
    gamspy_container.addSet("m", records=m["m"].records)

    setup1 = {
        "c": gamspy_container,
        "IJK": m["ijk"].records,
        "JKL": m["jkl"].records,
        "KLM": m["klm"].records,
        "model_function": gamspy_model,
    }

    r1 = timeit.repeat(
        "model_function(c, IJK, JKL, KLM)",
        repeat=repeats,
        number=number,
        globals=setup1,
    )

    r1 = [r / number for r in r1]

    result = pd.DataFrame(
        {
            "I": [len(I)],
            "Language": ["GAMSPy"],
            "MinTime": [np.min(r1)],
            "MeanTime": [np.mean(r1)],
            "MedianTime": [np.median(r1)],
        }
    )

    return result


def gamspy_model(c, IJK, JKL, KLM):
    i, j, k, l, m = c["i"], c["j"], c["k"], c["l"], c["m"]
    ijk = gp.Set(c, "IJK", domain=[i, j, k], records=IJK)
    jkl = gp.Set(c, "JKL", domain=[j, k, l], records=JKL)
    klm = gp.Set(c, "KLM", domain=[k, l, m], records=KLM)

    x = gp.Variable(c, "x", type="free", domain=[i, j, k, l, m])

    ei = gp.Equation(c, "ei", domain=i)
    ei[i] = (
        gp.Sum((ijk[i, j, k], jkl[j, k, l], klm[k, l, m]), x[i, j, k, l, m])
        >= 0
    )

    model = gp.Model(c, "IJKLM_model", "LP", c.getEquations(), "FEASIBILITY")

    c.addGamsCode("IJKLM_model.JustScrDir = 1;")

    model.solve(
        options=gp.Options(
            report_solution=0,
            time_limit=0,
            equation_listing_limit=0,
            variable_listing_limit=0,
            write_listing_file=False,
            generate_name_dict=False,
        ),
    )
