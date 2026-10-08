import timeit

import linopy
import numpy as np
import pandas as pd
import xarray as xr


########## Linopy ##########
def run_linopy(I, IJK, JKL, KLM, repeats, number):
    setup = {
        "IJK": IJK[IJK["value"] == 1][["i", "j", "k"]],
        "JKL": JKL[JKL["value"] == 1][["j", "k", "l"]],
        "KLM": KLM[KLM["value"] == 1][["k", "l", "m"]],
        "model_function": linopy_model,
    }
    r = timeit.repeat(
        "model_function(IJK, JKL, KLM)",
        repeat=repeats,
        number=number,
        globals=setup,
    )

    r = [x / number for x in r]

    result = pd.DataFrame(
        {
            "I": [len(I)],
            "Language": ["Linopy"],
            "MinTime": [np.min(r)],
            "MeanTime": [np.mean(r)],
            "MedianTime": [np.median(r)],
        }
    )
    return result


def linopy_model(IJK, JKL, KLM):
    ijklm = IJK.merge(JKL, on=["j", "k"]).merge(KLM, on=["k", "l"])

    model = linopy.Model()

    # free variable to match GAMSPy's type="free"
    x = model.add_variables(coords=[pd.RangeIndex(len(ijklm), name="n")], name="x")

    i = xr.DataArray(ijklm["i"].to_numpy(), dims="n", name="i")
    model.add_constraints(x.groupby(i).sum() >= 0, name="ei")

    model.add_objective(0 * x)

    model.solve(
        solver_name="highs",
        io_api="direct",
        time_limit=0,
        output_flag=False,
        log_to_console=False,
    )
