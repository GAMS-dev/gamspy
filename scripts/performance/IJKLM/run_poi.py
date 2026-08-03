import timeit
from collections import defaultdict

import numpy as np
import pandas as pd
import pyoptinterface as poi
from pyoptinterface import highs


########## PyOptInterface ##########
def run_poi(I, ijk, jkl, klm, repeats, number):
    setup = {
        "I": I,
        "ijk": ijk,
        "jkl": jkl,
        "klm": klm,
        "model_function": poi_model,
    }
    r = timeit.repeat(
        "model_function(I, ijk, jkl, klm)",
        repeat=repeats,
        number=number,
        globals=setup,
    )

    r = [x / number for x in r]

    result = pd.DataFrame(
        {
            "I": [len(I)],
            "Language": ["PyOptInterface"],
            "MinTime": [np.min(r)],
            "MeanTime": [np.mean(r)],
            "MedianTime": [np.median(r)],
        }
    )
    return result


def poi_model(I, ijk, jkl, klm):
    model = highs.Model()
    model.set_model_attribute(poi.ModelAttribute.Silent, True)

    i_vars = defaultdict(list)
    for (i, j), ks in ijk.items():
        for k in ks:
            for l in jkl.get((j, k), []):
                for _m in klm.get((k, l), []):
                    # free variable to match GAMSPy's type="free"
                    var = model.add_variable(lb=float("-inf"), ub=float("inf"))
                    i_vars[i].append(var)

    model.set_objective(0, poi.ObjectiveSense.Minimize)

    for i in I:
        vars_i = i_vars.get(i, [])
        if vars_i:
            model.add_linear_constraint(poi.quicksum(vars_i), poi.Geq, 0)

    model.set_model_attribute(poi.ModelAttribute.TimeLimitSec, 0)
    model.optimize()
