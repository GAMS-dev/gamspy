import os
from collections import defaultdict

import IJKLM.data_generation as data
import numpy as np
import pandas as pd
from IJKLM.help import (
    below_time_limit,
    create_data_frame,
    incremental_range,
    print_log_message,
    process_results,
    save_results,
)
from IJKLM.run_gams import data_to_gdx, run_gams
from IJKLM.run_gamspy import run_gamspy
from IJKLM.run_poi import run_poi


############## Experiment ##########################
def run_experiment(cardinality_of_i, cardinality_of_j, repeats, number, time_limit):
    os.makedirs(os.path.join(os.path.dirname(__file__), "IJKLM", "data"), exist_ok=True)
    os.makedirs(
        os.path.join(os.path.dirname(__file__), "IJKLM", "results"),
        exist_ok=True,
    )

    np.random.seed(13)

    # create empty frames for results
    df_poi = create_data_frame()
    df_gams = create_data_frame()
    df_gamspy = create_data_frame()

    # define the x axis
    N = list(incremental_range(100000, cardinality_of_i + 1, 10000, 1000))

    # create fixed data
    J, K, L, M, JKL, KLM = data.create_fixed_data(m=cardinality_of_j)
    jkl_tuple, klm_tuple = data.fixed_data_to_tuples(JKL, KLM)
    jkl_dict, klm_dict = data.fixed_data_to_dicts(jkl_tuple, klm_tuple)

    # run experiment for every n in |I|
    for n in N:
        # create variable data and convert to tuples
        I, IJK = data.create_variable_data(n=n, j=J, k=K)
        ijk_tuple = data.variable_data_to_tuples(IJK)

        ijk_dict = defaultdict(list)
        for i, j, k in ijk_tuple:
            ijk_dict[i, j].append(k)

        # Pyoptinterface
        if below_time_limit(df_poi, time_limit):
            rr = run_poi(
                I, ijk_dict, jkl_dict, klm_dict, repeats=repeats, number=number
            )
            df_poi = process_results(rr, df_poi)
            print_log_message(language="PyOptInterface", n=n, df=df_poi)

        # GAMS
        if below_time_limit(df_gams, time_limit):
            data_to_gdx(I, J, K, L, M, IJK, JKL, KLM)
            rr = run_gams(n, repeats=repeats, number=number)
            df_gams = process_results(rr, df_gams)
            print_log_message(language="GAMS", n=n, df=df_gams)

        # GAMSPy
        if below_time_limit(df_gamspy, time_limit):
            data_to_gdx(I, J, K, L, M, IJK, JKL, KLM)
            gamspy = run_gamspy(
                I=I,
                repeats=repeats,
                number=number,
            )
            df_gamspy = process_results(gamspy, df_gamspy)
            print_log_message(language="GAMSPy", n=n, df=df_gamspy)

    # merge all results
    df = pd.concat([df_poi, df_gams, df_gamspy]).reset_index(drop=True)

    # save results
    save_results(df, os.path.join(os.path.dirname(__file__), "IJKLM"))

    return df


if __name__ == "__main__":
    CI = 200000
    CJ = 20

    df = run_experiment(
        cardinality_of_i=CI,
        cardinality_of_j=CJ,
        repeats=5,
        number=1,
        time_limit=60,
    )

    print(
        df.pivot(
            index="I",
            columns="Language",
            values=["MeanTime", "MedianTime", "MinTime"],
        )
    )
