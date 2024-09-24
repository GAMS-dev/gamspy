from collections import defaultdict

import numpy as np
import pandas as pd


########## Data ##########
def create_fixed_data(m):
    J = [f"j{x}" for x in range(1, m + 1)]
    K = [f"k{x}" for x in range(1, m + 1)]
    L = [f"l{x}" for x in range(1, m + 1)]
    M = [f"m{x}" for x in range(1, m + 1)]

    jkl = pd.DataFrame(
        np.random.binomial(1, 0.05, size=(len(J) * len(K) * len(L))),
        index=pd.MultiIndex.from_product([J, K, L], names=["j", "k", "l"]),
        columns=["value"],
    ).reset_index()
    klm = pd.DataFrame(
        np.random.binomial(1, 0.05, size=(len(K) * len(L) * len(M))),
        index=pd.MultiIndex.from_product([K, L, M], names=["k", "l", "m"]),
        columns=["value"],
    ).reset_index()

    return J, K, L, M, jkl, klm


def create_variable_data(n, j, k):
    i = [f"i{x}" for x in range(1, n + 1)]

    ijk = pd.DataFrame(
        np.random.binomial(1, 0.05, size=(len(i) * len(j) * len(k))),
        index=pd.MultiIndex.from_product([i, j, k], names=["i", "j", "k"]),
        columns=["value"],
    ).reset_index()

    return i, ijk


def fixed_data_to_tuples(JKL, KLM):
    jkl = [
        tuple(x)
        for x in JKL.loc[JKL["value"] == 1][["j", "k", "l"]].to_dict("split")[
            "data"
        ]
    ]
    klm = [
        tuple(x)
        for x in KLM.loc[KLM["value"] == 1][["k", "l", "m"]].to_dict("split")[
            "data"
        ]
    ]
    return jkl, klm


def variable_data_to_tuples(IJK):
    ijk = [
        tuple(x)
        for x in IJK.loc[IJK["value"] == 1][["i", "j", "k"]].to_dict("split")[
            "data"
        ]
    ]
    return ijk


def fixed_data_to_dicts(JKL, KLM):
    JKL_dict = defaultdict(list)
    KLM_dict = defaultdict(list)
    for j, k, l in JKL:
        JKL_dict[j, k].append(l)
    for k, l, m in KLM:
        KLM_dict[k, l].append(m)
    return JKL_dict, KLM_dict
