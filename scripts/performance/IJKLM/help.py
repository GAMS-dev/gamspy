import json
import os

import pandas as pd


def incremental_range(start, stop, step, inc):
    value = start
    while value < stop:
        yield value
        value += step
        step += inc


def create_data_frame():
    return pd.DataFrame(
        {
            "I": [],
            "Language": [],
            "MeanTime": [],
            "MedianTime": [],
            "MinTime": [],
        }
    )


def create_directories(model):
    for d in ["data", "results"]:
        if not os.path.exists(os.path.join(model, d)):
            os.makedirs(os.path.join(model, d))


def save_to_json(symbol, name, i, model):
    file = os.path.join(model, "data", f"data_{name}{i}.json")
    with open(file, "w") as f:
        json.dump(list(symbol), f)


def save_to_json_d(d, name, i, model):
    file = os.path.join(model, "data", f"data_{name}{i}.json")
    df = pd.DataFrame(
        [(i, m, d[i, m]) for i, m in d], columns=["i", "m", "value"]
    )
    df.to_json(file, orient="values")


def below_time_limit(df, limit):
    return (df["MinTime"].max() < limit) or (df.empty)


def process_results(r, res_df):
    return pd.concat([res_df, r])


def print_log_message(language, n, df):
    # define a standardized log
    log = "{language:<19} done {n:>6} in {time:>}s"
    print(
        log.format(
            language=language,
            n=n,
            time=round(df["MinTime"].mean(), 2),
        )
    )


def save_results(df, model):
    file = os.path.join(model, "results", "experiment_results_solve.csv")
    df.pivot(index="I", columns="Language", values="MinTime").to_csv(file)
