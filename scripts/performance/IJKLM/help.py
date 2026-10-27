import json
import os
from collections.abc import Iterable, Iterator

import pandas as pd


def incremental_range(start: int, stop: int, step: int, inc: int) -> Iterator[int]:
    value = start
    while value < stop:
        yield value
        value += step
        step += inc


def create_data_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "I": [],
            "Language": [],
            "MeanTime": [],
            "MedianTime": [],
            "MinTime": [],
        }
    )


def save_to_json(symbol: Iterable, name: str, i: int, model: str) -> None:
    file = os.path.join(model, "data", f"data_{name}{i}.json")
    with open(file, "w") as f:
        json.dump(list(symbol), f)


def save_to_json_d(
    d: dict[tuple[str, str], float], name: str, i: int, model: str
) -> None:
    file = os.path.join(model, "data", f"data_{name}{i}.json")
    df = pd.DataFrame([(i, m, d[i, m]) for i, m in d], columns=["i", "m", "value"])
    df.to_json(file, orient="values")


def below_time_limit(df: pd.DataFrame, limit: float) -> bool:
    return (df["MinTime"].max() < limit) or (df.empty)


def process_results(r: pd.DataFrame, res_df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([res_df, r])


def print_log_message(language: str, n: int, df: pd.DataFrame) -> None:
    # define a standardized log
    log = "{language:<19} done {n:>6} in {time:>}s"
    print(
        log.format(
            language=language,
            n=n,
            time=round(df["MinTime"].mean(), 2),
        )
    )


def save_results(df: pd.DataFrame, model: str) -> None:
    file = os.path.join(model, "results", "experiment_results_solve.csv")
    df.pivot(index="I", columns="Language", values="MinTime").to_csv(file)
