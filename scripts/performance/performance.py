import glob
import os
import pstats
import subprocess
import sys
from pathlib import Path

import pandas as pd

base_dir = Path(__file__).parent.parent.parent
usual_suspects = [
    "main",
    "preprocess",
    "execute_gams",
    "postprocess",
    "_send_job",
]


def load_pstats_as_dict(file_path):
    stats = pstats.Stats(file_path)  # Load the .pstats file
    stats_data = {}

    for func, func_stats in stats.stats.items():
        func_name = func[2]
        cumtime = func_stats[3]
        if func_name in usual_suspects:
            stats_data[func_name] = cumtime

    return stats_data


def main():
    all_models = glob.glob(
        os.path.join(base_dir, "tests", "integration", "models", "*.py")
    )

    final_df = pd.DataFrame(
        index=[
            "preprocess",
            "execute_gams",
            "postprocess",
            "_send_job",
            "overhead",
        ]
    )
    for model in all_models:
        model_name = model.split("/")[-1][:-3]
        print(f"Running {model_name}...")
        process = subprocess.run(
            [sys.executable, "-m", "cProfile", "-o", "profile.pstats", model],
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0, process.stderr

        pstats_dict = load_pstats_as_dict("profile.pstats")
        main_time = pstats_dict.pop("main")
        for key, value in pstats_dict.items():
            pstats_dict[key] = value / main_time

        pstats_dict["overhead"] = (
            pstats_dict["preprocess"]
            + pstats_dict["execute_gams"]
            + pstats_dict["postprocess"]
            - pstats_dict["_send_job"]
        )
        df = pd.DataFrame(
            pstats_dict.values(),
            index=pstats_dict.keys(),
            columns=[model_name],
        )

        final_df[model_name] = df[model_name]

    final_df = final_df.T.round(2)
    print(final_df)
    final_df.to_csv("stats.csv")


if __name__ == "__main__":
    main()
