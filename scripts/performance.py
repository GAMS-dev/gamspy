from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--iters", default=5, type=int)
args = parser.parse_args()

root_directory = Path(__file__).parent.parent
test_models_path = os.path.join(
    root_directory, "tests", "integration", "test_models.py"
)

commands = ["python", test_models_path]

results = []
for i in range(args.iters):
    start = time.time()
    subprocess.run(commands, check=True)
    end = time.time()
    print(f"[{i + 1}] took: {end - start}")
    results.append(end - start)

print(f"Took {sum(results) / args.iters} on average")
