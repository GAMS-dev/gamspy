from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--iters", default=5, type=int)
parser.add_argument("--delayed", action="store_true")
args = parser.parse_args()

root_directory = Path(__file__).parent.parent
test_models_path = os.path.join(
    root_directory, "tests", "integration", "test_models.py"
)

commands = ["python", test_models_path]
env = os.environ.copy()
if args.delayed:
    env["DELAYED_EXECUTION"] = "1"
else:
    env["DELAYED_EXECUTION"] = "0"


results = []
for i in range(args.iters):
    start = time.time()
    subprocess.run(commands)
    end = time.time()
    print(f"[{i + 1}] took: {end - start}")
    results.append(end - start)

print(f"Took {sum(results) / args.iters} on averag")
