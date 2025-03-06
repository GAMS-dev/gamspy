import glob
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.model_library


@pytest.fixture
def teardown():
    # Act and assert
    yield

    # Cleanup
    files = glob.glob("*.csv")
    for file in files:
        if os.path.isfile(file):
            os.remove(file)

    files = glob.glob("*.xlsx")
    for file in files:
        if os.path.isfile(file):
            os.remove(file)

    mpsge_file_path = os.path.join(os.getcwd(), "HANSEN.GEN")
    if os.path.exists(mpsge_file_path):
        os.remove(mpsge_file_path)

    solution_file = os.path.join(os.getcwd(), "solution.txt")
    if os.path.exists(solution_file):
        os.remove(solution_file)


def test_full_models(teardown):
    paths = glob.glob(
        os.path.join(str(Path(__file__).parent), "models", "*.py")
    )

    for idx, path in enumerate(paths):
        print(f"[{idx + 1}/{len(paths)}] {path.split(os.sep)[-1]}", flush=True)
        process = subprocess.run(
            [sys.executable, "-Wd", path], capture_output=True, text=True
        )
        assert process.returncode == 0, process.stderr
