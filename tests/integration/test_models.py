import glob
import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.model_library


def test_full_models():
    print("Running the models with the following license:")
    process = subprocess.run(
        ["gamspy", "show", "license"], capture_output=True
    )
    assert process.returncode == 0
    print(process.stdout)

    paths = glob.glob(
        str(Path(__file__).parent) + os.sep + "models" + os.sep + "*.py"
    )

    print()
    for idx, path in enumerate(paths):
        print(f"[{idx + 1}/{len(paths)}] {path.split(os.sep)[-1]}")
        process = subprocess.run(["python", path], capture_output=True)
        print(process.stdout.decode())
        print(process.stderr.decode())

        assert process.returncode == 0
