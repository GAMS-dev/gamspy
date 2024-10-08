import glob
import os
import subprocess
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
        os.remove(file)

    files = glob.glob("*.xlsx")
    for file in files:
        os.remove(file)

    mpsge_file_path = os.path.join(os.getcwd(), "HANSEN.GEN")
    if os.path.exists(mpsge_file_path):
        os.remove(mpsge_file_path)

    solution_file = os.path.join(os.getcwd(), "solution.txt")
    if os.path.exists(solution_file):
        os.remove(solution_file)


def test_full_models(teardown):
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
