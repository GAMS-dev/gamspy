import glob
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.model_library

MODELS = sorted(glob.glob(os.path.join(str(Path(__file__).parent), "models", "*.py")))

SINGLE_THREADED_LAUNCHER = """
import os, runpy, sys

from gamspy import Options

Options.model_fields["threads"].default = 1
Options.model_rebuild(force=True)

sys.argv = sys.argv[1:]
sys.path[0] = os.path.dirname(sys.argv[0])
runpy.run_path(sys.argv[0], run_name="__main__")
"""


@pytest.mark.parametrize("path", MODELS, ids=[Path(p).stem for p in MODELS])
def test_full_models(path, tmp_path):
    process = subprocess.run(
        [sys.executable, "-B", "-Wd", "-c", SINGLE_THREADED_LAUNCHER, path],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert process.returncode == 0, process.stderr
