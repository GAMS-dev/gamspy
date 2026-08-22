import glob
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.model_library

MODELS = sorted(glob.glob(os.path.join(str(Path(__file__).parent), "models", "*.py")))


@pytest.mark.parametrize("path", MODELS, ids=[Path(p).stem for p in MODELS])
def test_full_models(path, tmp_path):
    process = subprocess.run(
        [sys.executable, "-B", "-Wd", path],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert process.returncode == 0, process.stderr
