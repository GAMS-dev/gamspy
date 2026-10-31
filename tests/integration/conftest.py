from __future__ import annotations

import glob
import os
import tempfile

import pytest


@pytest.fixture
def cleanup_generated_files():
    yield

    for file in glob.glob("_*"):
        if os.path.isfile(file):
            os.remove(file)


@pytest.fixture
def temp_license_path(monkeypatch, set_options):
    with tempfile.TemporaryDirectory() as directory:
        license_path = os.path.join(directory, "gamspy_license.txt")
        monkeypatch.setenv("GAMSPY_LICENSE_PATH", license_path)
        set_options({"LICENSE_PATH": license_path})
        yield license_path
