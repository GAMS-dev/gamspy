from __future__ import annotations

import glob
import os

import pytest


@pytest.fixture
def cleanup_generated_files():
    yield

    for file in glob.glob("_*"):
        if os.path.isfile(file):
            os.remove(file)
