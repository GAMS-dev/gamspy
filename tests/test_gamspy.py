from __future__ import annotations

import doctest
import glob
import json
import os
from os.path import join

import gamspy
import pytest

try:
    from dotenv import load_dotenv

    load_dotenv(join(os.getcwd(), ".env"))
except Exception:
    pass


@pytest.mark.unit
def test_version():
    import gamspy

    assert gamspy.__version__ == "1.0.1"


@pytest.mark.doc
def test_switcher():
    this = os.path.dirname(os.path.abspath(__file__))
    root = this.rsplit(os.sep, maxsplit=1)[0]
    with open(join(root, "docs", "_static", "switcher.json")) as file:
        switcher = json.loads(file.read())
        versions = [elem["version"] for elem in switcher]
        assert f"v{gamspy.__version__}" in versions


@pytest.mark.doc
def test_docs():
    src_path = join(os.getcwd(), "src", "gamspy")
    api_files = [
        file
        for file in glob.glob("**", root_dir=src_path, recursive=True)
        if file.endswith(".py")
    ]

    for file in api_files:
        results = doctest.testfile(
            join(src_path, file),
            verbose=True,
            module_relative=False,
        )

        assert results.failed == 0
