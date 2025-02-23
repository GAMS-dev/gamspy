from __future__ import annotations

import doctest
import glob
import json
import os
import shutil
import time
from os.path import join

import gamspy_base
import pytest

import gamspy as gp
from gamspy.exceptions import GamspyException, ValidationError


@pytest.mark.unit
def test_version():
    assert gp.__version__ == "1.6.0"


@pytest.mark.unit
def test_config():
    m = gp.Container()
    assert m.system_directory == gamspy_base.directory

    i = gp.Set(m, records=["i1", "i2"])
    a = gp.Parameter(m, domain=i, records=[("i1", 1), ("i2", 2)])
    with pytest.raises(ValidationError):  # GAMSPy catches the domain violation
        a["i3"] = 5

    gp.set_options({"DOMAIN_VALIDATION": 0})
    with pytest.raises(GamspyException):  # GAMS returns domain violation error
        a["i3"] = 5

    gp.set_options({"DOMAIN_VALIDATION": 1})


@pytest.mark.unit
def test_domain_checking_config_performance():
    gp.set_options({"DOMAIN_VALIDATION": 1})
    m = gp.Container()
    i = gp.Set(m, records=range(999))
    a = gp.Parameter(m, domain=i)
    start = time.time()
    for idx in range(999):
        _ = (
            a[idx]
            + a[idx]
            + a[idx]
            + a[idx]
            + a[idx]
            + a[idx]
            + a[idx]
            + a[idx]
        )

    timing_with_validation = time.time() - start

    gp.set_options({"DOMAIN_VALIDATION": 0})
    m = gp.Container()
    i = gp.Set(m, records=range(999))
    a = gp.Parameter(m, domain=i)
    start = time.time()
    for idx in range(999):
        _ = (
            a[idx]
            + a[idx]
            + a[idx]
            + a[idx]
            + a[idx]
            + a[idx]
            + a[idx]
            + a[idx]
        )
    timing_without_validation = time.time() - start
    gp.set_options({"DOMAIN_VALIDATION": 1})

    print(f"{timing_with_validation=}, {timing_without_validation=}")
    assert timing_without_validation < timing_with_validation, (
        f"{timing_with_validation=}, {timing_without_validation=}"
    )


def test_map_special_values():
    m = gp.Container()
    a = gp.Parameter(m, "a")
    a[...] = gp.SpecialValues.EPS
    assert a.getAssignment() == "a = EPS;"

    gp.set_options({"MAP_SPECIAL_VALUES": 0})
    m = gp.Container()
    a = gp.Parameter(m, "a")
    a[...] = gp.SpecialValues.EPS
    assert a.getAssignment() == "a = -0.0;"
    gp.set_options({"MAP_SPECIAL_VALUES": 1})


@pytest.mark.doc
def test_switcher():
    this = os.path.dirname(os.path.abspath(__file__))
    root = this.rsplit(os.sep, maxsplit=1)[0]
    with open(join(root, "docs", "_static", "switcher.json")) as file:
        switcher = json.loads(file.read())
        versions = [elem["version"] for elem in switcher]
        assert f"v{gp.__version__}" in versions


@pytest.fixture
def teardown():
    # Act and assert
    yield

    # Cleanup
    files = glob.glob("_*")
    for file in files:
        os.remove(file)

    if os.path.exists("test"):
        shutil.rmtree("test")


@pytest.mark.doc
def test_docs():
    api_files = [
        file
        for file in glob.glob("src/**", recursive=True)
        if file.endswith(".py")
    ]

    for file in api_files:
        results = doctest.testfile(
            file,
            verbose=True,
            module_relative=False,
        )

        assert results.failed == 0
