from __future__ import annotations

import doctest
import glob
import json
import os
import shutil
from os.path import join

import gamspy_base
import pytest

import gamspy as gp
from gamspy._config import OPTION_NAMES, configuration
from gamspy.exceptions import GamspyException, ValidationError


@pytest.mark.unit
def test_version():
    assert gp.__version__ == "1.27.0"


@pytest.mark.unit
def test_config(set_options):
    m = gp.Container()
    assert m.system_directory == gamspy_base.directory

    i = gp.Set(m, records=["i1", "i2"])
    a = gp.Parameter(m, domain=i, records=[("i1", 1), ("i2", 2)])
    with pytest.raises(TypeError):  # GAMSPy catches the domain violation
        a[a["bla"]] = 5

    set_options({"DOMAIN_VALIDATION": 0})
    with pytest.raises(GamspyException):  # GAMS returns domain violation error
        a["i3"] = 5

    set_options({"DOMAIN_VALIDATION": 1})

    e = gp.Equation(m, "e")
    model = gp.Model(m, "my_model", equations=[e])

    # no equation definition was found. ValidationError by default.
    with pytest.raises(ValidationError):
        model.solve()

    # no equation definition was found. Raises GamspyException because the validation is disabled.
    set_options({"VALIDATION": 0})
    with pytest.raises(GamspyException):
        model.solve()

    set_options({"VALIDATION": 1})

    # do not validate solver. This should raise GamspyException
    set_options({"SOLVER_VALIDATION": 0})
    m = gp.Container()
    v = gp.Variable(m)
    p = gp.Parameter(m)
    e = gp.Equation(m)
    e[...] = v <= p
    model = gp.Model(m)
    with pytest.raises(GamspyException):
        model.solve(solver="unknown")


@pytest.mark.unit
def test_set_options_rejects_unknown_name():
    with pytest.raises(ValidationError, match="Did you mean `DOMAIN_VALIDATION`"):
        gp.set_options({"DOMAIN_VALIDATIN": 0})

    with pytest.raises(ValidationError, match="not a valid GAMSPy option"):
        gp.set_options({"NOT_AN_OPTION": 42})

    assert "DOMAIN_VALIDATIN" not in configuration
    assert "NOT_AN_OPTION" not in configuration


@pytest.mark.unit
def test_set_options_does_not_partially_apply(set_options):
    before = gp.get_option("VALIDATION")

    # A single bad name must reject the whole call, not the bad entry only.
    with pytest.raises(ValidationError):
        gp.set_options({"VALIDATION": 0, "NOT_AN_OPTION": 42})

    assert gp.get_option("VALIDATION") == before


@pytest.mark.unit
def test_every_option_name_is_settable(set_options):
    for name in OPTION_NAMES:
        set_options({name: gp.get_option(name)})


@pytest.mark.unit
def test_option_names_cover_the_defaults():
    assert set(configuration) <= set(OPTION_NAMES)

    for name in OPTION_NAMES:
        assert gp.get_option(name) is not None


@pytest.mark.unit
def test_get_option_reports_unknown_name():
    with pytest.raises(KeyError, match="Did you mean `STRICT_POWER_OPERATOR`"):
        gp.get_option("STRICT_POWER_OPERATR")


@pytest.mark.unit
def test_map_special_values(set_options):
    m = gp.Container()
    a = gp.Parameter(m, "a")
    a[...] = gp.SpecialValues.EPS
    assert a.getAssignment() == "a = EPS;"

    set_options({"MAP_SPECIAL_VALUES": 0})
    m = gp.Container()
    a = gp.Parameter(m, "a")
    a[...] = gp.SpecialValues.EPS
    assert a.getAssignment() == "a = -0.0;"


@pytest.mark.doc
def test_switcher():
    this = os.path.dirname(os.path.abspath(__file__))
    root = this.rsplit(os.sep, maxsplit=1)[0]
    with open(join(root, "docs", "_static", "switcher.json")) as file:
        switcher = json.loads(file.read())
        versions = [elem["version"] for elem in switcher]
        assert f"v{gp.__version__}" in versions


@pytest.mark.doc
def test_every_option_is_documented():
    this = os.path.dirname(os.path.abspath(__file__))
    root = this.rsplit(os.sep, maxsplit=1)[0]
    table = join(root, "docs", "user", "advanced", "performance.rst")
    with open(table, encoding="utf-8") as file:
        content = file.read()

    for name in OPTION_NAMES:
        assert f"| {name} " in content, (
            f"`{name}` is missing from the package wide options table in {table}."
        )


@pytest.fixture
def teardown():
    # Act and assert
    yield

    # Cleanup
    files = glob.glob("_*")
    for file in files:
        if os.path.isfile(file):
            os.remove(file)

    if os.path.exists("test"):
        shutil.rmtree("test")


@pytest.mark.doc
def test_docs():
    api_files = [
        file for file in glob.glob("src/**", recursive=True) if file.endswith(".py")
    ]

    for file in api_files:
        results = doctest.testfile(
            file,
            verbose=True,
            module_relative=False,
        )

        assert results.failed == 0
