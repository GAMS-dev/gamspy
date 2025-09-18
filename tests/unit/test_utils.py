from __future__ import annotations

import platform
import time

import pytest

import gamspy.utils as utils
from gamspy import Container, Set
from gamspy.exceptions import FatalError, ValidationError

pytestmark = pytest.mark.unit


@pytest.fixture
def data():
    m = Container()
    yield m
    m.close()


def test_utils(data):
    m = data
    i = Set(m, "i", records=["i1", "i2"])
    assert utils._get_domain_str([i, "b", "*"]) == '(i,"b",*)'
    pytest.raises(ValidationError, utils._get_domain_str, [5])

    # invalid system directory
    pytest.raises(FatalError, utils._open_gdx_file, "bla", "bla")

    assert not utils.checkAllSame([1, 2], [2])
    assert not utils.checkAllSame([1, 2], [2, 3])
    assert utils.checkAllSame([1, 2], [1, 2])

    # invalid load from path
    pytest.raises(FatalError, utils._open_gdx_file, m.system_directory, "bla.gdx")


def test_isin(data):
    m = data
    i = Set(m, "i")
    j = Set(m, "j")
    k = Set(m, "k")
    symbols = [i, j]

    assert utils.isin(i, symbols)
    assert not utils.isin(k, symbols)


def test_available_solvers(data):
    available_solvers = utils.getAvailableSolvers()

    if platform.system() == "Linux":
        expected = [
            "BARON",
            "CBC",
            "CONOPT3",
            "CONOPT4",
            "CONVERT",
            "COPT",
            "CPLEX",
            "DICOPT",
            "EXAMINER",
            "EXAMINER2",
            "GUROBI",
            "HIGHS",
            "IPOPT",
            "IPOPTH",
            "KESTREL",
            "KNITRO",
            "MILES",
            "MINOS",
            "MOSEK",
            "MPSGE",
            "NLPEC",
            "PATH",
            "PATHNLP",
            "RESHOP",
            "SBB",
            "SCIP",
            "SHOT",
            "SNOPT",
            "SOPLEX",
            "XPRESS",
        ]
    else:
        expected = [
            "BARON",
            "CBC",
            "CONOPT3",
            "CONOPT4",
            "CONVERT",
            "COPT",
            "CPLEX",
            "DICOPT",
            "EXAMINER",
            "EXAMINER2",
            "GUROBI",
            "HIGHS",
            "IPOPT",
            "IPOPTH",
            "KESTREL",
            "KNITRO",
            "MILES",
            "MINOS",
            "MOSEK",
            "MPSGE",
            "NLPEC",
            "PATH",
            "PATHNLP",
            "RESHOP",
            "SBB",
            "SBB",  # extra SBB required only for gamspy_base 51.1.0. will be deleted as soon as 51.2.0 is out.
            "SCIP",
            "SHOT",
            "SNOPT",
            "SOPLEX",
            "XPRESS",
        ]

    if platform.system() == "Linux" and platform.machine() == "aarch64":
        expected.remove("BARON")
        expected.remove("KNITRO")

    assert available_solvers == expected


def test_default_solvers():
    import gamspy_base

    default_solvers = utils.getDefaultSolvers(gamspy_base.directory)

    expected = {
        "CNS": "PATH",
        "DNLP": "CONOPT",
        "EMP": "CONVERT",
        "LP": "CPLEX",
        "MCP": "PATH",
        "MINLP": "SBB",
        "MIP": "CPLEX",
        "MIQCP": "SBB",
        "MPEC": "NLPEC",
        "NLP": "CONOPT",
        "QCP": "CONOPT",
        "RMINLP": "CONOPT",
        "RMIP": "CPLEX",
        "RMIQCP": "CONOPT",
    }

    assert default_solvers == expected


def test_solver_and_capability_caching():
    import gamspy_base

    start = time.perf_counter_ns()
    _ = utils.getDefaultSolvers(gamspy_base.directory)
    first_time = time.perf_counter_ns() - start

    start = time.perf_counter_ns()
    _ = utils.getDefaultSolvers(gamspy_base.directory)
    second_time = time.perf_counter_ns() - start

    assert second_time < first_time, f"{first_time=}, {second_time=}"

    start = time.perf_counter_ns()
    _ = utils.getSolverCapabilities(gamspy_base.directory)
    first_time = time.perf_counter_ns() - start

    start = time.perf_counter_ns()
    _ = utils.getSolverCapabilities(gamspy_base.directory)
    second_time = time.perf_counter_ns() - start

    assert second_time < first_time, f"{first_time=}, {second_time=}"

    start = time.perf_counter_ns()
    _ = utils.getInstalledSolvers(gamspy_base.directory)
    first_time = time.perf_counter_ns() - start

    start = time.perf_counter_ns()
    _ = utils.getInstalledSolvers(gamspy_base.directory)
    second_time = time.perf_counter_ns() - start

    assert second_time < first_time, f"{first_time=}, {second_time=}"
