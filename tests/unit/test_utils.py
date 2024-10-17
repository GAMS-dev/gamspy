from __future__ import annotations

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
    pytest.raises(
        Exception, utils._open_gdx_file, m.system_directory, "bla.gdx"
    )


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

    expected = [
        "BARON",
        "CBC",
        "CONOPT",
        "CONOPT3",
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

    assert available_solvers == expected
