from __future__ import annotations

import pytest

from gamspy.exceptions import (
    EngineClientException,
    EngineException,
    FatalError,
    GamspyBaseException,
    GamspyException,
    GdxException,
    GmdException,
    LatexException,
    NeosClientException,
    NeosException,
    ValidationError,
)

pytestmark = pytest.mark.unit

ALL_EXCEPTIONS = (
    EngineClientException,
    EngineException,
    FatalError,
    GamspyException,
    GdxException,
    GmdException,
    LatexException,
    NeosClientException,
    NeosException,
    ValidationError,
)

GAMS_EXECUTION_EXCEPTIONS = (
    GamspyException,
    EngineException,
    LatexException,
    NeosException,
)


@pytest.mark.parametrize("exception", ALL_EXCEPTIONS)
def test_every_exception_is_catchable_through_the_base_class(exception):
    assert issubclass(exception, GamspyBaseException)


@pytest.mark.parametrize("exception", ALL_EXCEPTIONS)
def test_gamspy_exception_catches_exactly_the_execution_errors(exception):
    assert issubclass(exception, GamspyException) is (
        exception in GAMS_EXECUTION_EXCEPTIONS
    )


def test_fatal_error_is_not_caught_by_gamspy_exception():
    # FatalError is documented as never to be caught, so the recoverable-error
    # idiom must not swallow it.
    assert not issubclass(FatalError, GamspyException)

    with pytest.raises(FatalError):
        try:
            raise FatalError("unrecoverable")
        except GamspyException:  # pragma: no cover - must not catch
            pytest.fail("GamspyException caught a FatalError")


@pytest.mark.parametrize(
    "exception",
    (
        ValidationError,
        GdxException,
        GmdException,
        NeosClientException,
        EngineClientException,
        FatalError,
    ),
)
def test_plain_exceptions_take_a_single_message(exception):
    # The base class carries no __init__, so these must not inherit
    # GamspyException's (message, return_code) signature or its attributes.
    error = exception("boom")
    assert error.args == ("boom",)
    assert not hasattr(error, "message")
    assert not hasattr(error, "return_code")


def test_gamspy_exception_keeps_message_and_return_code():
    # _backend/local.py reads both off the caught exception.
    error = GamspyException("boom", return_code=3)
    assert error.message == "boom"
    assert error.return_code == 3


def test_engine_exception_keeps_its_status_code():
    error = EngineException("boom", return_code=3, status_code=7)
    assert error.message == "boom"
    assert error.return_code == 3
    assert error.status_code == 7


def test_catching_the_base_class_handles_unrelated_errors_normally():
    with pytest.raises(RuntimeError):
        try:
            raise RuntimeError("not ours")
        except GamspyBaseException:  # pragma: no cover - must not catch
            pytest.fail("GamspyBaseException caught a non-GAMSPy error")
