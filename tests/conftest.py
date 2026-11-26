from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, NamedTuple

import pytest

import gamspy as gp
import gamspy.utils as utils
from gamspy import Container
from gamspy._execution import socket_engine

if TYPE_CHECKING:
    from collections.abc import Generator


@functools.cache
def installed_solvers() -> frozenset[str]:
    return frozenset(utils.getInstalledSolvers(gp.get_option("GAMS_SYSDIR")))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_solvers(*names): skip unless every named solver is installed.",
    )


SELECTION_MARKERS = frozenset(
    {
        "unit",
        "integration",
        "cli",
        "doc",
        "engine",
        "neos",
        "model_library",
        "requires_license",
    }
)


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(
    collector: pytest.Collector,
) -> Generator[None, pytest.CollectReport, pytest.CollectReport]:
    report = yield
    if not isinstance(collector, (pytest.Module, pytest.Class)) or not report.passed:
        return report

    unmarked = [
        node.name
        for node in report.result
        if isinstance(node, pytest.Item)
        and not SELECTION_MARKERS & {marker.name for marker in node.iter_markers()}
    ]
    if not unmarked:
        return report

    return pytest.CollectReport(
        collector.nodeid,
        "failed",
        "These tests have no selection marker, so CI never runs them. Mark them "
        f"with one of {', '.join(sorted(SELECTION_MARKERS))}:\n  "
        + "\n  ".join(unmarked),
        [],
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    SOLVER_ALIASES = {"CONOPT": "CONOPT4"}
    required = {
        SOLVER_ALIASES.get(name.upper(), name.upper())
        for marker in item.iter_markers("requires_solvers")
        for name in marker.args
    }

    missing = sorted(required - installed_solvers())
    if missing:
        pytest.skip(
            "requires solver(s) that are not installed: "
            f"{', '.join(missing)}. Install with `gamspy install solver "
            f"{' '.join(solver.lower() for solver in missing)}`."
        )


class TransportData(NamedTuple):
    container: Container
    canning_plants: list[str]
    markets: list[str]
    distances: list[list[Any]]
    capacities: list[list[Any]]
    demands: list[list[Any]]


@pytest.fixture(autouse=True)
def close_containers():
    before = set(socket_engine._running_engines)

    yield

    for engine in set(socket_engine._running_engines) - before:
        engine.stop()


@pytest.fixture
def container():
    m = Container()
    yield m
    m.close()


@pytest.fixture
def transport(container):
    yield TransportData(
        container=container,
        canning_plants=["seattle", "san-diego"],
        markets=["new-york", "chicago", "topeka"],
        distances=[
            ["seattle", "new-york", 2.5],
            ["seattle", "chicago", 1.7],
            ["seattle", "topeka", 1.8],
            ["san-diego", "new-york", 2.5],
            ["san-diego", "chicago", 1.8],
            ["san-diego", "topeka", 1.4],
        ],
        capacities=[["seattle", 350], ["san-diego", 600]],
        demands=[["new-york", 325], ["chicago", 300], ["topeka", 275]],
    )


@pytest.fixture
def set_options():
    originals = {}

    def _set_options(options: dict) -> None:
        for key in options:
            if key not in originals:
                originals[key] = gp.get_option(key)

        gp.set_options(options)

    yield _set_options

    if originals:
        gp.set_options(originals)
