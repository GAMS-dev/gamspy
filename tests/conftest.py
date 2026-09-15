from __future__ import annotations

import functools
from typing import Any, NamedTuple

import pytest

import gamspy as gp
import gamspy.utils as utils
from gamspy import Container, _communication


@functools.cache
def installed_solvers() -> frozenset[str]:
    return frozenset(utils.getInstalledSolvers(utils._get_gamspy_base_directory()))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_solvers(*names): skip unless every named solver is installed.",
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
    before = set(_communication._comm_pairs)

    yield

    for pair_id in set(_communication._comm_pairs) - before:
        _communication.close_connection(pair_id)


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
