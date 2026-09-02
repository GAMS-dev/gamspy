from __future__ import annotations

from typing import Any, NamedTuple

import pytest

import gamspy as gp
from gamspy import Container, _communication


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
