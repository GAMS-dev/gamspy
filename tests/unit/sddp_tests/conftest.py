from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import gamspy as gp
from gamspy import Container
from gamspy.formulations import SDDP

# ClearLake 4-stage reservoir - the canonical SDDP regression problem. With
# probabilities [0.25, 0.5, 0.25] the deterministic lower bound is exactly
# 112.3046875 (== 28750 / 256, so it round-trips through float bit-for-bit).
EXACT_LB = 112.3046875
SCENARIOS = np.array(
    [
        [50.0, 150.0, 350.0],
        [50.0, 150.0, 350.0],
        [-50.0, 100.0, 250.0],
        [-50.0, 100.0, 250.0],
    ]
)
PROBABILITIES = [0.25, 0.5, 0.25]


def _make_clearlake() -> SimpleNamespace:
    m = Container()
    t = gp.Set(m, "t", records=["jan", "feb", "mar", "apr"])
    sddp = SDDP(m, stage_set=t, n_trials=2, seed=42, verbose=False)
    stage = sddp.active_stage

    precip = gp.Parameter(m, "precip")
    lev = gp.Variable(m, "L", type="positive", domain=t)
    rel = gp.Variable(m, "R", type="positive", domain=t)
    flood = gp.Variable(m, "F", type="positive", domain=t)
    imp = gp.Variable(m, "Z", type="positive", domain=t)
    cost = gp.Variable(m, "COST")
    rel.up[t] = 200.0
    lev.up[t] = 250.0

    cont = gp.Equation(m, "Cont", domain=t)
    obj = gp.Equation(m, "Obj")
    cont[t].where[stage[t]] = (
        lev[t] - lev[t.lag(1, "circular")] + rel[t] + flood[t] - imp[t] == precip
    )
    obj[...] = cost == gp.Sum(stage[t], 10.0 * flood[t] + 5.0 * imp[t])

    return SimpleNamespace(
        m=m,
        sddp=sddp,
        t=t,
        precip=precip,
        lev=lev,
        rel=rel,
        flood=flood,
        imp=imp,
        cost=cost,
        scenarios=SCENARIOS,
        probabilities=PROBABILITIES,
    )


@pytest.fixture
def clearlake():
    c = _make_clearlake()
    yield c
    c.m.close()


@pytest.fixture
def clearlake_built(clearlake):
    c = clearlake
    c.sddp.add_state(variable=c.lev, initial_state=100.0)
    c.sddp.set_noise(
        parameter=c.precip,
        scenario_data=c.scenarios,
        probabilities=c.probabilities,
    )
    c.sddp.build(stage_cost=c.cost)
    return c


@pytest.fixture
def clearlake_trained(clearlake_built):
    c = clearlake_built
    c.result = c.sddp.train(n_iter=20, rel_tol=1e-3, patience=3)
    return c
