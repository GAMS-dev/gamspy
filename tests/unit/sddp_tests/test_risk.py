from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import gamspy as gp
from gamspy import Container
from gamspy.exceptions import ValidationError
from gamspy.formulations import SDDP
from gamspy.formulations.sddp import CVaR

# Same ClearLake regression problem as conftest. The risk-neutral lower bound
# is exactly 112.3046875 (== 28750 / 256); CVaR with weight=0 (or tail=1,
# weight=1) must collapse back onto it.
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


def _clearlake_built(verbose: bool = False) -> tuple[Container, SDDP]:
    m = Container()
    t = gp.Set(m, "t", records=["jan", "feb", "mar", "apr"])
    sddp = SDDP(m, stage_set=t, n_trials=2, seed=42, verbose=verbose)
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

    sddp.add_state(variable=lev, initial_state=100.0)
    sddp.set_noise(
        parameter=precip, scenario_data=SCENARIOS, probabilities=PROBABILITIES
    )
    sddp.build(stage_cost=cost)
    return m, sddp


@pytest.mark.unit
def test_cvar_validation():
    r = CVaR(tail=0.05, weight=0.5)
    assert r.tail == 0.05
    assert r.weight == 0.5

    # boundaries are valid
    CVaR(tail=1.0, weight=0.0)
    CVaR(tail=1.0, weight=1.0)

    # tail must be in (0, 1]
    with pytest.raises(ValidationError):
        CVaR(tail=0.0, weight=0.5)
    with pytest.raises(ValidationError):
        CVaR(tail=1.5, weight=0.5)
    with pytest.raises(ValidationError):
        CVaR(tail=-0.1, weight=0.5)

    # weight must be in [0, 1]
    with pytest.raises(ValidationError):
        CVaR(tail=0.05, weight=-0.1)
    with pytest.raises(ValidationError):
        CVaR(tail=0.05, weight=1.1)

    # non-numeric inputs
    with pytest.raises(ValidationError):
        CVaR(tail="0.05", weight=0.5)
    with pytest.raises(ValidationError):
        CVaR(tail=0.05, weight=None)


@pytest.mark.unit
def test_cvar_is_frozen():
    r = CVaR(tail=0.05, weight=0.5)
    with pytest.raises(FrozenInstanceError):
        r.tail = 0.1


@pytest.mark.unit
def test_train_rejects_invalid_risk(clearlake_built):
    sddp = clearlake_built.sddp
    # risk must be a CVaR instance or None
    with pytest.raises(ValidationError):
        sddp.train(n_iter=1, risk="cvar")
    with pytest.raises(ValidationError):
        sddp.train(n_iter=1, risk=42)


@pytest.mark.requires_license
def test_cvar_weight_zero_matches_expectation():
    # weight=0 cancels the CVaR term

    m, sddp = _clearlake_built()
    risk = CVaR(tail=0.05, weight=0.0)
    res = sddp.train(n_iter=20, risk=risk)
    assert res.lower_bound == EXACT_LB
    assert res.risk is not None
    m.close()


@pytest.mark.requires_license
def test_cvar_tail_one_weight_one_matches_expectation():
    # CVaR over the entire distribution (tail=1) equals the expectation
    m, sddp = _clearlake_built()
    res = sddp.train(n_iter=20, risk=CVaR(tail=1.0, weight=1.0))
    assert np.isclose(res.lower_bound, EXACT_LB, rtol=1e-6)
    m.close()


@pytest.mark.requires_license
def test_cvar_raises_risk_adjusted_lower_bound():
    # Expectation baseline.
    m1, sddp1 = _clearlake_built()
    exp = sddp1.train(n_iter=20)

    # For a cost distribution with scenario variability CVaR strictly exceeds the expectation
    m2, sddp2 = _clearlake_built()
    cvar = sddp2.train(n_iter=20, risk=CVaR(tail=0.25, weight=1.0))

    assert cvar.lower_bound > exp.lower_bound
    assert cvar.risk is not None

    # A CVaR-trained policy still answers point queries.
    pol = sddp2.policy("mar", 180.0, 100.0)
    assert np.isfinite(pol.approx_cost_to_go)

    m1.close()
    m2.close()


@pytest.mark.requires_license
def test_pure_cvar_value_lb():
    m, sddp = _clearlake_built()
    res = sddp.train(n_iter=80, risk=CVaR(tail=0.25, weight=1.0))
    assert np.isclose(res.lower_bound, 916.6667, rtol=1e-3)
    m.close()


@pytest.mark.requires_license
def test_blend_cvar_value_lb():
    m, sddp = _clearlake_built()
    res = sddp.train(n_iter=80, risk=CVaR(tail=0.25, weight=0.5))
    assert np.isclose(res.lower_bound, 399.1699, rtol=1e-3)
    m.close()
