from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

import gamspy as gp
from gamspy import Container
from gamspy.exceptions import ValidationError
from gamspy.formulations import SDDP
from gamspy.formulations.sddp import CVaR

# Two independent identical ClearLake reservoirs sharing one noise. The problem
# decomposes, so the risk-neutral lower bound is exactly 2 x 112.3046875.
EXACT_2X = 224.609375

SCENARIOS = np.array(
    [
        [50.0, 150.0, 350.0],
        [50.0, 150.0, 350.0],
        [-50.0, 100.0, 250.0],
        [-50.0, 100.0, 250.0],
    ]
)
PROBABILITIES = [0.25, 0.5, 0.25]


def _two_independent_clearlakes(verbose: bool = False) -> SimpleNamespace:
    m = Container()
    t = gp.Set(m, "t", records=["jan", "feb", "mar", "apr"])
    sddp = SDDP(m, stage_set=t, n_trials=2, seed=42, verbose=verbose)
    stage = sddp.active_stage
    precip = gp.Parameter(m, "precip")

    v: dict[str, gp.Variable] = {}
    for k in (1, 2):
        v[f"L{k}"] = gp.Variable(m, f"L{k}", type="positive", domain=t)
        v[f"R{k}"] = gp.Variable(m, f"R{k}", type="positive", domain=t)
        v[f"F{k}"] = gp.Variable(m, f"F{k}", type="positive", domain=t)
        v[f"Z{k}"] = gp.Variable(m, f"Z{k}", type="positive", domain=t)
        v[f"R{k}"].up[t] = 200.0
        v[f"L{k}"].up[t] = 250.0
    cost = gp.Variable(m, "COST")

    for k in (1, 2):
        cont = gp.Equation(m, f"Cont{k}", domain=t)
        lk, rk, fk, zk = v[f"L{k}"], v[f"R{k}"], v[f"F{k}"], v[f"Z{k}"]
        cont[t].where[stage[t]] = (
            lk[t] - lk[t.lag(1, "circular")] + rk[t] + fk[t] - zk[t] == precip
        )
    obj = gp.Equation(m, "Obj")
    obj[...] = cost == gp.Sum(
        stage[t],
        10.0 * v["F1"][t] + 5.0 * v["Z1"][t] + 10.0 * v["F2"][t] + 5.0 * v["Z2"][t],
    )

    sddp.add_state(variable=v["L1"], initial_state=100.0, upper_bound=250.0)
    sddp.add_state(variable=v["L2"], initial_state=100.0, upper_bound=250.0)
    sddp.set_noise(
        parameter=precip, scenario_data=SCENARIOS, probabilities=PROBABILITIES
    )
    sddp.build(stage_cost=cost)
    return SimpleNamespace(m=m, sddp=sddp, cost=cost, **v)


@pytest.mark.requires_license
def test_two_independent_reservoirs_decompose():
    # Two independent ClearLakes -> exactly 2x the bound.
    c = _two_independent_clearlakes()
    result = c.sddp.train(n_iter=20)
    assert math.isclose(result.lower_bound, EXACT_2X, rel_tol=1e-9)
    c.m.close()


@pytest.mark.requires_license
def test_multistate_policy_and_simulate():
    c = _two_independent_clearlakes()
    c.sddp.train(n_iter=20)

    # policy() with a dict keyed by state-variable name.
    pol = c.sddp.policy(
        "mar", {"L1": 180.0, "L2": 120.0}, 100.0, report=[c.R1, c.R2, c.L1, c.L2]
    )
    assert isinstance(pol.incoming_state, dict)
    assert set(pol.incoming_state) == {"L1", "L2"}
    assert {"R1", "R2", "L1", "L2"} <= set(pol.decisions)
    assert math.isfinite(pol.approx_cost_to_go)

    # simulate(): default report captures every state; both propagate.
    sim = c.sddp.simulate(n_paths=20)
    assert sim.total_cost.shape == (20,)
    assert sim.stage_costs.shape == (20, 4)
    assert "L1" in sim.variables and "L2" in sim.variables
    c.m.close()


@pytest.mark.requires_license
def test_multistate_cvar_composition():
    # CVaR over the whole distribution (tail=1) equals the expectation.
    c = _two_independent_clearlakes()
    res = c.sddp.train(n_iter=40, risk=CVaR(tail=1.0, weight=1.0))
    assert np.isclose(res.lower_bound, EXACT_2X, rtol=1e-6)
    assert res.risk is not None
    c.m.close()


@pytest.mark.unit
def test_policy_scalar_rejected_for_multistate():
    # A bare scalar is ambiguous with several states -> must raise.
    c = _two_independent_clearlakes()
    with pytest.raises(ValidationError):
        with pytest.warns(UserWarning):
            c.sddp.policy("mar", 180.0, 100.0)
    c.m.close()


@pytest.mark.unit
def test_policy_bad_state_keys_rejected():
    c = _two_independent_clearlakes()
    # Missing a key.
    with pytest.raises(ValidationError):
        with pytest.warns(UserWarning):
            c.sddp.policy("mar", {"L1": 180.0}, 100.0)
    # Unknown key.
    with pytest.raises(ValidationError):
        with pytest.warns(UserWarning):
            c.sddp.policy("mar", {"L1": 180.0, "L3": 1.0}, 100.0)
    c.m.close()


@pytest.mark.requires_license
def test_policy_multiple_calls_respect_bounds_multistate():
    c = _two_independent_clearlakes()
    c.sddp.train(n_iter=20)
    report = [c.R1, c.F1, c.Z1, c.R2, c.F2, c.Z2]
    p1 = c.sddp.policy("jan", {"L1": 100.0, "L2": 100.0}, 150.0, report=report)
    p2 = c.sddp.policy("feb", {"L1": 230.0, "L2": 230.0}, 350.0, report=report)
    for k in ("R1", "R2"):
        assert p1.decisions[k] <= 200.0 + 1e-6
        assert p2.decisions[k] <= 200.0 + 1e-6
    assert np.isclose(p2.decisions["R1"], 200.0, atol=1e-3)
    assert np.isclose(p2.decisions["R2"], 200.0, atol=1e-3)
    assert np.isclose(p2.decisions["F1"], 130.0, atol=1e-3)
    assert np.isclose(p2.decisions["F2"], 130.0, atol=1e-3)
    c.m.close()


# ── Coupled cascade vs its deterministic-equivalent optimum
_CASC_STAGES = ["w1", "w2", "w3"]
_CASC_SUPPORT = [("lo", 20.0, 0.5), ("hi", 80.0, 0.5)]
_CASC_CAP = 100.0
_CASC_RMAX = 40.0
_CASC_U0 = 50.0
_CASC_D0 = 50.0
_CASC_INFLOW = np.array([[s[1] for s in _CASC_SUPPORT] for _ in _CASC_STAGES])
_CASC_PROBS = [s[2] for s in _CASC_SUPPORT]
_CASC_DE_OPTIMUM = 1150.0


def _train_cascade_sddp(n_iter: int = 60) -> float:
    m = Container()
    t = gp.Set(m, "t", records=_CASC_STAGES)
    sddp = SDDP(m, stage_set=t, n_trials=3, seed=42, verbose=False)
    stage = sddp.active_stage
    precip = gp.Parameter(m, "precip")

    u = gp.Variable(m, "U", type="positive", domain=t)
    d = gp.Variable(m, "D", type="positive", domain=t)
    ru = gp.Variable(m, "Ru", type="positive", domain=t)
    fu = gp.Variable(m, "Fu", type="positive", domain=t)
    zu = gp.Variable(m, "Zu", type="positive", domain=t)
    rd = gp.Variable(m, "Rd", type="positive", domain=t)
    fd = gp.Variable(m, "Fd", type="positive", domain=t)
    zd = gp.Variable(m, "Zd", type="positive", domain=t)
    u.up[t] = _CASC_CAP
    d.up[t] = _CASC_CAP
    ru.up[t] = _CASC_RMAX
    rd.up[t] = _CASC_RMAX
    cost = gp.Variable(m, "COST")

    bu = gp.Equation(m, "bu", domain=t)
    bd = gp.Equation(m, "bd", domain=t)
    obj = gp.Equation(m, "obj")
    bu[t].where[stage[t]] = (
        u[t] - u[t.lag(1, "circular")] + ru[t] + fu[t] - zu[t] == precip
    )
    bd[t].where[stage[t]] = (
        d[t] - d[t.lag(1, "circular")] + rd[t] + fd[t] - zd[t] - ru[t] - fu[t] == precip
    )
    obj[...] = cost == gp.Sum(stage[t], 10.0 * (fu[t] + fd[t]) + 5.0 * (zu[t] + zd[t]))

    sddp.add_state(variable=u, initial_state=_CASC_U0, upper_bound=_CASC_CAP)
    sddp.add_state(variable=d, initial_state=_CASC_D0, upper_bound=_CASC_CAP)
    sddp.set_noise(
        parameter=precip, scenario_data=_CASC_INFLOW, probabilities=_CASC_PROBS
    )
    sddp.build(stage_cost=cost)
    res = sddp.train(n_iter=n_iter)
    m.close()
    return res.lower_bound


@pytest.mark.requires_license
def test_coupled_cascade_matches_deterministic_equivalent():
    # SDDP bound must converge to the deterministic-equivalent optimum.
    sddp_lb = _train_cascade_sddp(n_iter=60)
    assert math.isclose(sddp_lb, _CASC_DE_OPTIMUM, rel_tol=1e-5, abs_tol=1e-3)
