from __future__ import annotations

import numpy as np
import pytest

from gamspy import Equation, Parameter, Set, Sum, Variable
from gamspy.exceptions import ValidationError
from gamspy.formulations.sddp import SDDP


@pytest.mark.requires_license
def test_simulate_output_shapes(clearlake_trained):
    c = clearlake_trained
    sim = c.sddp.simulate(n_paths=20, report=[c.lev])
    assert sim.n_paths == 20
    assert sim.stage_costs.shape == (20, 4)  # paths x stages
    assert sim.total_cost.shape == (20,)
    assert "L" in sim.variables


@pytest.mark.requires_license
def test_simulate_is_reproducible(clearlake_trained):
    c = clearlake_trained
    a = c.sddp.simulate(n_paths=20, report=[c.lev], seed=7)
    b = c.sddp.simulate(n_paths=20, report=[c.lev], seed=7)
    assert a.total_cost.equals(b.total_cost)


@pytest.mark.unit
@pytest.mark.requires_license
def test_state_propagation(container):
    scen_data = np.array([[1, 2, 3], [1, 2, 3], [2, 2, 2]])

    t = Set(container, name="t", records=["a", "b", "c"])
    model = SDDP(container, stage_set=t, n_trials=6, seed=42, verbose=False)
    active_stage = model.active_stage
    noise = Parameter(container, name="noise")
    state = Variable(container, name="state", domain=t)
    state.lo[t] = -2
    state.up[t] = 4
    cost = Variable(container, name="cost")

    # Nonzero noise produces outgoing states of -1, 0, or 1.
    balance = Equation(container, name="balance", domain=t)
    balance[t].where[active_stage[t]] = state[t] == noise - 2
    objective = Equation(container, name="objective")
    objective[...] = cost == 10 + Sum(
        active_stage[t], state[t.lag(1, "circular")] - state[t]
    )

    model.add_state(variable=state, initial_state=1)

    model.set_noise(parameter=noise, scenario_data=scen_data)
    model.build(stage_cost=cost)
    model.train(n_iter=1)

    # Incoming/outgoing terms cancel: total = 3 * 10 + initial 1 - final 0.
    assert np.allclose(model._forward_path_cost.toDense(), 31)

    sim = model.simulate(n_paths=12, seed=0)
    levels = sim.variables["state"]
    costs = sim.stage_costs

    assert np.allclose(levels, sim.noise - 2)
    assert np.allclose(costs["a"] + levels["a"], 11)
    assert np.allclose(costs["b"] + levels["b"], 10 + levels["a"])
    assert np.allclose(costs["c"] + levels["c"], 10 + levels["b"])


@pytest.mark.requires_license
def test_policy_report_via_container(clearlake_trained):
    c = clearlake_trained
    release = c.sddp.container["R"]
    pol = c.sddp.policy(stage="mar", state=180, noise=100, report=[release])
    assert "R" in pol.decisions
    assert isinstance(pol.decisions["R"], float)


@pytest.mark.unit
def test_report_rejects_string(clearlake_built):
    with pytest.raises(ValidationError):
        clearlake_built.sddp.simulate(n_paths=5, report=["R"])


@pytest.mark.requires_license
def test_policy_multiple_calls_respect_bounds(clearlake_trained):
    c = clearlake_trained
    report = [c.rel, c.flood, c.imp]
    p1 = c.sddp.policy("jan", 100.0, 150.0, report=report)
    p2 = c.sddp.policy("feb", 230.0, 350.0, report=report)
    assert p1.decisions["R"] <= 200.0 + 1e-6
    assert p2.decisions["R"] <= 200.0 + 1e-6
    assert np.isclose(p2.decisions["R"], 200.0, atol=1e-3)
    assert np.isclose(p2.decisions["F"], 130.0, atol=1e-3)
