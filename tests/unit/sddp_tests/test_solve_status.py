from __future__ import annotations

import pytest

from gamspy.exceptions import ValidationError

pytestmark = [pytest.mark.unit, pytest.mark.requires_license]


@pytest.fixture
def clearlake_with_limited_imports(clearlake):
    c = clearlake
    # Normal ClearLake inflows remain feasible; an extreme drought will not.
    c.imp.up[c.t] = 50.0
    c.sddp.add_state(variable=c.lev, initial_state=100.0)
    c.sddp.set_noise(
        parameter=c.precip,
        scenario_data=c.scenarios,
        probabilities=c.probabilities,
    )
    c.sddp.build(stage_cost=c.cost)
    return c


def test_backward_rejects_infeasible_scenario(clearlake_with_limited_imports):
    c = clearlake_with_limited_imports
    c.sddp._stage_scenario_noise["feb", "s1"] = -1000.0

    with pytest.raises(ValidationError) as error:
        c.sddp.train(n_iter=1)

    message = str(error.value)
    assert "backward stage 'feb', trial 'trial1', scenario 's1'" in message
    assert "ModelStat=Infeasible" in message
    assert c.sddp.active_stage.toList() == ["feb"]
    assert c.sddp._active_cut_iteration_set.toList() == []
    assert ("iteration1", "lo") not in c.sddp._convergence_metrics.toDict()


def test_initial_stage_rejects_infeasible_scenario(clearlake_with_limited_imports):
    c = clearlake_with_limited_imports
    c.sddp._stage_scenario_noise["jan", "s1"] = -1000.0

    with pytest.raises(ValidationError) as error:
        c.sddp.train(n_iter=1)

    message = str(error.value)
    assert "initial stage 'jan', iteration 'iteration1', scenario 's1'" in message
    assert "ModelStat=Infeasible" in message
    assert ("iteration1", "lo") not in c.sddp._convergence_metrics.toDict()
    assert c.sddp._active_cut_iteration_set.toList() == []


def test_forward_rejects_infeasible_scenario(
    clearlake_with_limited_imports, monkeypatch
):
    c = clearlake_with_limited_imports
    original_solve = c.sddp._gp_model.solve

    # Only forward solves get the drought; earlier passes keep normal inflows.
    def solve_with_forward_drought(options, scenario, **kwargs):
        if scenario is c.sddp._forward_scenario_dict:
            c.sddp._forward_noise[...] = -1000.0
        return original_solve(options=options, scenario=scenario, **kwargs)

    monkeypatch.setattr(c.sddp._gp_model, "solve", solve_with_forward_drought)
    with pytest.raises(ValidationError) as error:
        c.sddp.train(n_iter=1)

    message = str(error.value)
    assert "forward stage 'feb'" in message
    assert "ModelStat=Infeasible" in message
    assert c.sddp.active_stage.toList() == ["feb"]
    assert ("iteration1", "up") not in c.sddp._convergence_metrics.toDict()
    assert c.sddp._active_cut_iteration_set.toList() == []


def test_simulation_rejects_infeasible_path(clearlake_with_limited_imports):
    c = clearlake_with_limited_imports
    c.sddp.train(n_iter=2)
    # Every sampled February scenario is infeasible, regardless of the seed.
    c.sddp._stage_scenario_noise["feb", ...] = -1000.0

    with pytest.raises(ValidationError) as error:
        c.sddp.simulate(n_paths=2, seed=7)

    message = str(error.value)
    assert "simulation stage 'feb', path 'p1'" in message
    assert "ModelStat=Infeasible" in message
    assert c.sddp.active_stage.toList() == ["feb"]


def test_policy_failure_restores_bounds(clearlake_with_limited_imports):
    c = clearlake_with_limited_imports
    c.sddp.train(n_iter=2)
    baseline = c.sddp.policy("feb", state=100.0, noise=50.0)

    with pytest.raises(ValidationError) as error:
        c.sddp.policy("feb", state=100.0, noise=-1000.0)

    message = str(error.value)
    assert "policy stage 'feb'" in message
    assert "ModelStat=Infeasible" in message

    decision = c.sddp.policy("feb", state=100.0, noise=50.0)
    assert decision.approx_cost_to_go == pytest.approx(baseline.approx_cost_to_go)
