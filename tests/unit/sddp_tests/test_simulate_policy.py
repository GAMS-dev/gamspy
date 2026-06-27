from __future__ import annotations

import numpy as np
import pytest

from gamspy.exceptions import ValidationError


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
