from __future__ import annotations

import numpy as np
import pytest

from gamspy.formulations.sddp import LastCuts

pytestmark = pytest.mark.unit

EXACT_LB = 112.3046875


def test_documented_workflow_with_bundled_demo_license(demo_clearlake_built):
    c = demo_clearlake_built

    result = c.sddp.train(
        n_iter=20,
        rel_tol=1e-3,
        patience=3,
        gap_paths=500,
    )

    assert result.stop_reason == "converged"
    assert result.iterations_run == 8
    assert result.lower_bound == EXACT_LB
    assert result.policy_cost_paths == 500

    decision = c.sddp.policy(
        stage="mar",
        state=150.0,
        noise=350.0,
        report=[c.rel, c.lev, c.imp, c.flood],
    )
    assert decision.decisions == pytest.approx(
        {"R": 200.0, "L": 250.0, "Z": 0.0, "F": 50.0}
    )
    assert decision.approx_cost_to_go == pytest.approx(625.0)

    simulation = c.sddp.simulate(n_paths=20, seed=0)
    assert simulation.n_paths == 20
    assert simulation.stage_costs.shape == (20, 4)
    assert simulation.total_cost.shape == (20,)
    assert np.isfinite(simulation.total_cost.to_numpy(dtype=float)).all()


def test_cut_selection_bound_with_bundled_demo_license(demo_clearlake_built):
    result = demo_clearlake_built.sddp.train(
        n_iter=5,
        cut_selection=LastCuts(keep_iter=4),
    )

    assert result.stop_reason == "max_iter"
    assert result.iterations_run == 5
    assert result.lower_bound == EXACT_LB
