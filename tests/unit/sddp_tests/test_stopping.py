from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.requires_license

EXACT_LB = 112.3046875


def test_disabled_runs_all_iterations(clearlake_built):
    result = clearlake_built.sddp.train(n_iter=10)
    assert result.stop_reason == "max_iter"
    assert result.iterations_run == 10
    assert result.lower_bound == EXACT_LB


def test_converges_before_cap(clearlake_built):
    result = clearlake_built.sddp.train(n_iter=50, rel_tol=1e-3, patience=3)
    assert result.stop_reason == "converged"
    assert result.iterations_run < 50
    assert math.isclose(result.lower_bound, EXACT_LB, rel_tol=1e-3, abs_tol=0.05)


def test_patience_exceeding_cap_hits_max_iter(clearlake_built):
    result = clearlake_built.sddp.train(n_iter=4, rel_tol=1e-3, patience=10)
    assert result.stop_reason == "max_iter"
    assert result.iterations_run == 4


def test_convergence_table_matches_iterations_run(clearlake_built):
    result = clearlake_built.sddp.train(n_iter=50, rel_tol=1e-3, patience=3)
    assert len(result.convergence_table) == result.iterations_run
