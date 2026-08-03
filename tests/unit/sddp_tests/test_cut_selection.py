from __future__ import annotations

import math

import pytest

from gamspy.exceptions import ValidationError
from gamspy.formulations.sddp import LastCuts

# ClearLake (the conftest fixtures) has n_trials=2, so one retained iteration is
# 2 cuts per stage transition. Its exact risk-neutral lower bound is 112.3046875
# (== 28750 / 256), reached by iteration 5 and held thereafter.
EXACT_LB = 112.3046875
N_TRIALS = 2


# validation - raises before any solve, so no license needed


@pytest.mark.unit
def test_last_cuts_validation():
    assert LastCuts(1).keep_iter == 1
    assert LastCuts(50).keep_iter == 50
    assert LastCuts(keep_iter=10).keep_iter == 10

    # positive int
    with pytest.raises(ValidationError):
        LastCuts(0)
    with pytest.raises(ValidationError):
        LastCuts(-3)

    # strict int
    with pytest.raises(ValidationError):
        LastCuts(True)
    with pytest.raises(ValidationError):
        LastCuts(5.0)
    with pytest.raises(ValidationError):
        LastCuts(2.5)
    with pytest.raises(ValidationError):
        LastCuts("5")


@pytest.mark.unit
def test_train_rejects_invalid_cut_selection(clearlake_built):
    sddp = clearlake_built.sddp
    with pytest.raises(ValidationError):
        sddp.train(n_iter=1, cut_selection="last")
    with pytest.raises(ValidationError):
        sddp.train(n_iter=1, cut_selection=50)


@pytest.mark.requires_license
def test_cut_selection_none_is_unchanged(clearlake_built):
    res = clearlake_built.sddp.train(n_iter=12)

    assert res.lower_bound == EXACT_LB
    assert res.cut_selection is None

    assert [r["active_cuts"] for r in res.convergence_table] == [
        N_TRIALS * k for k in range(1, 13)
    ]


@pytest.mark.requires_license
def test_last_cuts_window_and_restoration(clearlake_built):
    sddp = clearlake_built.sddp
    res = sddp.train(n_iter=12, cut_selection=LastCuts(keep_iter=5))

    # The active pool grows to keep_iter iterations (10 cuts at n_trials=2) and
    # then pins there - proof that pruning happens and keeps exactly the size
    # promised.
    assert [r["active_cuts"] for r in res.convergence_table] == [
        2,
        4,
        6,
        8,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
    ]

    assert res.lower_bound == EXACT_LB
    assert res.iterations_run == 12
    assert res.cut_selection == LastCuts(5)


@pytest.mark.requires_license
def test_keep_iter_at_least_n_iter_warns(clearlake_built):
    # keep_iter >= n_iter can never retire a cut, so it warns but still runs.
    with pytest.warns(UserWarning):
        res = clearlake_built.sddp.train(n_iter=5, cut_selection=LastCuts(keep_iter=10))
    # Inert: it drops to the no-selection path, so the pool grows unbounded and
    # the result reports honestly that no selection was in effect.
    assert [r["active_cuts"] for r in res.convergence_table] == [2, 4, 6, 8, 10]
    assert res.lower_bound == EXACT_LB
    assert res.cut_selection is None
    assert math.isnan(res.selection_bound_gap_pct)
