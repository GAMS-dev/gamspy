from __future__ import annotations

import pytest

import gamspy as gp
from gamspy import Container
from gamspy.exceptions import ValidationError
from gamspy.formulations import SDDP

pytestmark = pytest.mark.unit


def test_constructor_validation():
    m = Container()
    t = gp.Set(m, "t", records=["a", "b"])

    with pytest.raises(ValidationError):
        SDDP("not a container", stage_set=t)

    with pytest.raises(ValidationError):
        SDDP(m, stage_set="not a set")

    with pytest.raises(ValidationError):
        SDDP(m, stage_set=t, n_trials=0)

    m.close()


def test_lifecycle_ordering(clearlake):
    c = clearlake

    # build() before add_state()
    with pytest.raises(ValidationError):
        c.sddp.build(stage_cost=c.cost)

    c.sddp.add_state(variable=c.lev, initial_state=100.0, upper_bound=250.0)

    # build() before set_noise()
    with pytest.raises(ValidationError):
        c.sddp.build(stage_cost=c.cost)

    c.sddp.set_noise(parameter=c.precip, scenario_data=c.scenarios)
    c.sddp.build(stage_cost=c.cost)

    # build() twice
    with pytest.raises(ValidationError):
        c.sddp.build(stage_cost=c.cost)

    # registration after build()
    with pytest.raises(ValidationError):
        c.sddp.add_state(variable=c.lev)
    with pytest.raises(ValidationError):
        c.sddp.set_noise(parameter=c.precip, scenario_data=c.scenarios)


def test_set_noise_called_twice(clearlake):
    c = clearlake
    c.sddp.add_state(variable=c.lev, initial_state=100.0, upper_bound=250.0)
    c.sddp.set_noise(parameter=c.precip, scenario_data=c.scenarios)
    with pytest.raises(ValidationError):
        c.sddp.set_noise(parameter=c.precip, scenario_data=c.scenarios)


def test_build_requires_stage_cost_reference(clearlake):
    c = clearlake
    c.sddp.add_state(variable=c.lev, initial_state=100.0, upper_bound=250.0)
    c.sddp.set_noise(parameter=c.precip, scenario_data=c.scenarios)

    # A stage-cost variable that no equation references must be rejected.
    orphan = gp.Variable(c.m, "ORPHAN")
    with pytest.raises(ValidationError):
        c.sddp.build(stage_cost=orphan)


def test_train_argument_validation(clearlake_built):
    sddp = clearlake_built.sddp

    with pytest.raises(ValidationError):
        sddp.train(n_iter=0)
    with pytest.raises(ValidationError):
        sddp.train(rel_tol=0.0)
    with pytest.raises(ValidationError):
        sddp.train(rel_tol=-1e-3)
    with pytest.raises(ValidationError):
        sddp.train(patience=0)
    with pytest.raises(ValidationError):
        sddp.train(n_iter=True)
    with pytest.raises(ValidationError):
        sddp.train(n_iter=20.0)
    with pytest.raises(ValidationError):
        sddp.train(patience=True)
    with pytest.raises(ValidationError):
        sddp.train(gap_paths=1.5)


def test_simulate_argument_validation(clearlake_built):
    with pytest.raises(ValidationError):
        clearlake_built.sddp.simulate(n_paths=0)
