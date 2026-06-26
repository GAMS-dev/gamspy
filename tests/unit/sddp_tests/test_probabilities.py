from __future__ import annotations

import numpy as np
import pytest

from gamspy.exceptions import ValidationError
from gamspy.formulations.sddp.noise import NoiseConfig

pytestmark = pytest.mark.unit


def test_probabilities_validation():
    scenarios = np.ones((4, 3))

    # wrong length (2 != 3 scenarios)
    nc = NoiseConfig(
        parameter=None, scenario_data=scenarios, probabilities=np.array([0.5, 0.5])
    )
    with pytest.raises(ValidationError):
        nc.validate(4)

    # negative entry
    nc = NoiseConfig(
        parameter=None,
        scenario_data=scenarios,
        probabilities=np.array([-0.1, 0.6, 0.5]),
    )
    with pytest.raises(ValidationError):
        nc.validate(4)

    # does not sum to 1
    nc = NoiseConfig(
        parameter=None,
        scenario_data=scenarios,
        probabilities=np.array([0.5, 0.5, 0.5]),
    )
    with pytest.raises(ValidationError):
        nc.validate(4)

    # wrong number of dimensions
    nc = NoiseConfig(
        parameter=None, scenario_data=scenarios, probabilities=np.ones((2, 3))
    )
    with pytest.raises(ValidationError):
        nc.validate(4)


def test_uniform_default_is_valid():
    nc = NoiseConfig(parameter=None, scenario_data=np.ones((4, 3)), probabilities=None)
    nc.validate(4)


def test_probabilities_stored_in_container(clearlake_built):
    prob = clearlake_built.m["sddp_prob"].toDense()
    assert np.allclose(prob, [0.25, 0.5, 0.25])
