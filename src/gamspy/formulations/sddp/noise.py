from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    import gamspy as gp


@dataclass
class NoiseConfig:
    """The stochastic noise model: scenario data, probabilities, and backing symbols.

    User-supplied fields are set by SDDP.set_noise().
    Build-time fields are populated by SDDP.build().
    """

    # user-supplied
    parameter: gp.Parameter
    scenario_data: np.ndarray  # shape (n_stages, n_scenarios)
    # 1-D probability per scenario, shape (n_scenarios,). None -> uniform 1/S.
    probabilities: np.ndarray | None = None

    # derived immediately
    @property
    def n_stages(self) -> int:
        return self.scenario_data.shape[0]

    @property
    def n_scenarios(self) -> int:
        return self.scenario_data.shape[1]

    # created by build()
    scenario_set: gp.Set | None = field(default=None, repr=False)

    def validate(self, n_stages: int) -> None:
        if self.scenario_data.ndim != 2:
            raise ValidationError(
                "scenario_data must be a 2-D array (n_stages x n_scenarios)"
            )
        if self.scenario_data.shape[0] != n_stages:
            raise ValidationError(
                f"scenario_data has {self.scenario_data.shape[0]} rows "
                f"but stage_set has {n_stages} elements"
            )
        if self.n_scenarios < 1:
            raise ValidationError(
                "scenario_data must have at least one scenario column"
            )

        if self.probabilities is not None:
            p = self.probabilities
            if p.ndim != 1:
                raise ValidationError(
                    f"probabilities must be a 1-D array, got shape {p.shape}"
                )
            if p.shape[0] != self.n_scenarios:
                raise ValidationError(
                    f"probabilities has length {p.shape[0]} but scenario_data "
                    f"has {self.n_scenarios} scenarios"
                )
            if np.any(p < 0):
                raise ValidationError(
                    f"probabilities must be non-negative; got min={float(p.min())}"
                )
            total = float(p.sum())
            if abs(total - 1.0) > 1e-9:
                raise ValidationError(
                    f"probabilities must sum to 1.0 within 1e-9; got sum={total}"
                )
