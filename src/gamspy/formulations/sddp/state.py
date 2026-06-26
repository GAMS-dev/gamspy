from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    import gamspy as gp


@dataclass
class StateVar:
    """One SDDP state variable: its bounds and the GAMSPy symbols backing it.

    User-supplied fields are set by SDDP.add_state().
    Build-time fields are populated by SDDP.build().
    """

    # user-supplied
    variable: gp.Variable
    lower_bound: float
    upper_bound: float
    # Value the state takes BEFORE stage 1, i.e. the boundary the week-1
    # solve sees on its predecessor. If None, falls back to lower_bound.
    initial_state: float | None = None

    # created by build()
    # The trial set is the sddp-owned composite set `sddp_i` (shared by all
    # states). The cut intercept is sddp-owned too (a single scalar per cut),
    # so it does NOT live here; only the per-state slope does.
    trial_set: gp.Set | None = field(default=None, repr=False)
    trial_param: gp.Parameter | None = field(default=None, repr=False)
    cut_slope: gp.Parameter | None = field(default=None, repr=False)  # cont_m

    # Per-state backward-pass GUSS scatter/extract + slope accumulator.
    is_res_fx: gp.Parameter | None = field(default=None, repr=False)
    is_res_m: gp.Parameter | None = field(default=None, repr=False)
    guss_cm: gp.Parameter | None = field(default=None, repr=False)

    # Per-state forward-pass GUSS scatter/extract + forward-state trackers.
    f_res_fixed: gp.Parameter | None = field(default=None, repr=False)
    f_res_level: gp.Parameter | None = field(default=None, repr=False)
    forward_state: gp.Parameter | None = field(default=None, repr=False)
    forward_res_state: gp.Parameter | None = field(default=None, repr=False)

    # Per-state stage-1 wait-and-see GUSS scatter/extract.
    is_w1_init: gp.Parameter | None = field(default=None, repr=False)
    is_w1_lstate: gp.Parameter | None = field(default=None, repr=False)

    # Per-state snapshot of the user's original variable bounds.
    orig_lo_param: gp.Parameter | None = field(default=None, repr=False)
    orig_up_param: gp.Parameter | None = field(default=None, repr=False)

    def validate(self) -> None:
        if self.lower_bound >= self.upper_bound:
            raise ValidationError(
                f"StateVar '{self.variable.name}': "
                f"lower_bound ({self.lower_bound}) must be < upper_bound ({self.upper_bound})"
            )

    @property
    def name(self) -> str:
        return self.variable.name
