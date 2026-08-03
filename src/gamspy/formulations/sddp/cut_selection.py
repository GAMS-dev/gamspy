from __future__ import annotations

from dataclasses import dataclass

from gamspy.exceptions import ValidationError

# LastCuts is currently the only cut-selection strategy. If others are added
# later (e.g. a dominance-based one), factor out a shared base or Protocol at
# that point. Note that a dominance strategy's size parameter counts cuts per
# trial point, not iterations, so it would not share this class's field.


@dataclass(frozen=True)
class LastCuts:
    """Keep only the most recently generated cuts, deactivating older ones.

    Every SDDP iteration adds ``n_trials`` cuts per stage transition. If left
    unbounded, the cut pool grows linearly, and each stage subproblem grows with
    it. ``LastCuts`` caps the pool at the cuts from the most recent
    ``keep_iter`` completed iterations, so the subproblems stop growing.

    Older cuts are only *deactivated*, never deleted, and the full pool is
    restored at the end of training: cut selection speeds up training without
    costing the trained policy anything.

    Parameters
    ----------
    keep_iter : int
        How many completed iterations' cuts to keep.

    Notes
    -----
    Keeping only recent cuts is one of the cut-selection strategies studied by
    de Matos, Philpott and Finardi, *Improving the performance of Stochastic
    Dual Dynamic Programming*, Journal of Computational and Applied Mathematics
    290:196-208 (2015).

    Because cuts are dropped, the lower bound is no longer monotone during
    training. Training accounts for this: convergence is confirmed against the
    full cut pool before it is declared.

    """

    keep_iter: int

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if isinstance(self.keep_iter, bool) or not isinstance(self.keep_iter, int):
            raise ValidationError("LastCuts keep_iter must be an int")
        if self.keep_iter < 1:
            raise ValidationError("LastCuts keep_iter must be >= 1.")
