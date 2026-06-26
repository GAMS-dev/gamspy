from __future__ import annotations

from dataclasses import dataclass

from gamspy.exceptions import ValidationError

# CVaR is currently the only risk measure. If others are added later (e.g. a
# worst-case measure), factor out a shared base or Protocol at that point.


@dataclass(frozen=True)
class CVaR:
    """Conditional Value-at-Risk: a convex blend of expectation and CVaR.

    At every stage the cost-to-go is aggregated as

    ``(1 - weight) * E[cost + future]  +  weight * CVaR_tail[cost + future]``

    that is, a blend of the expectation and the mean cost over the worst
    ``tail`` fraction of outcomes. With ``weight = 0`` this is the
    risk-neutral expectation; with ``weight = 1`` it is pure CVaR over the
    tail.

    Parameters
    ----------
    tail : float
        Tail probability in ``(0, 1]``; the fraction of worst-case
        outcomes that CVaR averages over.
    weight : float
        Risk weight in ``[0, 1]``; how much mass to place on the CVaR term
        relative to the expectation.
    """

    tail: float
    weight: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name, val in (("tail", self.tail), ("weight", self.weight)):
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValidationError(
                    f"CVaR {name} must be a number, got {type(val).__name__}"
                )
        if not 0.0 < self.tail <= 1.0:
            raise ValidationError(f"CVaR tail must be in (0, 1], got {self.tail}")
        if not 0.0 <= self.weight <= 1.0:
            raise ValidationError(f"CVaR weight must be in [0, 1], got {self.weight}")
