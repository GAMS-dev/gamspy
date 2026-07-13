from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gamspy.formulations.sddp.risk import CVaR


def _sci(v: float) -> str:
    s = f"{v:.6e}".upper()
    mantissa, exp = s.split("E")
    sign = exp[0]
    exp_val = int(exp[1:])
    return f"{mantissa}E{sign}{exp_val}"


@dataclass
class SDDPResult:
    """Outcome of an ``SDDP.train`` run.

    Attributes
    ----------
    convergence_table : list[dict[str, Any]]
        One row per iteration with the per-iteration bounds and timings.
    lower_bound : float
        Deterministic lower bound on the optimal cost at the last iteration.
    upper_bound : float
        Mean realised cost of the last forward simulation pass.
    upper_bound_95 : float
        Upper end of the 95% confidence interval around ``upper_bound``.
    sigma : float
        Sample standard deviation of the forward-pass path costs.
    total_time : float
        Wall-clock training time in seconds.
    stop_reason : str
        Why training stopped (e.g. ``"max_iter"``, ``"converged"``,
        ``"interrupted"``).
    iterations_run : int
        Number of SDDP iterations actually completed.
    risk : CVaR | None
        Risk measure used during training, or ``None`` for the risk-neutral
        expectation.
    policy_cost_mean : float
        Mean policy cost from the end-of-training Monte-Carlo run, populated
        only when ``train(gap_paths >= 1)``.
    policy_cost_stderr : float
        Standard error of ``policy_cost_mean``.
    policy_cost_paths : int
        Number of Monte-Carlo paths behind ``policy_cost_mean`` (0 when the
        gap run was skipped).
    """

    convergence_table: list[dict[str, Any]] = field(default_factory=list)
    lower_bound: float = float("nan")
    upper_bound: float = float("nan")
    upper_bound_95: float = float("nan")
    sigma: float = float("nan")
    total_time: float = 0.0
    stop_reason: str = "max_iter"
    iterations_run: int = 0
    risk: CVaR | None = None
    policy_cost_mean: float = float("nan")
    policy_cost_stderr: float = float("nan")
    policy_cost_paths: int = 0

    @property
    def optimality_gap_pct(self) -> float:
        """Estimated optimality gap from the end-of-training Monte-Carlo run.

        Point estimate ``100 * (UB - LB) / |UB|`` where ``UB`` is the mean
        realised policy cost (``policy_cost_mean``) and ``LB`` the lower bound.
        The trained policy's expected cost upper-bounds the true optimum, which
        ``lower_bound`` bounds from below, so this is the "how far from optimal"
        measure; the ``policy_cost_stderr`` confidence interval communicates the
        Monte-Carlo uncertainty around it separately.
        """
        if self.policy_cost_paths < 1 or math.isnan(self.policy_cost_mean):
            return float("nan")
        ub = self.policy_cost_mean
        denom = max(abs(ub), 1e-10)
        return max(0.0, 100.0 * (ub - self.lower_bound) / denom)

    def __str__(self) -> str:
        risk_adjusted = self.risk is not None
        bound_label = "Lower bound (risk-adj.)" if risk_adjusted else "Lower bound"
        lines = ["=" * 72]
        if risk_adjusted:
            lines.append(f"  Risk measure         : {self.risk!r}")
        lines.append(f"  {bound_label:<21s}: {_sci(self.lower_bound):>14s}")
        lines.append(f"  Iterations run       : {self.iterations_run:>14d}")
        lines.append(f"  Stop reason          : {self.stop_reason:>14s}")
        lines.append(f"  Total time           : {self.total_time:>14.2f} s")
        if self.policy_cost_paths >= 1 and not math.isnan(self.policy_cost_mean):
            ci = 1.96 * self.policy_cost_stderr
            lines.append("  " + "-" * 68)
            lines.append(
                f"  Policy cost          : {_sci(self.policy_cost_mean):>14s} "
                f"± {_sci(ci):>12s}   ({self.policy_cost_paths} MC paths, 95% CI)"
            )
            lines.append(f"  Optimality gap       : {self.optimality_gap_pct:>13.4f} %")
        lines.append("=" * 72)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"SDDPResult("
            f"lower_bound={self.lower_bound:,.3f}, "
            f"iterations_run={self.iterations_run}, "
            f"stop_reason={self.stop_reason!r})"
        )
