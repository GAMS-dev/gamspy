from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class SimulationResult:
    """Per-path Monte Carlo evaluation of a trained SDDP policy.

    All DataFrame attributes are indexed by ``path`` (rows) and ``stage``
    (columns) so the standard ClearLake-style pivot is the natural shape.

    Attributes
    ----------
    n_paths : int
        Number of independent simulation paths.
    total_cost : pd.Series
        Total realised cost per path, indexed by ``path``.
    stage_costs : pd.DataFrame
        Per-stage cost, indexed by ``path`` with one column per ``stage``.
    noise : pd.DataFrame
        Realised noise per (path, stage).
    variables : dict[str, pd.DataFrame]
        ``{variable_name: per (path, stage) levels}`` for each reported
        variable.
    elapsed : float
        Wall-clock simulation time in seconds. By default 0.0.
    """

    n_paths: int
    total_cost: pd.Series
    stage_costs: pd.DataFrame
    noise: pd.DataFrame
    variables: dict[str, pd.DataFrame]
    elapsed: float = 0.0

    @property
    def summary(self) -> pd.Series:
        """Mean / std / percentiles of ``total_cost`` across paths."""
        tc = self.total_cost
        return pd.Series(
            {
                "n_paths": float(self.n_paths),
                "mean": float(tc.mean()),
                "std": float(tc.std(ddof=1) if len(tc) > 1 else 0.0),
                "p5": float(tc.quantile(0.05)),
                "p50": float(tc.quantile(0.50)),
                "p95": float(tc.quantile(0.95)),
                "max": float(tc.max()),
            },
            name="total_cost",
        )

    def __repr__(self) -> str:
        tc = self.total_cost
        std = float(tc.std(ddof=1)) if len(tc) > 1 else 0.0
        return (
            f"SimulationResult("
            f"n_paths={self.n_paths}, "
            f"mean={tc.mean():,.3f}, "
            f"std={std:,.3f}, "
            f"p95={tc.quantile(0.95):,.3f}, "
            f"max={tc.max():,.3f})"
        )
