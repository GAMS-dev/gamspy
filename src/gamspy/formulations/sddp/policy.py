from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PolicyResult:
    """The trained policy's decision at a single point.

    Returned by `SDDP.policy`. Answers the operational question
    *"I'm standing in `stage`, my state came in at `incoming_state`, the
    noise for this stage realised as `noise`. What should I do, and what
    does it cost me from here on?"*

    Attributes
    ----------
    stage : str
        The stage label that was queried (e.g. ``"mar"`` or ``"w17"``).
    incoming_state : float | dict[str, float]
        The state value entering the stage.
    noise : float
        The realised noise value injected for the stage.
    approx_cost_to_go : float
        ``acost.l`` from the point solve: the immediate stage cost **plus**
        the cut-approximated expected future cost.
    decisions : dict[str, Any]
        ``{variable_name: level}`` for each reported variable, evaluated at
        the stage's last time step. The value shape depends on the
        variable's domain:

        - Variables with domain ``[time_set]`` give a ``float``.
        - Variables with domain ``[time_set, other_dim]`` give a
          ``dict[str, float]`` keyed by the non-time dimension's label
          (e.g. ``{"Hydro": 100.0, "HardCoal": 200.0}``).
        - Variables with domain ``[time_set, dim1, dim2, ...]`` (3+-D)
          give a ``dict[tuple[str, ...], float]`` keyed by a tuple of
          the non-time dim labels in declaration order.
    """

    stage: str
    incoming_state: float | dict[str, float]
    noise: float
    approx_cost_to_go: float
    decisions: dict[str, Any]

    def __repr__(self) -> str:
        parts = []
        for k, v in self.decisions.items():
            if isinstance(v, dict):
                inner = ", ".join(f"{ik}={iv:,.3f}" for ik, iv in v.items())
                parts.append(f"{k}={{{inner}}}")
            else:
                parts.append(f"{k}={v:,.3f}")
        d = ", ".join(parts)
        if isinstance(self.incoming_state, dict):
            inner = ", ".join(f"{k}={v:,.3f}" for k, v in self.incoming_state.items())
            incoming_str = f"{{{inner}}}"
        else:
            incoming_str = f"{self.incoming_state:,.3f}"
        return (
            f"PolicyResult(stage={self.stage!r}, "
            f"incoming_state={incoming_str}, "
            f"noise={self.noise:,.3f}, "
            f"approx_cost_to_go={self.approx_cost_to_go:,.3f}, "
            f"decisions={{{d}}})"
        )
