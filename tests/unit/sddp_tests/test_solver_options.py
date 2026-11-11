from __future__ import annotations

import io

import numpy as np
import pytest

import gamspy as gp
from gamspy.formulations.sddp import SDDP

pytestmark = [pytest.mark.unit, pytest.mark.requires_license]


@pytest.mark.requires_solvers("cplex", "conopt4")
def test_solver_options(tmp_path):
    log = io.StringIO()
    m = gp.Container(output=log)
    t = gp.Set(m, "t", records=["jan", "feb", "mar", "apr"])
    sddp = SDDP(
        m,
        stage_set=t,
        n_trials=2,
        verbose=False,
        solver="conopt",
        solver_options={"rtmaxv": "1.e12"},
    )
    stage = sddp.active_stage
    precip = gp.Parameter(m, "precip")
    level = gp.Variable(m, "L", type="positive", domain=t)
    release = gp.Variable(m, "R", type="positive", domain=t)
    flood = gp.Variable(m, "F", type="positive", domain=t)
    shortfall = gp.Variable(m, "Z", type="positive", domain=t)
    cost = gp.Variable(m, "COST")
    level.up[t] = 250.0
    release.up[t] = 200.0
    balance = gp.Equation(m, "balance", domain=t)
    objective = gp.Equation(m, "objective")
    balance[t].where[stage[t]] = (
        level[t] - level[t.lag(1, "circular")] + release[t] + flood[t] - shortfall[t]
        == precip
    )
    objective[...] = cost == gp.Sum(stage[t], 10.0 * flood[t] + 5.0 * shortfall[t])
    sddp.add_state(level, initial_state=100.0)
    sddp.set_noise(
        precip,
        scenario_data=np.array(
            [[-50, 100, 250], [-50, 100, 250], [50, 150, 350], [50, 150, 350]]
        ),
        probabilities=[0.25, 0.5, 0.25],
    )
    sddp.build(stage_cost=cost)

    sddp.train(n_iter=1)
    assert "--- Executing CONOPT4 (" in log.getvalue()
    assert "--- Executing CPLEX (" not in log.getvalue()

    path = str(tmp_path / "clearlake.sddp")
    sddp.save(path)

    loaded_cplx = SDDP.load(path, solver="cplex", solver_options={"threads": 1})
    log = io.StringIO()
    loaded_cplx.container.output = log
    loaded_cplx.policy("mar", state=180, noise=100)
    assert "--- Executing CPLEX (" in log.getvalue()

    loaded_conopt = SDDP.load(path, solver="conopt", solver_options={"threads": 1})
    log = io.StringIO()
    loaded_conopt.container.output = log
    loaded_conopt.policy("mar", state=180, noise=100)
    assert "--- Executing CONOPT4 (" in log.getvalue()
