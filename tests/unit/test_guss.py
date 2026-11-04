from __future__ import annotations

import pytest

import gamspy as gp
from gamspy.exceptions import ValidationError

pytestmark = pytest.mark.unit

# (ModelStat, SolveStat) per scenario: s1 and s2 are optimal, s3 is infeasible.
EXPECTED_REPORT = [[1.0, 1.0], [1.0, 1.0], [19.0, 1.0]]


@pytest.fixture
def guss_model(container):
    m = container
    s = gp.Set(m, "s", records=["s1", "s2", "s3"])
    x = gp.Variable(m, "x")
    p = gp.Parameter(m, "p")
    ps = gp.Parameter(m, "ps", domain=s, records=[("s1", 1), ("s2", -5), ("s3", 3)])
    lower = gp.Equation(m, "lower")
    lower[...] = x >= p
    upper = gp.Equation(m, "upper")
    upper[...] = x <= 2
    model = gp.Model(
        m,
        "mdl",
        equations=[lower, upper],
        problem="lp",
        sense="min",
        objective=x,
    )
    attributes = gp.Set(m, "attr", records=["ModelStat", "SolveStat"])
    report = gp.Parameter(m, "rep", domain=[s, attributes])
    options = gp.Parameter(m, "opt", domain="*", records=[("SkipBaseCase", 1)])
    scenario_dict = gp.GUSSScenarioDict(m, "dict", s)
    scenario_dict.add_param(p, ps)

    return m, model, scenario_dict, options, report


def test_report_after_plain_solve(guss_model):
    _, model, scenario_dict, options, report = guss_model
    scenario_dict.add_options(options, report)

    model.solve(scenario=scenario_dict)

    assert report.toDense().tolist() == EXPECTED_REPORT


def test_report_in_loop(guss_model):
    m, model, scenario_dict, options, report = guss_model
    scenario_dict.add_options(options, report)
    once = gp.Set(m, "once", records=["k1"])

    with gp.Loop(once):
        model.solve(scenario=scenario_dict)

    assert report.toDense().tolist() == EXPECTED_REPORT


def test_report_validation(guss_model):
    m, _, scenario_dict, options, report = guss_model

    with pytest.raises(ValidationError, match="GUSS report must be a Parameter"):
        scenario_dict.add_options(options, report=m["attr"])

    with pytest.raises(ValidationError, match="to have 2 not 1 dimensions"):
        scenario_dict.add_options(options, report=m["ps"])

    other = gp.Container()
    other_report = gp.Parameter(other, "rep", domain=["*", "*"])
    with pytest.raises(ValidationError, match="must belong to the same container"):
        scenario_dict.add_options(options, report=other_report)
    other.close()

    # A rejected report must not register the options entry.
    scenario_dict.add_options(options, report)
