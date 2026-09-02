from __future__ import annotations

import os

import pytest

import gamspy._validation as validation
from gamspy import (
    Alias,
    Equation,
    Model,
    Options,
    Parameter,
    Problem,
    Sense,
    Set,
    UniverseAlias,
)
from gamspy.exceptions import ValidationError

pytestmark = pytest.mark.unit


@pytest.fixture
def validation_off(set_options):
    """Turns the global VALIDATION option off for the duration of a test."""
    set_options({"VALIDATION": 0})


def test_get_domain_path_with_string_domain(container):
    s = Set(container, "s", domain=["someLabel"])

    # a relaxed domain given as a bare label terminates the walk
    assert validation.get_domain_path(s) == ["someLabel", "s"]


def test_get_domain_path_with_alias(container):
    i = Set(container, "i", records=["a"])
    al = Alias(container, "al", i)

    # an alias contributes the set it aliases as well as itself
    assert validation.get_domain_path(al) == ["i", "al"]


def test_get_domain_path_with_subset(container):
    i = Set(container, "i", records=["a"])
    j = Set(container, "j", domain=[i])

    assert validation.get_domain_path(j) == ["i", "j"]


def test_index_components_reinserts_literals(container):
    i = Set(container, "i", records=["a"])
    j = Set(container, "j", records=["x"])
    t = Set(container, "t", records=["t1"])
    two_d = Set(container, "two_d", domain=[i, j])
    target = Parameter(container, "target", domain=[i, j, t])

    # the literal "a" occupies the first position of two_d, so the remaining
    # positions still line up with the target domain
    assert target[two_d["a", j], t].gamsRepr() == 'target(two_d("a",j),t)'


def test_expand_leaf_for_unvalidatable_leaf(container):
    u = UniverseAlias(container, "u")

    # a universe alias carries no set to validate
    assert validation._expand_leaf(u) == [None]
    assert validation._index_components(u) is None


def test_expand_leaf_for_relaxed_implicit_set(container):
    s = Set(container, "s", domain=["*", "*"])
    assert validation._expand_leaf(s["a", "b"]) == [None, None]


def test_validate_name_skipped_without_validation(validation_off):
    assert validation.validate_name("sum") == "sum"


def test_validate_model_name_skipped_without_validation(validation_off):
    assert validation.validate_model_name("_leading_underscore") == (
        "_leading_underscore"
    )


def test_validate_model_name_errors():
    with pytest.raises(ValueError, match="at least one character"):
        validation.validate_model_name("")

    with pytest.raises(ValueError, match="too long"):
        validation.validate_model_name("m" * 64)

    with pytest.raises(ValidationError, match="cannot begin with a '_'"):
        validation.validate_model_name("_model")

    with pytest.raises(ValidationError, match="invalid model name"):
        validation.validate_model_name("my-model")


def test_validate_model_name_accepts_underscores():
    assert validation.validate_model_name("my_model_2") == "my_model_2"


def test_validate_model_skipped_without_validation(validation_off):
    # the strings are still converted, but nothing is checked
    assert validation.validate_model([], None, "lp", "min") == (
        Problem.LP,
        Sense.MIN,
    )
    assert validation.validate_model([], None, Problem.MIP, Sense.MAX) == (
        Problem.MIP,
        Sense.MAX,
    )


def test_validate_model_requires_iterable_equations():
    with pytest.raises(TypeError, match="`equations` must be an Iterable"):
        validation.validate_model(5, None, "lp", "min")

    # CNS and MCP models are allowed to skip the equation list entirely
    assert validation.validate_model(5, None, "cns", "feasibility") == (
        Problem.CNS,
        Sense.FEASIBILITY,
    )


def test_validate_global_options_skipped_without_validation(validation_off):
    assert isinstance(validation.validate_global_options(None), Options)
    assert isinstance(validation.validate_global_options({"reslim": 5}), Options)

    given = Options()
    assert validation.validate_global_options(given) is given


def test_validate_solver_args_non_string_solver(container):
    with pytest.raises(TypeError, match="`solver` argument must be a string"):
        validation.validate_solver_args(
            container.system_directory, "local", 5, "LP", None, None
        )


def test_validate_solver_args_unsupported_remote_solvers(container):
    for solver in ("mpsge", "kestrel"):
        with pytest.raises(ValidationError, match="not a valid solver for NEOS"):
            validation.validate_solver_args(
                container.system_directory, "neos", solver, "LP", None, None
            )

    with pytest.raises(ValidationError, match="not a valid solver for GAMS Engine"):
        validation.validate_solver_args(
            container.system_directory, "engine", "mpsge", "LP", None, None
        )


def test_validate_equations_undefined_domained_equation(container):
    i = Set(container, "i", records=["a"])
    e = Equation(container, "e", domain=[i])
    model = Model(container, "mdl", equations=[e], problem="LP", sense="min")

    with pytest.raises(ValidationError, match="no equation definition was found"):
        validation.validate_equations(model)


def test_validate_equations_undefined_scalar_equation(container):
    e = Equation(container, "e")
    model = Model(container, "mdl", equations=[e], problem="LP", sense="min")

    with pytest.raises(ValidationError, match="declared as a scalar equation"):
        validation.validate_equations(model)


def test_validate_equations_skips_mpsge_models(container):
    i = Set(container, "i", records=["a"])
    e = Equation(container, "e", domain=[i])
    model = Model(container, "hansen", equations=[e], problem="mcp")

    # MPSGE equations are generated on the GAMS side, so there is no definition
    # to inspect and the undefined-equation check must be skipped
    container._mpsge_models.append("hansen")
    assert validation.validate_equations(model) is None


def test_get_def_file_special_cases(container):
    sysdir = container.system_directory

    assert (
        os.path.basename(validation._get_def_file(sysdir, "conopt4")) == "optconopt.def"
    )
    assert (
        os.path.basename(validation._get_def_file(sysdir, "examiner2"))
        == "optexaminer.def"
    )
    assert (
        os.path.basename(validation._get_def_file(sysdir, "ipopth")) == "optipopt.def"
    )
    assert os.path.basename(validation._get_def_file(sysdir, "cplex")) == "optcplex.def"


def test_validate_solver_options_missing_file(container):
    missing = os.path.join(container.working_directory, "does_not_exist.opt")

    with pytest.raises(RuntimeError, match="Error while reading"):
        validation.validate_solver_options(container.system_directory, missing, "cplex")


def test_validate_solver_options_rejects_unknown_option(container):
    path = os.path.join(container.working_directory, "bad.opt")
    with open(path, "w") as file:
        file.write("this is not a valid option\n")

    with pytest.raises(ValidationError, match="Unknown option"):
        validation.validate_solver_options(container.system_directory, path, "cplex")


def test_validate_solver_options_with_bad_system_directory(container):
    path = os.path.join(container.working_directory, "empty.opt")
    with open(path, "w") as file:
        file.write("")

    with pytest.raises(RuntimeError):
        validation.validate_solver_options(
            os.path.join(container.working_directory, "not_a_sysdir"), path, "cplex"
        )


def test_validate_solver_options_skipped_without_validation(container, validation_off):
    missing = os.path.join(container.working_directory, "does_not_exist.opt")

    assert (
        validation.validate_solver_options(container.system_directory, missing, "cplex")
        is None
    )
