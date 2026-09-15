from __future__ import annotations

import ast
import importlib
import os
import pathlib
import platform
import time

import pytest

import gamspy as gp
import gamspy._validation as validation
import gamspy.utils as utils
from gamspy import Problem, Set
from gamspy.exceptions import ValidationError

pytestmark = pytest.mark.unit


def test_utils(container):
    m = container
    i = Set(m, "i", records=["i1", "i2"])
    assert utils._get_domain_str([i, "b", "*"]) == '(i,"b",*)'
    with pytest.raises(ValidationError):
        utils._get_domain_str([5])

    assert not utils.checkAllSame([1, 2], [2])
    assert not utils.checkAllSame([1, 2], [2, 3])
    assert utils.checkAllSame([1, 2], [1, 2])


def test_isin(container):
    m = container
    i = Set(m, "i")
    j = Set(m, "j")
    k = Set(m, "k")
    symbols = [i, j]

    assert utils.isin(i, symbols)
    assert not utils.isin(k, symbols)


def test_available_solvers(container):
    available_solvers = utils.getAvailableSolvers()

    expected = [
        "BARON",
        "CBC",
        "CONOPT3",
        "CONOPT4",
        "CONVERT",
        "COPT",
        "CPLEX",
        "CUOPT",
        "DE",
        "DICOPT",
        "EXAMINER",
        "EXAMINER2",
        "GUROBI",
        "HIGHS",
        "IPOPT",
        "IPOPTH",
        "KESTREL",
        "KNITRO",
        "MILES",
        "MINOS",
        "MOSEK",
        "MPSGE",
        "NLPEC",
        "PATH",
        "PATHNLP",
        "RESHOP",
        "SBB",
        "SCIP",
        "SHOT",
        "SNOPT",
        "SOPLEX",
        "XPRESS",
    ]

    if platform.system() == "Linux" and platform.machine() == "aarch64":
        expected.remove("BARON")
        expected.remove("KNITRO")

    # cuOpt is only available on Linux (and on Windows through WSL2).
    if platform.system() != "Linux":
        expected.remove("CUOPT")

    assert available_solvers == expected


def test_default_solvers():
    import gamspy_base

    default_solvers = utils.getDefaultSolvers(gamspy_base.directory)

    expected = {
        "CNS": "PATH",
        "DNLP": "CONOPT",
        "EMP": "CONVERT",
        "LP": "CPLEX",
        "MCP": "PATH",
        "MINLP": "SBB",
        "MIP": "CPLEX",
        "MIQCP": "SBB",
        "MPEC": "NLPEC",
        "NLP": "CONOPT",
        "QCP": "CONOPT",
        "RMINLP": "CONOPT",
        "RMIP": "CPLEX",
        "RMIQCP": "CONOPT",
    }

    assert default_solvers == expected


def test_solver_and_capability_caching():
    import gamspy_base

    start = time.perf_counter_ns()
    _ = utils.getDefaultSolvers(gamspy_base.directory)
    first_time = time.perf_counter_ns() - start

    start = time.perf_counter_ns()
    _ = utils.getDefaultSolvers(gamspy_base.directory)
    second_time = time.perf_counter_ns() - start

    assert second_time < first_time, f"{first_time=}, {second_time=}"

    start = time.perf_counter_ns()
    _ = utils.getSolverCapabilities(gamspy_base.directory)
    first_time = time.perf_counter_ns() - start

    start = time.perf_counter_ns()
    _ = utils.getSolverCapabilities(gamspy_base.directory)
    second_time = time.perf_counter_ns() - start

    assert second_time < first_time, f"{first_time=}, {second_time=}"

    start = time.perf_counter_ns()
    _ = utils.getInstalledSolvers(gamspy_base.directory)
    first_time = time.perf_counter_ns() - start

    start = time.perf_counter_ns()
    _ = utils.getInstalledSolvers(gamspy_base.directory)
    second_time = time.perf_counter_ns() - start

    assert second_time < first_time, f"{first_time=}, {second_time=}"


CUOPT_CONFIG = """solverConfig:
  - cuopt:
      minVersion: 49
      fileType: 1001
      dictType: 0
      licCodes: 000102030405
      scriptName: gmscuopt.run
      executableName: gmscuopt.out
      modelTypes:
        - LP
        - RMIP
        - MIP
        - QCP
        - MIQCP
        - RMIQCP
"""

CUOPT_MODEL_TYPES = ["LP", "RMIP", "MIP", "QCP", "MIQCP", "RMIQCP"]


@pytest.fixture
def config_solver():
    """
    Registers a solver through a gamsconfig.yaml file in the system directory
    the same way the GAMS solver link for NVIDIA cuOpt does.
    """
    import gamspy_base

    caches = (
        utils._defaults,
        utils._capabilities,
        utils._installed_solvers,
        utils._config_solvers,
    )
    for cache in caches:
        cache.clear()

    config_path = os.path.join(gamspy_base.directory, "gamsconfig.yaml")
    backup = None
    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as file:
            backup = file.read()

    with open(config_path, "w", encoding="utf-8") as file:
        file.write(CUOPT_CONFIG)

    try:
        yield gamspy_base.directory
    finally:
        if backup is None:
            os.unlink(config_path)
        else:
            with open(config_path, "w", encoding="utf-8") as file:
                file.write(backup)

        for cache in caches:
            cache.clear()


@pytest.mark.skipif(
    platform.system() != "Linux", reason="Cuopt is only supported on Linux."
)
def test_config_solvers(config_solver):
    system_directory = config_solver

    assert utils._get_config_solvers(system_directory) == {"CUOPT": CUOPT_MODEL_TYPES}

    # Solvers of the capabilities file are not affected.
    assert "CUOPT" not in utils._get_capabilities_file_solvers(system_directory)
    assert "CPLEX" in utils._get_capabilities_file_solvers(system_directory)

    assert "CUOPT" in utils.getInstalledSolvers(system_directory)
    assert utils.getSolverCapabilities(system_directory)["CUOPT"] == CUOPT_MODEL_TYPES

    assert "CUOPT" in utils.getAvailableSolvers()
    assert "CUOPT" in utils.getInstallableSolvers()


@pytest.mark.skipif(
    platform.system() != "Linux", reason="Cuopt is only supported on Linux."
)
def test_config_solver_validation(config_solver):
    system_directory = config_solver

    validation.validate_solver_args(
        system_directory, "local", "cuopt", Problem.LP, None, None
    )

    with pytest.raises(ValidationError):
        validation.validate_solver_args(
            system_directory, "local", "cuopt", Problem.NLP, None, None
        )


def test_parse_solver_config():
    text = """solverConfig:
  - SHOT2:
      minVersion: 33
      modelTypes:
        - MINLP
        - MIQCP
  - SHOT3:
      minVersion: 34
      modelTypes:
        - LP
        - NLP
"""
    assert utils._parse_solver_config(text) == {
        "SHOT2": ["MINLP", "MIQCP"],
        "SHOT3": ["LP", "NLP"],
    }


def _declaration_plumbing_callers():
    """Every function in the package that asks for an auto-generated symbol name."""
    package_root = pathlib.Path(utils.__file__).parent
    callers = []

    for path in sorted(package_root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        module_name = "gamspy." + ".".join(
            path.relative_to(package_root).with_suffix("").parts
        )

        stack: list[ast.AST] = []

        def visit(node, stack=stack, module_name=module_name, path=path):
            enclosing = isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            )
            if enclosing:
                stack.append(node)

            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_get_symbol_name"
            ):
                owners = [n.name for n in stack]
                callers.append((module_name, tuple(owners), node.lineno, path))

            for child in ast.iter_child_nodes(node):
                visit(child)

            if enclosing:
                stack.pop()

        visit(tree)

    resolved = []
    for module_name, owners, lineno, path in callers:
        target = importlib.import_module(module_name)
        for owner in owners:
            target = getattr(target, owner)
        resolved.append((f"{path.name}:{lineno} {'.'.join(owners)}", target))

    return resolved


def test_declaration_plumbing_is_complete():
    plumbing, _pinned = utils._declaration_plumbing()
    callers = _declaration_plumbing_callers()

    assert callers, "found no callers of _get_symbol_name; the AST scan is broken"

    unregistered = [
        where for where, function in callers if id(function.__code__) not in plumbing
    ]
    assert not unregistered, (
        "these call Container._get_symbol_name but are not registered in "
        f"gamspy.utils._declaration_plumbing(): {unregistered}"
    )


def test_python_name_is_used_under_the_default_setting(set_options):
    set_options({"USE_PY_VAR_NAME": "yes-or-autogenerate"})

    with gp.Container() as m:
        declared_set = gp.Set()
        added_set = m.addSet()
        declared_parameter = gp.Parameter()

        assert declared_set.name == "declared_set"
        assert added_set.name == "added_set"
        assert declared_parameter.name == "declared_parameter"
