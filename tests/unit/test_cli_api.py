from __future__ import annotations

import os
import sys

import gamspy_base
import pytest

import gamspy as gp
import gamspy.utils as utils
from gamspy import cli
from gamspy.exceptions import CLIError, GamspyBaseException
from gamspy_cli.cli import main

pytestmark = pytest.mark.unit


@pytest.fixture
def gdx_files(tmp_path):
    m = gp.Container()
    i = gp.Set(m, "i", records=["i1", "i2"])
    first = str(tmp_path / "first.gdx")
    m.write(first)

    i.setRecords(["i1", "i3"])
    second = str(tmp_path / "second.gdx")
    m.write(second)

    return first, second


def test_cli_error():
    error = CLIError("message", 3)
    assert isinstance(error, GamspyBaseException)
    assert error.message == "message"
    assert error.return_code == 3
    assert CLIError("message").return_code == 1


def test_gdx_dump(gdx_files):
    first, _ = gdx_files
    assert cli.gdx_dump(first, symb="i", format="csv").splitlines() == [
        '"Dim1"',
        '"i1"',
        '"i2"',
    ]

    # The extension is optional
    assert cli.gdx_dump(first.removesuffix(".gdx"), symbols=True) == cli.gdx_dump(
        first, symbols=True
    )

    with pytest.raises(CLIError, match="File not found"):
        cli.gdx_dump("does_not_exist.gdx")

    with pytest.raises(CLIError, match="Invalid Delim"):
        cli.gdx_dump(first, delim="bla")


def test_gdx_diff(gdx_files, tmp_path):
    first, second = gdx_files
    diffile = str(tmp_path / "diff.gdx")

    result = cli.gdx_diff(first, first, diffile)
    assert not result.has_differences
    assert "No differences found" in result.output

    result = cli.gdx_diff(first, second, diffile)
    assert result.has_differences
    assert os.path.isfile(diffile)

    result = cli.gdx_diff(first, second, diffile, skipid=["i"])
    assert not result.has_differences

    result = cli.gdx_diff(first, second, diffile, skip_regex="^i")
    assert not result.has_differences

    with pytest.raises(CLIError, match="File not found"):
        cli.gdx_diff(first, "does_not_exist.gdx", diffile)

    with pytest.raises(CLIError, match="Invalid field"):
        cli.gdx_diff(first, second, diffile, field="bla")


def test_gdx_diff_with_diffile_in_another_directory(gdx_files, tmp_path, monkeypatch):
    first, second = gdx_files
    os.makedirs(tmp_path / "cwd")
    os.makedirs(tmp_path / "out")
    monkeypatch.chdir(tmp_path / "cwd")

    # Relative paths are relative to the working directory of the caller.
    result = cli.gdx_diff(
        os.path.relpath(first),
        os.path.relpath(second),
        os.path.join("..", "out", "diff"),
    )
    assert result.has_differences
    assert os.listdir(tmp_path / "out") == ["diff.gdx"]

    # gdxdiff must not leave its temporary file behind.
    assert os.listdir(tmp_path / "cwd") == []


def test_show():
    assert cli.show_base() == gamspy_base.directory

    info = cli.show_license()
    assert info.path == utils._get_license_path(gamspy_base.directory)
    with open(info.path, encoding="utf-8") as file:
        assert info.content.splitlines() == [line.strip() for line in file]

    assert info.licensed_solvers


def test_list_solvers():
    installed = set(utils.getInstalledSolvers(gamspy_base.directory))

    solvers = cli.list_solvers()
    assert solvers
    assert all(solver.installed for solver in solvers)
    assert {solver.name for solver in solvers} <= installed
    assert all(not solver.capabilities for solver in solvers if solver.is_tool)

    names = {solver.name for solver in cli.list_solvers(all=True)}
    assert {solver.name for solver in solvers} <= names
    assert set(utils.getAvailableSolvers()) <= names


def test_install_solver_errors():
    with pytest.raises(CLIError, match="Solver name is missing"):
        cli.install_solver()

    with pytest.raises(CLIError, match="is not valid"):
        cli.install_solver("bla")


def test_uninstall_solver_errors():
    with pytest.raises(CLIError, match="Solver name is missing"):
        cli.uninstall_solver([])

    # Default solvers cannot be uninstalled.
    with pytest.raises(CLIError, match="is not valid"):
        cli.uninstall_solver(gamspy_base.default_solvers[0])


def test_install_license_errors():
    with pytest.raises(CLIError, match="Invalid access code or license file"):
        cli.install_license("bla")


def test_mps2gms_errors():
    with pytest.raises(CLIError, match="File not found"):
        cli.mps2gms("does_not_exist.mps")


def test_main_reports_cli_error(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["gamspy", "gdx", "dump", "does_not_exist.gdx"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    assert "File not found: does_not_exist.gdx" in capsys.readouterr().err


@pytest.fixture
def fake_gamspy_base(tmp_path, monkeypatch):
    """Records the solver (un)installations instead of running pip."""
    calls = []
    addons = set()

    def install_solver(name, *, use_uv=False, install_package=True):
        calls.append(("install", name, use_uv, install_package))
        if name == "bad":
            raise gamspy_base.InstallationError("pip failed")
        addons.add(name.upper())

    def uninstall_solver(name, *, use_uv=False, uninstall_package=True):
        calls.append(("uninstall", name, use_uv, uninstall_package))
        addons.discard(name.upper())

    monkeypatch.setattr(gamspy_base, "install_solver", install_solver)
    monkeypatch.setattr(gamspy_base, "uninstall_solver", uninstall_solver)
    monkeypatch.setattr(
        gamspy_base, "get_installed_addon_solvers", lambda: sorted(addons)
    )
    monkeypatch.setattr(
        utils, "getInstallableSolvers", lambda: ["BAD", "CUOPT", "HIGHS", "SCIP"]
    )
    monkeypatch.setattr(utils, "DEFAULT_DIR", str(tmp_path / "GAMSPy"))
    return calls, addons, tmp_path / "GAMSPy" / "solvers.txt"


def test_install_and_uninstall_solver(fake_gamspy_base):
    calls, addons, addons_path = fake_gamspy_base

    cli.install_solver(["highs", "SCIP"], use_uv=True)
    assert calls == [
        ("install", "highs", True, True),
        ("install", "scip", True, True),
    ]
    assert addons_path.read_text().split() == ["HIGHS", "SCIP"]

    cli.install_solver("highs", skip_pip_install=True)
    assert calls[-1] == ("install", "highs", False, False)

    # The errors of gamspy_base are reported as CLIError.
    with pytest.raises(CLIError, match="pip failed"):
        cli.install_solver("bad")
    assert addons_path.read_text().split() == ["HIGHS", "SCIP"]

    # solvers.txt is the record for --existing-solvers.
    calls.clear()
    cli.install_solver(existing_solvers=True)
    assert [call[1] for call in calls] == ["highs", "scip"]

    calls.clear()
    cli.uninstall_solver("highs", skip_pip_uninstall=True)
    assert calls == [("uninstall", "highs", False, False)]
    assert addons_path.read_text().split() == ["SCIP"]

    # Only installed add-on solvers can be uninstalled.
    with pytest.raises(CLIError, match="is not valid"):
        cli.uninstall_solver("highs")

    cli.uninstall_solver(uninstall_all_solvers=True)
    assert calls[-1] == ("uninstall", "scip", False, True)
    assert not addons


def test_install_licensed_solvers_skips_deprecated(fake_gamspy_base, monkeypatch):
    calls, _, _ = fake_gamspy_base
    monkeypatch.setattr(
        utils, "getInstallableSolvers", lambda: ["EXAMINER2", "HIGHS", "SCIP"]
    )
    monkeypatch.setattr(
        gamspy_base,
        "get_licensed_solvers",
        lambda path: ["EXAMINER2", "HIGHS", "SCIP"],
    )
    monkeypatch.setattr(gamspy_base, "get_solver_capabilities", lambda: {"SCIP": []})

    cli.install_solver(licensed=True)
    assert [call[1] for call in calls] == ["highs"]


def test_uninstall_same_solver_twice(fake_gamspy_base):
    calls, addons, _ = fake_gamspy_base
    addons.add("HIGHS")

    with pytest.raises(CLIError, match="is not valid"):
        cli.uninstall_solver(["highs", "highs"])
    assert calls == [("uninstall", "highs", False, True)]


def test_uninstall_unlicensed_solvers_keeps_cuopt(fake_gamspy_base, monkeypatch):
    calls, addons, _ = fake_gamspy_base
    addons.update({"CUOPT", "HIGHS", "SCIP"})
    monkeypatch.setattr(gamspy_base, "get_licensed_solvers", lambda path: ["SCIP"])

    cli.uninstall_solver(unlicensed=True)
    assert [call[1] for call in calls] == ["highs"]
    assert addons == {"CUOPT", "SCIP"}
