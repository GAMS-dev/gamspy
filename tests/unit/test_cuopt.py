from __future__ import annotations

import ctypes
import os
import platform
import shutil
import stat
import zipfile

import pytest
import typer

import gamspy.utils as utils
from gamspy_cli import cuopt

pytestmark = pytest.mark.unit


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "DEFAULT_DIR", str(tmp_path))
    yield tmp_path


@pytest.mark.skipif(
    platform.system() != "Linux", reason="Cuopt is only supported on Linux."
)
def test_architecture(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Linux")

    for machine in ("x86_64", "AMD64"):
        monkeypatch.setattr(platform, "machine", lambda machine=machine: machine)
        assert cuopt._get_architecture() == "x86_64"

    for machine in ("aarch64", "arm64"):
        monkeypatch.setattr(platform, "machine", lambda machine=machine: machine)
        assert cuopt._get_architecture() == "arm64"

    monkeypatch.setattr(platform, "machine", lambda: "ppc64le")
    with pytest.raises(typer.Exit):
        cuopt._get_architecture()

    # cuOpt is only available on Linux (and on Windows through WSL2).
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    for system in ("Windows", "Darwin"):
        monkeypatch.setattr(platform, "system", lambda system=system: system)
        with pytest.raises(typer.Exit):
            cuopt._get_architecture()


@pytest.mark.skipif(
    platform.system() != "Linux", reason="Cuopt is only supported on Linux."
)
def test_detect_cuda_version(monkeypatch):
    def get_loader(available: set[str]):
        def load(name):
            if name not in available:
                raise OSError(f"{name}: cannot open shared object file")

            return None

        return load

    monkeypatch.setattr(ctypes, "CDLL", get_loader(set()))
    assert cuopt._detect_cuda_version() is None

    monkeypatch.setattr(ctypes, "CDLL", get_loader({"libcudart.so.12"}))
    assert cuopt._detect_cuda_version() == "12"

    monkeypatch.setattr(ctypes, "CDLL", get_loader({"libcublas.so.13"}))
    assert cuopt._detect_cuda_version() == "13"

    # The newest CUDA runtime on the machine wins.
    monkeypatch.setattr(
        ctypes, "CDLL", get_loader({"libcudart.so.12", "libcudart.so.13"})
    )
    assert cuopt._detect_cuda_version() == "13"


def test_version(monkeypatch):
    # A pinned release is installed by default, not the latest one.
    monkeypatch.delenv(cuopt.CUOPT_VERSION_ENV, raising=False)
    assert cuopt._get_version() == cuopt.DEFAULT_VERSION

    monkeypatch.setenv(cuopt.CUOPT_VERSION_ENV, "")
    assert cuopt._get_version() == cuopt.DEFAULT_VERSION

    # The `v` of the release tags is optional.
    for version in ("0.0.9", "v0.0.9"):
        monkeypatch.setenv(cuopt.CUOPT_VERSION_ENV, version)
        assert cuopt._get_version() == "v0.0.9"


def test_cuda_version(monkeypatch):
    monkeypatch.delenv(cuopt.CUDA_VERSION_ENV, raising=False)
    assert cuopt._get_cuda_version() is None

    monkeypatch.setenv(cuopt.CUDA_VERSION_ENV, "")
    assert cuopt._get_cuda_version() is None

    for version in cuopt.CUDA_VERSIONS:
        monkeypatch.setenv(cuopt.CUDA_VERSION_ENV, version)
        assert cuopt._get_cuda_version() == version

    monkeypatch.setenv(cuopt.CUDA_VERSION_ENV, "11")
    with pytest.raises(typer.Exit):
        cuopt._get_cuda_version()


def test_cuda_runtime(monkeypatch):
    monkeypatch.delenv(cuopt.CUDA_RUNTIME_ENV, raising=False)
    assert cuopt._install_cuda_runtime()

    monkeypatch.setenv(cuopt.CUDA_RUNTIME_ENV, "")
    assert cuopt._install_cuda_runtime()

    monkeypatch.setenv(cuopt.CUDA_RUNTIME_ENV, "1")
    assert cuopt._install_cuda_runtime()

    monkeypatch.setenv(cuopt.CUDA_RUNTIME_ENV, "0")
    assert not cuopt._install_cuda_runtime()

    for value in ("true", "no", "2"):
        monkeypatch.setenv(cuopt.CUDA_RUNTIME_ENV, value)
        with pytest.raises(typer.Exit):
            cuopt._install_cuda_runtime()


@pytest.mark.skipif(
    platform.system() != "Linux", reason="Cuopt is only supported on Linux."
)
def test_extract(tmp_path):
    archive_path = os.path.join(tmp_path, "cuopt-link-release.zip")
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name, mode in (("gmscuopt.out", 0o755), ("optcuopt.def", 0o644)):
            info = zipfile.ZipInfo(name)
            info.external_attr = mode << 16
            archive.writestr(info, "content of " + name)

        # Archive members must not be able to escape the system directory.
        info = zipfile.ZipInfo("../escaped.so")
        info.external_attr = 0o755 << 16
        archive.writestr(info, "escaped")

    directory = tmp_path / "system_directory"
    directory.mkdir()
    names = cuopt._extract(archive_path, str(directory))

    assert sorted(names) == ["escaped.so", "gmscuopt.out", "optcuopt.def"]
    assert not os.path.exists(os.path.join(tmp_path, "escaped.so"))

    # The permissions of the archive members must be preserved. Otherwise, GAMS
    # cannot execute the solver link.
    executable = directory / "gmscuopt.out"
    assert stat.S_IMODE(executable.stat().st_mode) == 0o755
    assert stat.S_IMODE((directory / "optcuopt.def").stat().st_mode) == 0o644
    assert executable.read_text() == "content of gmscuopt.out"


@pytest.mark.skipif(
    platform.system() != "Linux", reason="Cuopt is only supported on Linux."
)
def test_installed_files(manifest):
    assert cuopt.get_installed_files() == []

    cuopt._set_installed_files(["gmscuopt.out", "gamsconfig.yaml"])
    assert cuopt.get_installed_files() == ["gamsconfig.yaml", "gmscuopt.out"]

    os.unlink(os.path.join(manifest, cuopt.MANIFEST_FILE))
    assert cuopt.get_installed_files() == []


@pytest.mark.skipif(
    platform.system() != "Linux", reason="Cuopt is only supported on Linux."
)
def test_backup_config(tmp_path):
    directory = str(tmp_path)
    config_path = os.path.join(directory, cuopt.CONFIG_FILE)
    backup_path = os.path.join(directory, cuopt.BACKUP_FILE)

    # Nothing to back up.
    cuopt._backup_config(directory, [])
    assert not os.path.exists(backup_path)

    with open(config_path, "w", encoding="utf-8") as file:
        file.write("commandLineParameters:\n")

    # A gamsconfig.yaml of the user is backed up.
    cuopt._backup_config(directory, [])
    with open(backup_path, encoding="utf-8") as file:
        assert file.read() == "commandLineParameters:\n"

    # A gamsconfig.yaml of a previous cuOpt installation is not.
    os.unlink(backup_path)
    cuopt._backup_config(directory, [cuopt.CONFIG_FILE])
    assert not os.path.exists(backup_path)


# licCodes must not be read as a number, it is a string of two digit license codes.
CUOPT_CONFIG = """solverConfig:
  - cuopt:
      minVersion: 49
      licCodes: 000102030405
      modelTypes:
        - LP
        - MIP
"""


@pytest.fixture
def system_directory(tmp_path):
    directory = tmp_path / "gamspy_base"
    directory.mkdir()
    with open(directory / cuopt.SOLVER_CONFIG_FILE, "w", encoding="utf-8") as file:
        file.write(CUOPT_CONFIG)

    yield directory


def test_load_config(tmp_path):
    path = os.path.join(tmp_path, cuopt.CONFIG_FILE)

    with open(path, "w", encoding="utf-8") as file:
        file.write("")

    assert cuopt._load_config(path) == {}

    with open(path, "w", encoding="utf-8") as file:
        file.write(CUOPT_CONFIG)

    assert list(cuopt._load_config(path)) == [cuopt.SOLVER_CONFIG_SECTION]

    # Neither an invalid YAML file nor a valid one that is not a mapping of
    # sections can be merged into.
    for content in ("solverConfig:\n  - cuopt:\n   bad: indentation\n", "- cuopt\n"):
        with open(path, "w", encoding="utf-8") as file:
            file.write(content)

        with pytest.raises(typer.Exit):
            cuopt._load_config(path)


def test_get_solver_config(system_directory):
    entries = [
        "  - cuopt:",
        "      minVersion: 49",
        "      licCodes: 000102030405",
        "      modelTypes:",
        "        - LP",
        "        - MIP",
    ]
    assert cuopt._get_solver_config(str(system_directory)) == entries

    # GAMSPy only knows how to merge a solverConfig section.
    source_path = system_directory / cuopt.SOLVER_CONFIG_FILE
    source_path.write_text(
        CUOPT_CONFIG + "commandLineParameters:\n  - LP:\n      value: cuopt\n"
    )
    with pytest.raises(typer.Exit):
        _ = cuopt._get_solver_config(str(system_directory))


def test_merge_solver_config():
    entries = ["  - cuopt:", "      licCodes: 000102030405"]
    contributed = [cuopt.MARKER_BEGIN, *entries, cuopt.MARKER_END]

    # A configuration without a solverConfig section gets the whole section.
    assert cuopt._merge_solver_config("", entries, "").splitlines() == [
        cuopt.MARKER_BEGIN,
        "solverConfig:",
        *entries,
        cuopt.MARKER_END,
    ]
    assert cuopt._merge_solver_config(
        "commandLineParameters:\n  - LP:\n      value: cplex\n", entries, ""
    ).splitlines() == [
        "commandLineParameters:",
        "  - LP:",
        "      value: cplex",
        cuopt.MARKER_BEGIN,
        "solverConfig:",
        *entries,
        cuopt.MARKER_END,
    ]

    # `...` ends the document, so the section has to go before it. GAMS and
    # pyyaml both ignore whatever follows the marker.
    for marker in ("...", "... # done"):
        assert cuopt._merge_solver_config(
            f"---\nfoo: bar\n{marker}\n*** ", entries, ""
        ).splitlines() == [
            "---",
            "foo: bar",
            cuopt.MARKER_BEGIN,
            "solverConfig:",
            *entries,
            cuopt.MARKER_END,
            marker,
            "*** ",
        ]

    # A scalar that only starts with dots is not a marker.
    assert cuopt._merge_solver_config("...foo\n", entries, "").splitlines()[-1] == (
        cuopt.MARKER_END
    )

    # The entries are added behind the ones of an existing section, since GAMS
    # lets the last entry of a solver win. Comments and blank lines are kept.
    text = (
        "# my own configuration\n"
        "solverConfig:  # my solvers\n"
        "  - mysolver:\n"
        "      licCodes: 000102030405\n"
        "\n"
        "# the parameters follow\n"
        "commandLineParameters:\n"
        "  - LP:\n"
        "      value: mysolver\n"
    )
    assert cuopt._merge_solver_config(text, entries, "").splitlines() == [
        "# my own configuration",
        "solverConfig:  # my solvers",
        "  - mysolver:",
        "      licCodes: 000102030405",
        *contributed,
        "",
        "# the parameters follow",
        "commandLineParameters:",
        "  - LP:",
        "      value: mysolver",
    ]

    # The entries are indented like the ones that are already there. A block
    # sequence may be indented like the section that holds it.
    text = "solverConfig:\n- mysolver:\n    licCodes: 000102030405\n"
    assert cuopt._merge_solver_config(text, entries, "").splitlines() == [
        "solverConfig:",
        "- mysolver:",
        "    licCodes: 000102030405",
        cuopt.MARKER_BEGIN,
        "- cuopt:",
        "    licCodes: 000102030405",
        cuopt.MARKER_END,
    ]

    # A solverConfig section that is not a block of entries cannot be merged
    # into, because a second solverConfig key would hide it.
    with pytest.raises(typer.Exit):
        _ = cuopt._merge_solver_config("solverConfig: []\n", entries, "")


def test_strip_solver_config():
    entries = ["  - cuopt:", "      licCodes: 000102030405"]
    text = (
        "# my own configuration\n"
        "solverConfig:\n"
        "  - mysolver:\n"
        "      licCodes: 000102030405\n"
    )

    # Merging and stripping have to restore the configuration byte by byte.
    assert (
        cuopt._strip_solver_config(cuopt._merge_solver_config(text, entries, ""))
        == text
    )
    assert cuopt._strip_solver_config(cuopt._merge_solver_config("", entries, "")) == ""

    # A configuration that GAMSPy did not merge into is left alone.
    assert cuopt._strip_solver_config(text) == text


def test_register_solver(system_directory):
    config_path = system_directory / cuopt.CONFIG_FILE

    # Without a gamsconfig.yaml, GAMSPy creates one.
    assert cuopt._register_solver(str(system_directory), [])
    assert utils._parse_solver_config(config_path.read_text()) == {
        "CUOPT": ["LP", "MIP"]
    }

    # A gamsconfig.yaml that GAMSPy created stays an installed file and is not
    # extended a second time.
    text = config_path.read_text()
    assert cuopt._register_solver(str(system_directory), [cuopt.CONFIG_FILE])
    assert config_path.read_text() == text

    # An existing gamsconfig.yaml of the user is merged into but not installed.
    config = (
        "commandLineParameters:\n"
        "  - threads:\n"
        "      value: 4\n"
        "solverConfig:\n"
        "  - mysolver:\n"
        "      licCodes: 000102030405\n"
        "      modelTypes:\n"
        "        - NLP\n"
    )
    config_path.write_text(config)
    assert not cuopt._register_solver(str(system_directory), [])
    assert utils._parse_solver_config(config_path.read_text()) == {
        "MYSOLVER": ["NLP"],
        "CUOPT": ["LP", "MIP"],
    }

    # The configuration of the user must survive unchanged.
    assert cuopt._strip_solver_config(config_path.read_text()) == config
    assert "licCodes: 000102030405" in config_path.read_text()


def test_unregister_solver(system_directory):
    config_path = system_directory / cuopt.CONFIG_FILE

    # Nothing to unregister.
    cuopt._unregister_solver(str(system_directory))
    assert not config_path.exists()

    # A gamsconfig.yaml that only holds the cuOpt entries is removed.
    _ = cuopt._register_solver(str(system_directory), [])
    cuopt._unregister_solver(str(system_directory))
    assert not config_path.exists()

    # The entries of the user are kept.
    config = "solverConfig:\n  - mysolver:\n      modelTypes:\n        - NLP\n"
    config_path.write_text(config)
    _ = cuopt._register_solver(str(system_directory), [])
    cuopt._unregister_solver(str(system_directory))
    assert config_path.read_text() == config


@pytest.fixture
def release(monkeypatch, system_directory, manifest):
    """A fake cuOpt installation into system_directory."""
    archive_path = manifest / "release.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(cuopt.SOLVER_CONFIG_FILE, CUOPT_CONFIG)
        archive.writestr("gmscuopt.out", "content of gmscuopt.out")

    monkeypatch.setattr(
        utils, "_get_gamspy_base_directory", lambda: str(system_directory)
    )
    monkeypatch.setattr(cuopt, "_get_architecture", lambda: "x86_64")
    monkeypatch.setattr(cuopt, "_get_cuda_version", lambda: "13")
    monkeypatch.setattr(cuopt, "_install_cuda_runtime", lambda: False)
    monkeypatch.setattr(cuopt, "_get_asset_urls", lambda names, version: [""])
    monkeypatch.setattr(
        cuopt, "_download", lambda url, path: shutil.copy(archive_path, path)
    )

    yield archive_path


@pytest.mark.skipif(
    platform.system() != "Linux", reason="Cuopt is only supported on Linux."
)
def test_install_creates_config(release, system_directory):
    config_path = system_directory / cuopt.CONFIG_FILE
    installed = [cuopt.CONFIG_FILE, cuopt.SOLVER_CONFIG_FILE, "gmscuopt.out"]

    # Without a gamsconfig.yaml of the user, the created one belongs to GAMSPy
    # and is reported for the RECORD of the wheel.
    assert cuopt.install() == installed
    assert cuopt.get_installed_files() == installed
    assert not os.path.exists(system_directory / cuopt.BACKUP_FILE)
    assert utils._parse_solver_config(config_path.read_text()) == {
        "CUOPT": ["LP", "MIP"]
    }

    # Reinstalling keeps gamsconfig.yaml an installed file.
    text = config_path.read_text()
    assert cuopt.install() == installed
    assert config_path.read_text() == text

    cuopt.uninstall()
    assert not config_path.exists()
    assert not os.path.exists(system_directory / cuopt.SOLVER_CONFIG_FILE)
    assert cuopt.get_installed_files() == []


@pytest.mark.skipif(
    platform.system() != "Linux", reason="Cuopt is only supported on Linux."
)
def test_install_merges_into_config_of_user(release, system_directory):
    config_path = system_directory / cuopt.CONFIG_FILE
    config = "solverConfig:\n  - mysolver:\n      modelTypes:\n        - NLP\n"
    config_path.write_text(config)

    # The gamsconfig.yaml of the user is neither reported for the RECORD of the
    # wheel nor removed on uninstallation, only merged into and backed up.
    assert cuopt.install() == [cuopt.SOLVER_CONFIG_FILE, "gmscuopt.out"]
    assert (system_directory / cuopt.BACKUP_FILE).read_text() == config
    assert utils._parse_solver_config(config_path.read_text()) == {
        "MYSOLVER": ["NLP"],
        "CUOPT": ["LP", "MIP"],
    }

    cuopt.uninstall()
    assert config_path.read_text() == config
    assert not os.path.exists(system_directory / cuopt.BACKUP_FILE)
