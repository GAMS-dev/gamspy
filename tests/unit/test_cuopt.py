from __future__ import annotations

import ctypes
import os
import platform
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
