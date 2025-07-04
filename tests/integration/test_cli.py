from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys

import pytest

from gamspy import Container, Set

pytestmark = pytest.mark.cli
try:
    from dotenv import load_dotenv

    load_dotenv(os.getcwd() + os.sep + ".env")
except Exception:
    pass

user_dir = os.path.expanduser("~")
if platform.system() == "Linux":
    DEFAULT_DIR = os.path.join(user_dir, ".local", "share", "GAMSPy")
elif platform.system() == "Darwin":
    DEFAULT_DIR = os.path.join(
        user_dir, "Library", "Application Support", "GAMSPy"
    )
elif platform.system() == "Windows":
    DEFAULT_DIR = os.path.join(user_dir, "Documents", "GAMSPy")


@pytest.fixture
def teardown():
    # Arrange
    os.makedirs("tmp", exist_ok=True)

    # Act and assert
    yield

    # Clean up
    shutil.rmtree("tmp")


def test_install_license(teardown):
    tmp_license_path = os.path.join("tmp", "gamspy_license.txt")

    # Try to install a license with GAMS access code
    with pytest.raises(subprocess.CalledProcessError):
        _ = subprocess.run(
            [
                sys.executable,
                "-Bm",
                "gamspy",
                "install",
                "license",
                os.environ["GAMS_ACCESS_CODE"],
            ],
            check=True,
            capture_output=True,
            encoding="utf-8",
        )

    # Try to install a GAMS license (+ license)
    with open(tmp_license_path, "w") as file:
        file.write(os.environ["GAMS_ACADEMIC_LICENSE"])

    with pytest.raises(subprocess.CalledProcessError):
        _ = subprocess.run(
            [
                sys.executable,
                "-Bm",
                "gamspy",
                "install",
                "license",
                tmp_license_path,
            ],
            check=True,
            capture_output=True,
            encoding="utf-8",
        )

    # Try to install a GAMS license (/ license)
    with open(tmp_license_path, "w") as file:
        file.write(os.environ["GAMS_ACADEMIC_LICENSE2"])

    with pytest.raises(subprocess.CalledProcessError):
        _ = subprocess.run(
            [
                sys.executable,
                "-Bm",
                "gamspy",
                "install",
                "license",
                tmp_license_path,
            ],
            check=True,
            capture_output=True,
            encoding="utf-8",
        )

    # Test network license
    _ = subprocess.run(
        [
            sys.executable,
            "-Bm",
            "gamspy",
            "install",
            "license",
            os.environ["NETWORK_LICENSE_NON_ACADEMIC"],
        ],
        check=True,
    )

    gamspy_license_path = os.path.join(DEFAULT_DIR, "gamspy_license.txt")

    assert os.path.exists(gamspy_license_path)

    m = Container()
    assert m._network_license

    _ = Set(m, "i", records=["bla"])
    m.close()

    # Test invalid access code / license
    with pytest.raises(subprocess.CalledProcessError):
        _ = subprocess.run(
            [sys.executable, "-Bm", "gamspy", "install", "license", "blabla"],
            check=True,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )

    # Test installing a license from a file path.
    shutil.copy(gamspy_license_path, tmp_license_path)

    process = subprocess.run(
        [
            sys.executable,
            "-Bm",
            "gamspy",
            "install",
            "license",
            tmp_license_path,
        ],
        capture_output=True,
        encoding="utf-8",
        text=True,
    )

    assert process.returncode == 0, process.stderr

    # Test checkout
    process = subprocess.run(
        [
            sys.executable,
            "-Bm",
            "gamspy",
            "install",
            "license",
            os.environ["CHECKOUT_LICENSE"],
            "-c",
            "1",
            "-o",
            tmp_license_path,
        ],
        capture_output=True,
        encoding="utf-8",
        text=True,
    )

    assert process.returncode == 0, process.stderr

    # Test invalid port (port 100 is below the minimum port (1024))
    process = subprocess.run(
        [
            sys.executable,
            "-Bm",
            "gamspy",
            "install",
            "license",
            os.environ["LOCAL_LICENSE"],
            "-p",
            "100",
        ],
        capture_output=True,
        encoding="utf-8",
        text=True,
    )

    assert process.returncode != 0

    if platform.system() == "Linux":
        process = subprocess.run(
            [
                sys.executable,
                "-Bm",
                "gamspy",
                "install",
                "license",
                os.environ["ON_PREM_LICENSE"],
                "-s",
                "alptest.gams.com",
            ],
            capture_output=True,
            text=True,
        )

    # Recover local license
    process = subprocess.run(
        [
            sys.executable,
            "-Bm",
            "gamspy",
            "install",
            "license",
            os.environ["LOCAL_LICENSE"],
        ],
        text=True,
        capture_output=True,
        encoding="utf-8",
    )

    assert process.returncode == 0, process.stderr


def test_install_solver():
    with pytest.raises(subprocess.CalledProcessError):
        _ = subprocess.run(
            [sys.executable, "-Bm", "gamspy", "install", "solver", "bla"],
            check=True,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )

    process = subprocess.run(
        [
            sys.executable,
            "-Bm",
            "gamspy",
            "install",
            "solver",
            "minos",
            "mosek",
        ],
        capture_output=True,
        encoding="utf-8",
        text=True,
    )
    assert process.returncode == 0, process.stdout + process.stderr

    with pytest.raises(subprocess.CalledProcessError):
        _ = subprocess.run(
            [sys.executable, "-Bm", "gamspy", "uninstall", "solver", "bla"],
            check=True,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )

    process = subprocess.run(
        [
            sys.executable,
            "-Bm",
            "gamspy",
            "install",
            "solver",
            "--install-all-solvers",
        ],
        capture_output=True,
        encoding="utf-8",
        text=True,
    )
    assert process.returncode == 0, process.stdout + process.stderr

    process = subprocess.run(
        [
            sys.executable,
            "-Bm",
            "gamspy",
            "uninstall",
            "solver",
            "minos",
            "mosek",
        ],
        capture_output=True,
        encoding="utf-8",
        text=True,
    )
    assert process.returncode == 0, process.stdout + process.stderr

    process = subprocess.run(
        [
            sys.executable,
            "-Bm",
            "gamspy",
            "uninstall",
            "solver",
            "--uninstall-all-solvers",
        ],
        capture_output=True,
        encoding="utf-8",
        text=True,
    )
    assert process.returncode == 0, process.stdout + process.stderr

    process = subprocess.run(
        [
            sys.executable,
            "-Bm",
            "gamspy",
            "install",
            "solver",
            "mpsge",
            "scip",
            "reshop",
        ],
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    )
    assert process.returncode == 0, process.stdout + process.stderr

    if platform.system() == "Linux":
        # use uv
        process = subprocess.run(
            [
                sys.executable,
                "-Bm",
                "gamspy",
                "install",
                "solver",
                "soplex",
                "--use-uv",
            ],
            capture_output=True,
            encoding="utf-8",
            text=True,
        )
        assert process.returncode == 0, process.stdout + process.stderr

        # use uv
        process = subprocess.run(
            [
                sys.executable,
                "-Bm",
                "gamspy",
                "uninstall",
                "solver",
                "soplex",
                "--use-uv",
            ],
            capture_output=True,
            encoding="utf-8",
            text=True,
        )
        assert process.returncode == 0, process.stdout + process.stderr


def test_list_solvers():
    process = subprocess.run(
        [sys.executable, "-Bm", "gamspy", "list", "solvers"],
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    )

    assert process.returncode == 0, process.stdout + process.stderr

    process = subprocess.run(
        [sys.executable, "-Bm", "gamspy", "list", "solvers", "-a"],
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    )

    assert process.returncode == 0, process.stdout + process.stderr


def test_show_license():
    process = subprocess.run(
        [sys.executable, "-Bm", "gamspy", "show", "license"],
        stderr=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        text=True,
    )

    assert process.returncode == 0, process.stdout + process.stderr
    assert isinstance(process.stdout, str)


def test_show_base():
    process = subprocess.run(
        [sys.executable, "-Bm", "gamspy", "show", "base"],
        stderr=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        text=True,
    )

    import gamspy_base

    assert process.returncode == 0, process.stdout + process.stderr
    assert gamspy_base.directory == process.stdout.strip()


def test_probe(teardown):
    node_info_path = os.path.join("tmp", "info.json")
    process = subprocess.run(
        [sys.executable, "-Bm", "gamspy", "probe", "-h"],
        capture_output=True,
        encoding="utf-8",
        text=True,
    )
    assert process.returncode == 0, process.stdout + process.stderr

    process = subprocess.run(
        [sys.executable, "-Bm", "gamspy", "probe", "-j", node_info_path],
        capture_output=True,
        encoding="utf-8",
        text=True,
    )
    assert process.returncode == 0, process.stdout + process.stderr

    process = subprocess.run(
        [
            sys.executable,
            "-Bm",
            "gamspy",
            "retrieve",
            "license",
            os.environ["LOCAL_LICENSE"],
            "-i",
            node_info_path,
            "-o",
            node_info_path[:-5] + ".txt",
        ],
        capture_output=True,
        encoding="utf-8",
        text=True,
    )

    assert process.returncode == 0, process.stdout + process.stderr
