from __future__ import annotations

import os
import platform
import select
import shutil
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlencode

import pytest
import urllib3

import gamspy.utils as utils
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
            os.environ["CHECKOUTABLE_NETWORK_LICENSE"],
            "-i",
            node_info_path,
            "-o",
            node_info_path[:-5] + ".txt",
            "-c",
            "1",
        ],
        capture_output=True,
        encoding="utf-8",
        text=True,
    )

    assert process.returncode == 0, process.stdout + process.stderr


DUMMY_GDX = os.path.join(os.path.dirname(__file__), "file1.gdx")
DUMMY_GDX2 = os.path.join(os.path.dirname(__file__), "file2.gdx")


def run_cli(args):
    return subprocess.run(
        [sys.executable, "-Bm", "gamspy", "gdx"] + args,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--output", "out.txt"),
        ("--symb", "a"),
        ("--ueltable", "uelVar"),
        ("--delim", "comma"),
        ("--decimalsep", "period"),
        ("--dformat", "hexponential"),
        ("--cdim", "Y"),
        ("--filterdef", "N"),
        ("--epsout", "EPSVAL"),
        ("--naout", "NA_VAL"),
        ("--pinfout", "PINFINITY"),
        ("--minfout", "-INFINITY"),
        ("--undfout", "UNDEFINED"),
        ("--zeroout", "0.000"),
        ("--header", "My Header"),
    ],
)
def test_gdx_dump_with_value_flags(flag, value):
    result = run_cli(["dump", DUMMY_GDX, flag, value])
    assert result.returncode == 0, f"Failed: {flag} = {value}"


@pytest.mark.parametrize(
    "flag",
    [
        "--version",
        "--noheader",
        "--nodata",
        "--csvallfields",
        "--csvsettext",
        "--symbols",
        "--domaininfo",
        "--symbolsasset",
        "--symbolsassetdi",
        "--settext",
    ],
)
def test_gdx_dump_with_boolean_flags(flag):
    result = run_cli(["dump", DUMMY_GDX, flag])
    assert result.returncode == 0, f"Failed: {flag}"


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--eps", "1e-6"),
        ("--releps", "1e-5"),
        ("--field", "L"),
        ("--setdesc", "N"),
    ],
)
def test_gdx_diff_with_value_flags(flag, value):
    result = run_cli(["diff", DUMMY_GDX, DUMMY_GDX2, flag, value])
    assert result.returncode == 0, f"Failed: {flag} = {value}"


@pytest.mark.parametrize(
    "flag",
    [
        "--cmpdefaults",
        "--cmpdomains",
        "--matrixfile",
        "--ignoreorder",
    ],
)
def test_gdx_diff_with_boolean_flags(flag):
    result = run_cli(["diff", DUMMY_GDX, DUMMY_GDX2, flag])
    assert result.returncode == 0, f"Failed: {flag}"


def test_gdx_diff_with_ids():
    result = run_cli(["diff", DUMMY_GDX, DUMMY_GDX2, "--id", "a", "--id", "b"])
    assert result.returncode == 0


def test_gdx_diff_with_skipids():
    result = run_cli(
        ["diff", DUMMY_GDX, DUMMY_GDX2, "--skipid", "a", "--skipid", "b"]
    )
    assert result.returncode == 0


class TunnelingProxy(BaseHTTPRequestHandler):
    # Class attribute to store the paths of CONNECT requests.
    # This will be used by the test to verify the proxy was used.
    handled_connect_requests: list[str] = []

    def do_CONNECT(self):
        """Handle CONNECT requests to set up the tunnel."""
        # Record the target host and port.
        self.handled_connect_requests.append(self.path)

        host, port_str = self.path.split(":")
        port = int(port_str)

        try:
            # Establish connection to the upstream server.
            upstream = socket.create_connection((host, port))
            self.send_response(200, "Connection established")
            self.end_headers()
        except OSError:
            self.send_error(502)
            return

        # Tunnel data between the client and the upstream server.
        sockets = [self.connection, upstream]
        TIMEOUT = 10
        while True:
            # Wait until a socket is ready for reading.
            read_list, _, _ = select.select(sockets, [], [], TIMEOUT)
            if not read_list:  # Timeout occurred
                break

            # if client-proxy socket is in the list, receive the data and send it to the upstream.
            if self.connection in read_list:
                data = self.connection.recv(8192)
                if not data:
                    break
                upstream.sendall(data)

            # if proxy-upstream socket is in the list, receive the data and send it to the client.
            if upstream in read_list:
                data = upstream.recv(8192)
                if not data:
                    break
                self.connection.sendall(data)

        upstream.close()

    def do_GET(self):
        self.send_response(200)
        self.end_headers()


@pytest.fixture
def proxy_server():
    """Starts the proxy server in a separate thread."""
    # Reset the log of handled requests before each test run.
    TunnelingProxy.handled_connect_requests = []

    server = HTTPServer(("127.0.0.1", 0), TunnelingProxy)
    port = server.server_address[1]

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    # Yield both the proxy URL and the handler class for inspection.
    yield f"http://127.0.0.1:{port}", TunnelingProxy

    server.shutdown()
    thread.join()


@pytest.mark.skipif(
    sys.version_info.minor != 13,
    reason="One Python version is enough to test it.",
)
def test_https_proxy(proxy_server):
    proxy_url, proxy_handler = proxy_server
    encoded_args = urlencode({"access_token": os.environ["LOCAL_LICENSE"]})

    os.environ["HTTPS_PROXY"] = proxy_url
    http = utils._make_http()
    assert isinstance(http, urllib3.ProxyManager)

    # Make a request to the target server, which should go through the proxy.
    try:
        request = http.request(
            "GET", f"https://license.gams.com/license-type?{encoded_args}"
        )
        assert request.status == 200
    finally:
        del os.environ["HTTPS_PROXY"]

    # Assert that the proxy handled exactly one CONNECT request.
    assert len(proxy_handler.handled_connect_requests) == 1
    # Assert that the request was for the correct destination.
    assert proxy_handler.handled_connect_requests[0] == "license.gams.com:443"
