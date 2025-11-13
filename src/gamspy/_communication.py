from __future__ import annotations

import os
import platform
import signal
import socket
import subprocess
import threading
import time
from typing import TYPE_CHECKING

from gamspy import utils
from gamspy.exceptions import FatalError, GamspyException, ValidationError

if TYPE_CHECKING:
    import io

    from gamspy import Container


_comm_pairs: dict[str, tuple[socket.socket, subprocess.Popen]] = {}


def open_connection(container: Container) -> None:
    LOOPBACK = "127.0.0.1"
    TIMEOUT = 30

    initial_pf_file = os.path.join(container._process_directory, "gamspy.pf")
    with open(initial_pf_file, "w") as file:
        file.write(
            'incrementalMode="2"\n'
            f'procdir="{container._process_directory}"\n'
            f'license="{container._license_path}"\n'
            f'curdir="{os.getcwd()}"\n'
        )

    command = [
        os.path.join(container.system_directory, "gams"),
        "GAMSPY_JOB",
        "pf",
        initial_pf_file,
    ]

    certificate_path = os.path.join(utils.DEFAULT_DIR, "gamspy_cert.crt")
    env = os.environ.copy()
    if os.path.isfile(certificate_path):
        env["GAMSLICECRT"] = certificate_path

    process = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        errors="replace",
        start_new_session=platform.system() != "Windows",
        env=env,
    )

    port_info = process.stdout.readline().strip()

    try:
        port = int(port_info.removeprefix("port: "))
    except ValueError as e:
        raise ValidationError(
            f"Error while reading the port! {port_info + process.stdout.read()}"
        ) from e

    def handler(signum, frame):
        if platform.system() != "Windows":
            os.kill(process.pid, signal.SIGINT)

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, handler)

    start = time.time()
    while True:
        if process.poll() is not None:  # pragma: no cover
            raise ValidationError(process.communicate()[0])

        try:
            new_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            new_socket.connect((LOOPBACK, port))
            break
        except (ConnectionRefusedError, OSError) as e:
            new_socket.close()
            end = time.time()

            if end - start > TIMEOUT:  # pragma: no cover
                raise FatalError(
                    f"Timeout while establishing the connection with socket. {process.communicate()[0]}"
                ) from e

    _comm_pairs[container._comm_pair_id] = (new_socket, process)


def get_connection(pair_id: str) -> tuple[socket.socket, subprocess.Popen]:
    return _comm_pairs[pair_id]


def close_connection(pair_id: str):
    try:
        _socket, process = get_connection(pair_id)
    except KeyError:
        # This means that the connection is already closed.
        return

    _socket.sendall(b"stop")
    _socket.close()

    if not process.stdout.closed:
        process.stdout.close()

    # Wait until the GAMS process dies.
    while process.poll() is None:
        ...

    del _comm_pairs[pair_id]


def _read_output(process: subprocess.Popen, output: io.TextIOWrapper | None) -> None:
    if output is not None:
        while True:
            data = process.stdout.readline()
            output.write(data)
            output.flush()
            if data.startswith("--- Job ") and "elapsed" in data:
                break


def check_response(response: bytes, job_name: str) -> None:
    GAMS_STATUS = {
        1: "Solver is to be called, the system should never return this number.",
        2: "There was a compilation error.",
        3: "There was an execution error.",
        4: "System limits were reached.",
        5: "There was a file error.",
        6: "There was a parameter error.",
        7: "The solve has failed due to a license error. The license you are using may impose model size limits (demo/community license) or you are using a GAMSPy incompatible professional license. Please contact sales@gams.com to find out about license options.",
        8: "There was a GAMS system error.",
        9: "GAMS could not be started.",
        10: "Out of memory.",
        11: "Out of disk.",
        109: "Could not create process/scratch directory.",
        110: "Too many process/scratch directories.",
        112: "Could not delete the process/scratch directory.",
        113: "Could not write the script gamsnext.",
        114: "Could not write the parameter file.",
        115: "Could not read environment variable.",
        400: "Could not spawn the GAMS language compiler (gamscmex).",
        401: "Current directory (curdir) does not exist.",
        402: "Cannot set current directory (curdir).",
        404: "Blank in system directory (UNIX only).",
        405: "Blank in current directory (UNIX only).",
        406: "Blank in scratch extension (scrext)",
        407: "Unexpected cmexRC.",
        408: "Could not find the process directory (procdir).",
        409: "CMEX library not be found (experimental).",
        410: "Entry point in CMEX library could not be found (experimental).",
        411: "Blank in process directory (UNIX only).",
        412: "Blank in scratch directory (UNIX only).",
        909: "Cannot add path / unknown UNIX environment / cannot set environment variable.",
        1000: "Driver error: incorrect command line parameters for gams.",
        2000: "Driver error: internal error: cannot install interrupt handler.",
        3000: "Driver error: problems getting current directory.",
        4000: "Driver error: internal error: GAMS compile and execute module not found.",
        5000: "Driver error: internal error: cannot load option handling library.",
    }

    value = response[: response.find(b"#")].decode("ascii")
    if not value:
        raise FatalError(
            "Error while getting the return code from GAMS backend. This means that GAMS is in a bad state. Try to backtrack for previous errors."
        )

    return_code = int(value)

    if return_code in GAMS_STATUS:
        try:
            info = GAMS_STATUS[return_code]
        except IndexError:  # pragma: no cover
            info = ""

        raise GamspyException(
            f"Return code {return_code}. {info} Check {job_name + '.lst'} for more information.",
            return_code,
        )


def send_job(
    comm_pair_id: str,
    job_name: str,
    pf_file: str,
    output: io.TextIOWrapper | None = None,
):
    _socket, process = get_connection(comm_pair_id)
    try:
        # Send pf file
        _socket.sendall(pf_file.encode("utf-8"))

        # Read output
        _read_output(process, output)

        # Receive response
        response = _socket.recv(256)
        check_response(response, job_name)
    except ConnectionError as e:
        raise FatalError(
            f"There was an error while communicating with GAMS server: {e}",
        ) from e
