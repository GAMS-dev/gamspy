from __future__ import annotations

import os
import platform
import signal
import socket
import subprocess
import threading
import time
import weakref
from typing import TYPE_CHECKING, TextIO

import certifi

from gamspy import utils
from gamspy._execution.base import ExecutionEngine
from gamspy.exceptions import FatalError, GamspyException, ValidationError

if TYPE_CHECKING:
    from gamspy import Container, Model, Options


# Engines with a live GAMS process.
_running_engines: set[SocketEngine] = set()


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


def _read_output(process: subprocess.Popen, output: TextIO | None) -> None:
    if output is not None:
        while True:
            data = process.stdout.readline()  # ty: ignore[unresolved-attribute]
            output.write(data)
            output.flush()
            if data.startswith("--- Job ") and "elapsed" in data:
                break


def check_response(response: bytes, job_name: str) -> None:
    value = response[: response.find(b"#")].decode("ascii")
    if not value:  # pragma: no cover
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


class SocketEngine(ExecutionEngine):
    """
    Runs jobs on a GAMS process in incremental mode. Each job is a pf file
    that is sent to the process over a local socket.
    """

    def __init__(self, container: Container) -> None:
        # A weak reference so that the finalizer of the container, which keeps
        # the engine alive, does not keep the container alive as well.
        self._container_ref = weakref.ref(container)
        self._socket: socket.socket | None = None
        self._process: subprocess.Popen | None = None

    def __deepcopy__(self, memo: dict) -> SocketEngine:
        # A copied container shares the GAMS process of the original one.
        return self

    @property
    def _container(self) -> Container:
        container = self._container_ref()
        if container is None:  # pragma: no cover
            raise ValidationError("The container of the execution engine is gone.")

        return container

    @property
    def is_running(self) -> bool:
        return self._process is not None

    def start(self) -> None:
        LOOPBACK = "127.0.0.1"
        TIMEOUT = 30

        container = self._container
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

        if container._options.monitor_process_tree_memory:
            command.append("ProcTreeMemMonitor=1")

        certificate_path = os.path.join(utils.DEFAULT_DIR, "gamspy_cert.crt")
        env = os.environ.copy()
        if os.path.isfile(certificate_path):
            env["GAMSLICECRT"] = certificate_path

        if "CURL_CA_BUNDLE" not in env:
            env["CURL_CA_BUNDLE"] = certifi.where()

        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            errors="replace",
            start_new_session=platform.system() != "Windows",
            env=env,
        )

        port_info = process.stdout.readline().strip()  # ty: ignore[unresolved-attribute]

        try:
            port = int(port_info.removeprefix("port: "))
        except ValueError as e:  # pragma: no cover
            raise ValidationError(
                f"Error while reading the port! {port_info + process.stdout.read()}"  # ty: ignore[unresolved-attribute]
            ) from e

        def handler(signum, frame):  # pragma: no cover
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
            except (ConnectionRefusedError, OSError) as e:  # pragma: no cover
                new_socket.close()
                end = time.time()

                if end - start > TIMEOUT:
                    raise FatalError(
                        f"Timeout while establishing the connection with socket. {process.communicate()[0]}"
                    ) from e

        self._socket = new_socket
        self._process = process
        _running_engines.add(self)

    def _prepare_hidden_options(self, job_name: str, has_model: bool) -> dict:
        container = self._container
        scrdir = container._process_directory

        hidden_options = {
            "input": job_name + ".gms",
            "output": job_name + ".lst",
            "optdir": container.working_directory,
            "sysdir": container.system_directory,
            "scrdir": scrdir,
            "scriptnext": os.path.join(scrdir, "gamsnext.sh"),
            "license": container._license_path,
        }

        if has_model:
            hidden_options["gdx"] = container._gdx_out
            hidden_options["gdxSymbols"] = "newOrChangedNoData"

        if container._network_license:
            hidden_options["netlicense"] = os.path.join(scrdir, "gamslice.dat")

        if container._restart_from is not None:
            hidden_options["restart"] = container._restart_from

        return hidden_options

    def execute(
        self,
        gams_code: str,
        options: Options,
        job_name: str,
        model: Model | None,
        output: TextIO | None,
    ) -> None:
        container = self._container

        # Write gms file
        with open(job_name + ".gms", "w", encoding="utf-8") as gams_file:
            gams_file.write(gams_code)

        # A hibernating container has no engine running, so start one.
        is_waking_up = container._restart_from is not None
        if is_waking_up and not self.is_running:
            self.start()

        # Write pf file
        pf_file = job_name + ".pf"
        hidden_options = self._prepare_hidden_options(job_name, model is not None)
        options._set_hidden_options(hidden_options)
        options._export(pf_file, output)

        self.send_job(job_name, pf_file, output)

        if is_waking_up:
            container._restart_from = None

    def send_job(
        self, job_name: str, pf_file: str, output: TextIO | None = None
    ) -> None:
        if self._socket is None or self._process is None:
            raise ValidationError(
                "The connection to the GAMS execution engine is closed. After "
                "`Container.close()`, only frozen models can be solved and only "
                "records that are already in Python can be read."
            )

        try:
            # Send pf file
            self._socket.sendall(pf_file.encode("utf-8"))

            # Read output
            _read_output(self._process, output)

            # Receive response
            response = self._socket.recv(256)
            check_response(response, job_name)
        except ConnectionError as e:  # pragma: no cover
            raise FatalError(
                f"There was an error while communicating with GAMS server: {e}",
            ) from e

    def interrupt(self) -> None:
        if self._process is None:
            return

        if platform.system() == "Windows":
            os.kill(self._process.pid, signal.CTRL_C_EVENT)  # ty: ignore[unresolved-attribute]
        else:
            os.kill(self._process.pid, signal.SIGINT)

    def stop(self) -> None:  # pragma: no cover
        if self._socket is None or self._process is None:
            return

        self._socket.sendall(b"stop")
        self._socket.close()

        stdout_data, stderr_data = self._process.communicate()

        if self._process.returncode != 0 and stderr_data:
            print(stderr_data, end="")

        if self._process.returncode == 0 and stdout_data:
            print(stdout_data, end="")

        self._socket = None
        self._process = None
        _running_engines.discard(self)
