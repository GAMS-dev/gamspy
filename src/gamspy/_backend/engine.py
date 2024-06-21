from __future__ import annotations

import copy
import io
import json
import logging
import os
import tempfile
import time
import urllib.parse
import uuid
import zipfile
from typing import TYPE_CHECKING

import certifi
import urllib3
from gams import GamsEngineConfiguration
from gams.control.workspace import GamsException
from gams.core.cfg import cfgModelTypeName
from gams.core.gmo import gmoProc_nrofmodeltypes
from gams.core.opt import optSetStrStr

import gamspy._backend.backend as backend
from gamspy.exceptions import (
    EngineClientException,
    EngineException,
    GamspyException,
    ValidationError,
)

if TYPE_CHECKING:
    from gamspy import Container, Model, Options


logger = logging.getLogger("ENGINE")
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(name)s - %(levelname)s] %(message)s")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


MAX_REQUEST_ATTEMPS = 3
ZIP_NAME = "data.zip"
INEX_FILE_NAME = "inex.json"
STATUS_MAP = {
    -10: "waiting",
    -3: "cancelled",
    -2: "cancelling",
    -1: "corrupted",
    0: "queued",
    1: "running",
    2: "outputting",
    10: "finished",
}


def get_relative_paths(paths: list[str], start: str) -> list[str]:
    relative_paths = []
    for path in paths:
        relative_path = os.path.relpath(path, start)
        if relative_path.startswith(f"..{os.sep}"):
            raise ValidationError(
                "Extra model file path must be relative to the working"
                f" directory. The given path: {path}, the working"
                f" directory: {start}, the"
                f" relative path: {relative_path}"
            )

        relative_paths.append(relative_path)

    return relative_paths


SCOPES = [
    "READONLY",
    "NAMESPACES",
    "JOBS",
    "USERS",
    "HYPERCUBE",
    "CLEANUP",
    "LICENSES",
    "USAGE",
    "AUTH",
    "CONFIGURATION",
]


class Endpoint:
    def get_request_headers(self):
        return {
            "Authorization": self.client._engine_config._get_auth_header(),
            "User-Agent": "GAMSPy EngineClient",
            "Accept": "application/json",
        }


class Auth(Endpoint):
    def __init__(self, client: EngineClient) -> None:
        self.client = client
        self._http = client._http
        self.extra_model_files = client.extra_model_files
        self.engine_options = client.engine_options

    def post(
        self,
        expires_in: int = 14400,
        scope: list[str] | None = None,
    ) -> str:
        """
        Creates a JSON Web Token(JWT) for authentication

        Parameters
        ----------
        expires_in : int, optional
            Expiration time, by default 14400
        scope : list[str] | None, optional
            Scope of the token, by default None

        Returns
        -------
        str
            token

        Raises
        ------
        EngineClientException
            In case bad request
        EngineClientException
            In case unauthorized request
        EngineClientException
            In case there is an internal error
        GamspyException
            In case the status code is unrecognized
        """
        info = {"expires_in": str(expires_in)}

        if isinstance(scope, list):
            for elem in scope:
                if elem not in SCOPES:
                    raise ValidationError(f"{elem} is not a valid scope")

            scope_info = " ".join(scope)
            info.update({"scope": scope_info})

        r = self._http.request(
            "POST",
            self.client._engine_config.host
            + "/auth/?"
            + urllib.parse.urlencode(info, doseq=True),
            headers=self.get_request_headers(),
        )

        response_data = r.data.decode("utf-8", errors="replace")
        info = json.loads(response_data)

        if r.status == 200:
            return info["token"]
        elif r.status == 400:
            raise EngineClientException(f"Bad request: {info['message']}")
        elif r.status == 401:
            raise EngineClientException(f"Unauthorized: {info['message']}")
        elif r.status == 500:
            raise EngineClientException(f"Internal error: {info['message']}")
        else:
            raise GamspyException(f"Unrecognized status code {r.status}")

    def login(
        self, expires_in: int = 14400, scope: list[str] | None = None
    ) -> str:
        """
        Creates a JSON Web Token(JWT) for authentication (username and password in request body)

        Parameters
        ----------
        expires_in : int, optional
            Expiration time for the token, by default 14400
        scope: list[str], optional
            Scope of the token, by default None

        Returns
        -------
        str
        """
        info = {
            "username": self.client._engine_config.username,
            "password": self.client._engine_config.password,
            "expires_in": expires_in,
        }

        if isinstance(scope, list):
            for elem in scope:
                if elem not in SCOPES:
                    raise ValidationError(f"{elem} is not a valid scope")

            scope_info = " ".join(scope)
            info.update({"scope": scope_info})

        r = self._http.request(
            "POST",
            self.client._engine_config.host + "/auth/login",
            fields=info,
        )

        response_data = r.data.decode("utf-8", errors="replace")
        info = json.loads(response_data)

        if r.status == 200:
            return info["token"]
        elif r.status == 400:
            raise EngineClientException(f"Bad request: {info['message']}")
        elif r.status == 401:
            raise EngineClientException(f"Unauthorized: {info['message']}")
        elif r.status == 500:
            raise EngineClientException(f"Internal error: {info['message']}")
        else:
            raise GamspyException(f"Unrecognized status code {r.status}")

    def logout(self) -> str:
        """
        Invalidates all of your JSON Web Tokens(JWTs)

        Returns
        -------
        str
            message
        """
        r = self._http.request(
            "POST",
            self.client._engine_config.host + "/auth/logout",
            headers=self.get_request_headers(),
        )

        if r.status == 200:
            response_data = r.data.decode("utf-8", errors="replace")
            info = json.loads(response_data)
            return info["message"]
        elif r.status == 400:
            raise EngineClientException("Bad request!")
        elif r.status == 401:
            raise EngineClientException("Unauthorized!")
        elif r.status == 500:
            raise EngineClientException("Internal error!")
        else:
            raise GamspyException(f"Unrecognized status code {r.status}")


class Job(Endpoint):
    def __init__(self, client: EngineClient) -> None:
        self.client = client
        self._http = client._http
        self.extra_model_files = client.extra_model_files
        self.engine_options = client.engine_options

    def get(self, token: str) -> tuple[int, str, int | None]:
        """
        Get request to /jobs/{token} which returns the details of a job.
        Refer to https://engine.gams.com/api/ for more details.

        Parameters
        ----------
        token : str
            Job token

        Returns
        -------
        tuple[int, str, int]
            Job status, job status message, and gams exit code

        Raises
        ------
        EngineClientException
            If get request has failed.
        """
        for attempt_number in range(MAX_REQUEST_ATTEMPS):
            r = self._http.request(
                "GET",
                self.client._engine_config.host + f"/jobs/{token}",
                headers=self.get_request_headers(),
            )
            response_data = r.data.decode("utf-8", errors="replace")

            if r.status == 200:
                info = json.loads(response_data)
                job_status = int(info["status"])
                return (
                    job_status,
                    STATUS_MAP[job_status],
                    info["process_status"],
                )
            elif r.status == 429:
                time.sleep(2**attempt_number)  # retry with exponential backoff
                continue

            raise EngineClientException(
                "Creating job on GAMS Engine failed with status code: "
                + str(r.status)
                + ". Message: "
                + response_data,
                r.status,
            )

        raise EngineClientException(
            "Creating job on GAMS Engine failed after: "
            + str(MAX_REQUEST_ATTEMPS)
            + " attempts. Message: "
            + response_data,
            r.status,
        )

    def post(
        self,
        working_directory: str,
        gms_file: str,
        pf_file: str | None = None,
    ) -> str:
        """
        Post request to /jobs which submits a new job to be solved.
        Refer to https://engine.gams.com/api/ for more details.

        Parameters
        ----------
        working_directory : str
            Working directory
        gms_file : str
            Name of the gms file
        pf_file: str | None
            Name of the pf file

        Returns
        -------
        str
            Token

        Raises
        ------
        EngineClientException
            If post request has failed.
        """
        model_data_zip = self._create_zip_file(
            working_directory, gms_file, pf_file
        )
        gms_file = os.path.relpath(gms_file, working_directory)
        pf_file = (
            os.path.relpath(pf_file, working_directory)
            if pf_file is not None
            else None
        )
        query_params, file_params = self._get_params(
            model_data_zip, gms_file, pf_file
        )

        for attempt_number in range(MAX_REQUEST_ATTEMPS):
            r = self._http.request(
                "POST",
                self.client._engine_config.host
                + "/jobs/?"
                + urllib.parse.urlencode(query_params, doseq=True),
                fields=file_params,
                headers=self.get_request_headers(),
            )
            response_data = r.data.decode("utf-8", errors="replace")
            if r.status == 201:
                return json.loads(response_data)["token"]
            elif r.status == 429:
                time.sleep(2**attempt_number)  # retry with exponential backoff
                continue

            raise EngineClientException(
                "Creating job on GAMS Engine failed with status code: "
                + str(r.status)
                + ". Message: "
                + response_data,
                r.status,
            )

        raise EngineClientException(
            "Creating job on GAMS Engine failed after: "
            + str(MAX_REQUEST_ATTEMPS)
            + " attempts. Message: "
            + response_data,
            r.status,
        )

    def get_results(self, token: str, working_directory: str):
        """
        Get request to /jobs/{token}/result which downloads the job results.
        Downloaded results are unpacked to working directory.
        Refer to https://engine.gams.com/api/ for more details.

        Parameters
        ----------
        token : str
            Job token
        working_directory : str
            Working directory

        Raises
        ------
        EngineClientException
            If get request has failed.
        """
        if not os.path.exists(working_directory):
            os.makedirs(working_directory, exist_ok=True)

        r = self._http.request(
            "GET",
            self.client._engine_config.host + f"/jobs/{token}/result",
            headers=self.get_request_headers(),
            preload_content=False,
        )

        if r.status == 200:
            fd, path = tempfile.mkstemp()

            try:
                with open(path, "wb") as out:
                    while True:
                        data = r.read(6000)
                        if not data:
                            break
                        out.write(data)

                r.release_conn()

                with zipfile.ZipFile(path, "r") as zip_ref:
                    zip_ref.extractall(working_directory)
            finally:
                os.close(fd)
                os.remove(path)
        else:
            response_data = r.data.decode("utf-8", errors="replace")
            raise EngineClientException(
                "Fatal error while getting the results back from engine. GAMS"
                f" Engine return code: {r.status}. Error message:"
                f" {response_data}",
                r.status,
            )

    def delete_results(self, token: str):
        """
        Delete request to /jobs/{token} which deletes the job results.
        Refer to https://engine.gams.com/api/ for more details.

        Parameters
        ----------
        token : str
            Job token

        Raises
        ------
        EngineClientException
            If job data does not exist in GAMS Engine.
        EngineClientException
            If delete request has failed.
        """
        for attempt_number in range(MAX_REQUEST_ATTEMPS):
            r = self._http.request(
                "DELETE",
                self.client._engine_config.host + "/jobs/" + token + "/result",
                headers=self.get_request_headers(),
            )
            response_data = r.data.decode("utf-8", errors="replace")

            if r.status == 200:
                return
            elif r.status == 403:
                raise EngineClientException(
                    "Job data does not exist in GAMS Engine!", r.status
                )
            elif r.status == 429:
                time.sleep(2**attempt_number)  # retry with exponential backoff
                continue

            raise EngineClientException(
                "Removing job result failed with status code: "
                + str(r.status)
                + ". Message: "
                + response_data,
                r.status,
            )

        raise EngineClientException(
            "Removing job result failed after: "
            + str(MAX_REQUEST_ATTEMPS)
            + " attempts. Message: "
            + response_data,
            r.status,
        )

    def get_logs(self, token: str) -> tuple[str, bool]:
        """
        Get request to /jobs/{token}/unread-logs which returns stdout of a job.
        Refer to https://engine.gams.com/api/ for more details.

        Parameters
        ----------
        token : str
            Job token

        Returns
        -------
        tuple[str, bool]
            Current output buffer and queue finished status

        Raises
        ------
        EngineClientException
            If get request has failed.
        """
        for attempt_number in range(MAX_REQUEST_ATTEMPS):
            r = self._http.request(
                "DELETE",
                self.client._engine_config.host + f"/jobs/{token}/unread-logs",
                headers=self.get_request_headers(),
            )
            response_data = r.data.decode("utf-8", errors="replace")
            response_data = json.loads(response_data)

            if r.status == 429:
                time.sleep(2**attempt_number)  # retry with exponential backoff
                continue
            elif r.status != 200:
                raise EngineClientException(
                    "Getting logs failed with status code: "
                    + str(r.status)
                    + ". "
                    + response_data["message"]
                    + ".",
                    r.status,
                )
            stdout_data = response_data["message"]
            break

        return stdout_data, response_data["queue_finished"]

    def _create_zip_file(
        self,
        working_directory: str,
        gms_file: str,
        pf_file: str | None,
    ) -> io.BytesIO:
        model_data_zip = io.BytesIO()
        model_files = [gms_file]
        if pf_file is not None:
            model_files.append(pf_file)

        model_files += self.extra_model_files
        model_files = get_relative_paths(model_files, working_directory)

        with zipfile.ZipFile(
            model_data_zip, "w", zipfile.ZIP_DEFLATED
        ) as model_data:
            for model_file in model_files:
                model_data.write(
                    os.path.join(working_directory, model_file),
                    arcname=model_file,
                )

        model_data_zip.seek(0)

        return model_data_zip

    def _get_params(self, model_data_zip, gms_file, pf_file: str | None):
        file_params = {}
        query_params = (
            copy.deepcopy(self.engine_options) if self.engine_options else {}
        )

        query_params["namespace"] = self.client._engine_config.namespace

        if "data" in query_params or "model_data" in query_params:
            raise ValidationError(
                "`engine_options` must not include keys `data` or "
                "`model_data`. Please use `extra_model_files` to "
                "provide additional files to send to GAMS Engine.",
            )

        if "inex_file" in query_params:
            if isinstance(query_params["inex_file"], io.IOBase):
                file_params["inex_file"] = (
                    INEX_FILE_NAME,
                    query_params["inex_file"].read(),
                    "application/json",
                )
            else:
                with open(query_params["inex_file"], "rb") as f:
                    file_params["inex_file"] = (
                        INEX_FILE_NAME,
                        f.read(),
                        "application/json",
                    )
            del query_params["inex_file"]

        if "model" in query_params:
            file_params["data"] = (
                ZIP_NAME,
                model_data_zip.read(),
                "application/zip",
            )
        else:
            query_params["run"] = gms_file
            query_params["model"] = os.path.splitext(gms_file)[0]
            file_params["model_data"] = (
                ZIP_NAME,
                model_data_zip.read(),
                "application/zip",
            )

        model_data_zip.close()

        if "arguments" in query_params:
            if not isinstance(query_params["arguments"], list):
                query_params["arguments"] = [query_params["arguments"]]

            if pf_file is not None:
                query_params["arguments"].append(f"pf={pf_file}")
        else:
            if pf_file is not None:
                query_params["arguments"] = [f"pf={pf_file}"]

        return query_params, file_params


class EngineClient:
    def __init__(
        self,
        host: str,
        username: str | None = None,
        password: str | None = None,
        jwt: str | None = None,
        namespace: str = "global",
        extra_model_files: list[str] = [],
        engine_options: dict | None = None,
        remove_results: bool = False,
        is_blocking: bool = True,
    ):
        self.host = host
        self.username = username
        self.password = password
        self.jwt = jwt
        self.namespace = namespace
        self.extra_model_files = extra_model_files
        self.engine_options = engine_options
        self.remove_results = remove_results
        self.is_blocking = is_blocking
        self.tokens: list[tuple] = []

        self._http = urllib3.PoolManager(
            cert_reqs="CERT_REQUIRED", ca_certs=certifi.where()
        )

        self._engine_config = self._get_engine_config()

        # Endpoints
        self.job = Job(self)

        self.auth = Auth(self)

    def _get_engine_config(self):
        try:
            return GamsEngineConfiguration(
                self.host,
                self.username,
                self.password,
                self.jwt,
                self.namespace,
            )

        except GamsException as e:
            raise ValidationError(e) from e


class GAMSEngine(backend.Backend):
    def __init__(
        self,
        container: Container,
        client: EngineClient | None,
        options: Options,
        output: io.TextIOWrapper | None = None,
        model: Model | None = None,
    ) -> None:
        if client is None:
            raise ValidationError(
                "`engine_client` must be provided to solve on GAMS Engine"
            )

        super().__init__(
            container,
            os.path.basename(container._gdx_in),
            os.path.basename(container._gdx_out),
        )

        self.client = client
        self.options = options._get_gams_options(
            self.container.workspace, output
        )
        self.options.trace = "trace.txt"
        self.output = output
        self.model = model
        self.job_name = f"_job_{uuid.uuid4()}"
        self.gms_file = self.job_name + ".gms"
        self.pf_file = self.job_name + ".pf"

    def is_async(self):
        return not self.client.is_blocking

    def preprocess(self, keep_flags: bool = False):
        gams_string, dirty_names = super().preprocess(keep_flags)

        # Set selected solvers
        for i in range(1, gmoProc_nrofmodeltypes):
            optSetStrStr(
                self.options._opt,
                cfgModelTypeName(self.options._cfg, i),
                self.options._selected_solvers[i],
            )

        # Set save file path
        self.options._save = self.container._save_to.name

        # Set restart file path
        self.options._restart = self.container._restart_from.name

        # Set input file path
        self.options._input = self.job_name + ".gms"

        # Set output file path
        if not self.options.output:
            self.options.output = self.job_name + ".lst"

        # Export pf file
        self.options.export(self.pf_file)

        # Export gms file
        gms_path = os.path.join(
            self.container.working_directory, self.gms_file
        )
        with open(gms_path, "w", encoding="utf-8") as file:
            file.write(gams_string)

        return dirty_names

    def solve(self, is_implicit: bool = False, keep_flags: bool = False):
        dirty_names = self.preprocess(keep_flags)

        self.run()

        if self.is_async():
            return None

        # Synchronize GAMSPy with checkpoint and return a summary
        summary = self.postprocess(dirty_names, is_implicit)

        return summary

    def run(self):
        try:
            original_extra_files = copy.deepcopy(
                self.client.job.extra_model_files
            )
            self.client.job.extra_model_files = self._append_gamspy_files()
            token = self.client.job.post(
                self.container.working_directory,
                os.path.join(self.container.working_directory, self.gms_file),
                os.path.join(self.container.working_directory, self.pf_file),
            )
            self.client.job.extra_model_files = original_extra_files
            self.client.tokens.append(token)

            if not self.is_async():
                job_status, message, _ = self.client.job.get(token)

                if job_status not in STATUS_MAP:
                    raise EngineException(
                        "Unknown job status code! Currently supported job"
                        f" status codes: {STATUS_MAP.keys()}",
                        status_code=job_status,
                    )

                if job_status in [-1, -3]:
                    raise EngineException(
                        "Could not get job results because the job is"
                        f" {message}.",
                        status_code=job_status,
                    )

                while job_status in [-10, -2, 0]:
                    logger.info(f"Job status is {message}...")
                    job_status = self.client.job.get(token)

                while job_status in [1, 2]:
                    message, is_finished = self.client.job.get_logs(token)

                    if self.output is not None:
                        self.output.write(message)

                    if is_finished:
                        job_status = 10

                self.client.job.get_results(
                    token, self.container.working_directory
                )

                self.model._update_model_attributes()
        finally:
            self.container._unsaved_statements = []
            self.container._delete_autogenerated_symbols()

    def postprocess(self, dirty_names: list[str], is_implicit: bool = False):
        symbols = dirty_names + self.container._import_symbols

        if len(symbols) != 0:
            self.container._load_records_from_gdx(
                self.container._gdx_out, symbols
            )

        self.container._swap_checkpoints()

        if self.client.remove_results or is_implicit:
            return None

        return self.prepare_summary(
            self.container.working_directory, self.options.trace
        )

    def _append_gamspy_files(self) -> list[str]:
        extra_model_files = self.client.job.extra_model_files + [
            self.container._gdx_in,
            self.container._restart_from._checkpoint_file_name,
        ]

        return extra_model_files
