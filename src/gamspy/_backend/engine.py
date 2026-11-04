from __future__ import annotations

import base64
import copy
import io
import json
import logging
import os
import shutil
import tempfile
import time
import urllib.parse
import zipfile
from typing import TYPE_CHECKING, TextIO

import gamspy._backend.backend as backend
import gamspy._gdx as gdxio
from gamspy._execution import SocketEngine
from gamspy._execution.socket_engine import GAMS_STATUS
from gamspy._internals import DataSource
from gamspy._options import Options
from gamspy.exceptions import (
    EngineClientException,
    EngineException,
    GamspyException,
    ValidationError,
    _customize_exception,
)

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    import pandas as pd
    import requests

    from gamspy import Container, Model


logger = logging.getLogger("ENGINE")
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(name)s - %(levelname)s] %(message)s")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


MAX_REQUEST_ATTEMPS = 3
# (connect, read) timeouts in seconds for each request to GAMS Engine
REQUEST_TIMEOUT = (30, 300)
JOB_POLL_INTERVAL = 0.5
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


class EngineConfiguration:
    """Configuration that allows the execution of jobs on a specific GAMS Engine instance"""

    def __init__(
        self, host: str, username: str, password: str, jwt: str, namespace: str
    ) -> None:
        self.jwt = jwt
        self.username = username
        self.password = password
        self.namespace = namespace
        self.host = self._verify_host(host)

    def _verify_host(self, host: str) -> str:
        validated_url = urllib.parse.urlparse(host)
        if validated_url.scheme not in ("http", "https"):
            raise ValidationError(
                "Invalid GAMS Engine host. Only HTTP and HTTPS protocols supported"
            )
        if validated_url.netloc == "":
            raise ValidationError(
                "Invalid GAMS Engine host. Make sure you provide a valid URL."
            )
        host = host.rstrip("/")
        if not host.endswith("/api"):
            host += "/api"

        return host

    def _get_auth_header(self) -> str:
        """Returns authentication header"""
        if not self.username:
            if not self.jwt:
                raise ValidationError(
                    "Neither username/password nor JWT token provided for authentication with the GAMS Engine instance."
                )
            return "Bearer " + self.jwt

        return "Basic " + base64.b64encode(
            (self.username + ":" + self.password).encode("utf-8")
        ).decode("utf-8")


class Endpoint: ...


class Auth(Endpoint):
    def __init__(self, client: EngineClient) -> None:
        self.client = client
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
        import requests

        params = {"expires_in": str(expires_in)}

        if isinstance(scope, list):
            for elem in scope:
                if elem not in SCOPES:
                    raise ValidationError(f"{elem} is not a valid scope")

            scope_info = " ".join(scope)
            params.update({"scope": scope_info})

        for attempt_number in range(MAX_REQUEST_ATTEMPS):
            r = requests.post(
                f"{self.client._engine_config.host}/auth",
                params=params,
                headers=self.client.get_request_headers(),
                timeout=REQUEST_TIMEOUT,
            )

            response_data = r.content.decode("utf-8", errors="replace")

            if r.status_code in (200, 400, 401, 500):
                info = json.loads(response_data)

            if r.status_code == 200:
                return info["token"]
            elif r.status_code == 400:
                raise EngineClientException(f"Bad request: {info['message']}")
            elif r.status_code == 401:
                raise EngineClientException("Unauthorized!")
            elif r.status_code == 429:
                time.sleep(2**attempt_number)  # retry with exponential backoff
                continue
            elif r.status_code == 500:
                raise EngineClientException(f"Internal error: {info['message']}")

        raise GamspyException(f"Unrecognized status code {r.status_code}")

    def login(self, expires_in: int = 14400, scope: list[str] | None = None) -> str:
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
        import requests

        payload = {
            "username": self.client._engine_config.username,
            "password": self.client._engine_config.password,
            "expires_in": expires_in,
        }

        if isinstance(scope, list):
            for elem in scope:
                if elem not in SCOPES:
                    raise ValidationError(f"{elem} is not a valid scope")

            scope_info = " ".join(scope)
            payload.update({"scope": scope_info})

        for attempt_number in range(MAX_REQUEST_ATTEMPS):
            r = requests.post(
                f"{self.client._engine_config.host}/auth/login",
                data=payload,
                timeout=REQUEST_TIMEOUT,
            )

            response_data = r.content.decode("utf-8", errors="replace")
            if r.status_code in (200, 400, 401, 500):
                info = json.loads(response_data)

            if r.status_code == 200:
                return info["token"]
            elif r.status_code == 400:
                raise EngineClientException(f"Bad request: {info['message']}")
            elif r.status_code == 401:
                raise EngineClientException(f"Unauthorized: {info['message']}")
            elif r.status_code == 429:
                time.sleep(2**attempt_number)  # retry with exponential backoff
                continue
            elif r.status_code == 500:
                raise EngineClientException(f"Internal error: {info['message']}")

        raise GamspyException(f"Unrecognized status code {r.status_code}")

    def logout(self) -> str:
        """
        Invalidates all of your JSON Web Tokens(JWTs)

        Returns
        -------
        str
        """
        import requests

        for attempt_number in range(MAX_REQUEST_ATTEMPS):
            r = requests.post(
                f"{self.client._engine_config.host}/auth/logout",
                headers=self.client.get_request_headers(),
                timeout=REQUEST_TIMEOUT,
            )

            if r.status_code == 200:
                response_data = r.content.decode("utf-8", errors="replace")
                info = json.loads(response_data)
                return info["message"]
            elif r.status_code == 400:
                raise EngineClientException("Bad request!")
            elif r.status_code == 401:
                raise EngineClientException("Unauthorized!")
            elif r.status_code == 429:
                time.sleep(2**attempt_number)  # retry with exponential backoff
                continue
            elif r.status_code == 500:
                raise EngineClientException("Internal error!")

        raise GamspyException(f"Unrecognized status code {r.status_code}")


class Job(Endpoint):
    def __init__(self, client: EngineClient) -> None:
        self.client = client
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
        import requests

        for attempt_number in range(MAX_REQUEST_ATTEMPS):
            r = requests.get(
                f"{self.client._engine_config.host}/jobs/{token}",
                headers=self.client.get_request_headers(),
                timeout=REQUEST_TIMEOUT,
            )
            response_data = r.content.decode("utf-8", errors="replace")

            if r.status_code == 200:
                info = json.loads(response_data)
                job_status = int(info["status"])
                return (
                    job_status,
                    STATUS_MAP.get(job_status, "unknown"),
                    info["process_status"],
                )
            elif r.status_code == 403:
                raise EngineClientException("Unauthorized!")
            elif r.status_code == 404:
                raise EngineClientException(f"Job {token} not found!")
            elif r.status_code == 429:
                time.sleep(2**attempt_number)  # retry with exponential backoff
                continue
            elif r.status_code == 500:
                raise EngineClientException("Internal error!")

        raise EngineClientException(
            "Getting job status from GAMS Engine failed after: "
            + str(MAX_REQUEST_ATTEMPS)
            + " attempts. Message: "
            + response_data
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
        import requests

        model_data_zip = self._create_zip_file(working_directory, gms_file, pf_file)
        gms_file = os.path.relpath(gms_file, working_directory)
        pf_file = (
            os.path.relpath(pf_file, working_directory) if pf_file is not None else None
        )
        query_params, file_params = self._get_params(model_data_zip, gms_file, pf_file)

        for attempt_number in range(MAX_REQUEST_ATTEMPS):
            r = requests.post(
                f"{self.client._engine_config.host}/jobs",
                params=query_params,
                files=file_params,
                headers=self.client.get_request_headers(),
                timeout=REQUEST_TIMEOUT,
            )
            response_data = r.content.decode("utf-8", errors="replace")
            if r.status_code == 201:
                return json.loads(response_data)["token"]
            elif r.status_code == 400:
                raise EngineClientException("Input is not valid!")
            elif r.status_code == 401:
                raise EngineClientException("Invalid authentication!")
            elif r.status_code == 402:
                raise EngineClientException("Quota exceeded!")
            elif r.status_code == 403:
                raise EngineClientException("Unauthorized access!")
            elif r.status_code == 404:
                raise EngineClientException("Namespace could not be found!")
            elif r.status_code == 429:
                time.sleep(2**attempt_number)  # retry with exponential backoff
                continue
            elif r.status_code == 500:
                raise EngineClientException("Internal error!")

        raise EngineClientException(
            "Creating job on GAMS Engine failed after: "
            + str(MAX_REQUEST_ATTEMPS)
            + " attempts. Message: "
            + response_data
        )

    def get_results(self, token: str, working_directory: str) -> None:
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
        import requests

        if not os.path.exists(working_directory):
            os.makedirs(working_directory, exist_ok=True)

        for attempt_number in range(MAX_REQUEST_ATTEMPS):
            r = requests.get(
                f"{self.client._engine_config.host}/jobs/{token}/result",
                headers=self.client.get_request_headers(),
                stream=True,
                timeout=REQUEST_TIMEOUT,
            )

            if r.status_code == 200:
                fd, path = tempfile.mkstemp()

                try:
                    with open(path, "wb") as out:
                        for chunk in r.iter_content(chunk_size=6000):
                            out.write(chunk)
                            out.flush()

                    r.close()

                    with zipfile.ZipFile(path, "r") as zip_ref:
                        zip_ref.extractall(working_directory)
                finally:
                    os.close(fd)
                    os.remove(path)

                logger.info(
                    f"Results have been extracted to your working directory: {working_directory}."
                )
                return
            elif r.status_code == 400:
                raise EngineClientException("Bad request!")
            elif r.status_code == 401:
                raise EngineClientException("Unauthorized!")
            elif r.status_code == 403:
                raise EngineClientException("Job data does not exist!")
            elif r.status_code == 404:
                raise EngineClientException("No job found!")
            elif r.status_code == 429:
                time.sleep(2**attempt_number)  # retry with exponential backoff
                continue
            else:
                response_data = r.content.decode("utf-8", errors="replace")
                raise EngineClientException(
                    "Fatal error while getting the results back from engine. GAMS"
                    f" Engine return code: {r.status_code}. Error message:"
                    f" {response_data}"
                )

        raise EngineClientException(
            "Getting job results from GAMS Engine failed after: "
            + str(MAX_REQUEST_ATTEMPS)
            + " attempts."
        )

    def delete_results(self, token: str) -> None:
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
        import requests

        for attempt_number in range(MAX_REQUEST_ATTEMPS):
            r = requests.delete(
                f"{self.client._engine_config.host}/jobs/{token}/result",
                headers=self.client.get_request_headers(),
                timeout=REQUEST_TIMEOUT,
            )
            response_data = r.content.decode("utf-8", errors="replace")

            if r.status_code == 200:
                logger.info(f"Results for {token} have been deleted.")
                return
            elif r.status_code == 400:
                raise EngineClientException("Bad request!")
            elif r.status_code == 401:
                raise EngineClientException("Unauthorized!")
            elif r.status_code == 403:
                raise EngineClientException("Job data does not exist!")
            elif r.status_code == 404:
                raise EngineClientException("No job found!")
            elif r.status_code == 429:
                time.sleep(2**attempt_number)  # retry with exponential backoff
                continue

        raise EngineClientException(
            "Removing job result failed with status code: "
            + str(r.status_code)
            + ". Message: "
            + response_data
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
        import requests

        for attempt_number in range(MAX_REQUEST_ATTEMPS):
            r = requests.delete(
                f"{self.client._engine_config.host}/jobs/{token}/unread-logs",
                headers=self.client.get_request_headers(),
                timeout=REQUEST_TIMEOUT,
            )
            response_data = r.content.decode("utf-8", errors="replace")

            if r.status_code == 200:
                info = json.loads(response_data)
                return info["message"], info["queue_finished"]
            elif r.status_code == 308:
                info = json.loads(response_data)
                return info["message"], True
            elif r.status_code == 403:
                raise EngineClientException(
                    "Cannot get logs of pending job / Unauthorized access!"
                )
            elif r.status_code == 404:
                raise EngineClientException("No job found!")
            elif r.status_code == 429:
                time.sleep(2**attempt_number)  # retry with exponential backoff
                continue

        raise EngineClientException(
            "Getting logs failed with status code: "
            + str(r.status_code)
            + ". "
            + response_data
            + "."
        )

    def list_jobs(
        self,
        page: int = 1,
        per_page: int = 10,
        order_by: str = "submitted_at",
        order_asc: bool = False,
        show_only_active: bool = False,
    ) -> dict:
        """
        Get request to /jobs/ which lists the jobs of the user.
        Refer to https://engine.gams.com/api/ for more details.

        Parameters
        ----------
        page : int, optional
            Page number, by default 1
        per_page : int, optional
            Number of jobs per page, by default 10
        order_by : str, optional
            One of "model", "namespace", "status", "process_status",
            "submitted_at", "finished_at" or "user", by default "submitted_at"
        order_asc : bool, optional
            Whether to sort in ascending order, by default False
        show_only_active : bool, optional
            Whether to list only the jobs that have not finished yet, by default False

        Returns
        -------
        dict
            Page with the keys "count", "next", "previous" and "results".
            "results" is the list of jobs on this page.

        Raises
        ------
        EngineClientException
            If get request has failed.
        """
        params = {
            "page": page,
            "per_page": per_page,
            "order_by": order_by,
            "order_asc": order_asc,
            "show_only_active": show_only_active,
        }
        r = self.client._send("GET", "/jobs/", params=params, headers={"X-Fields": "*"})
        return r.json()

    def cancel(self, token: str, hard_kill: bool = False) -> None:
        """
        Delete request to /jobs/{token} which cancels a pending or running job.
        Refer to https://engine.gams.com/api/ for more details.

        Parameters
        ----------
        token : str
            Job token
        hard_kill : bool, optional
            Sends SIGKILL instead of SIGINT if True, by default False

        Raises
        ------
        EngineClientException
            If the job is already finished or the delete request has failed.
        """
        self.client._send("DELETE", f"/jobs/{token}", params={"hard_kill": hard_kill})

    def get_text_entry(
        self,
        token: str,
        entry_name: str,
        start_position: int = 1,
        length: int | None = None,
    ) -> str:
        """
        Get request to /jobs/{token}/text-entry/{entry_name} which returns the
        value of a text entry. Text entries are requested with
        ``engine_options={"text_entries": [...]}``.
        Refer to https://engine.gams.com/api/ for more details.

        Parameters
        ----------
        token : str
            Job token
        entry_name : str
            Name of the text entry
        start_position : int, optional
            Start position of the substring, by default 1
        length : int | None, optional
            Number of characters to read, by default the whole entry

        Returns
        -------
        str
            Value of the text entry

        Raises
        ------
        EngineClientException
            If get request has failed.
        """
        params: dict[str, int] = {"start_position": start_position}
        if length is not None:
            params["length"] = length

        r = self.client._send(
            "GET", f"/jobs/{token}/text-entry/{entry_name}", params=params
        )
        return r.json()["entry_value"]

    def get_stream_entry(self, token: str, entry_name: str) -> tuple[str, bool]:
        """
        Delete request to /jobs/{token}/stream-entry/{entry_name} which returns
        the unread part of a stream entry. Stream entries are requested with
        ``engine_options={"stream_entries": [...]}``.
        Refer to https://engine.gams.com/api/ for more details.

        Parameters
        ----------
        token : str
            Job token
        entry_name : str
            Name of the stream entry

        Returns
        -------
        tuple[str, bool]
            Unread part of the stream entry and whether the stream is finished

        Raises
        ------
        EngineClientException
            If delete request has failed.
        """
        r = self.client._send(
            "DELETE",
            f"/jobs/{token}/stream-entry/{entry_name}",
            ok_codes=(200, 308),
        )
        if r.status_code == 308:
            return "", True

        info = r.json()
        return info["entry_value"], info["queue_finished"]

    def update_tag(self, token: str, tag: str) -> None:
        """
        Put request to /jobs/{token}/tag which updates the tag of a job.
        Refer to https://engine.gams.com/api/ for more details.

        Parameters
        ----------
        token : str
            Job token
        tag : str
            New tag, at most 255 characters

        Raises
        ------
        EngineClientException
            If put request has failed.
        """
        self.client._send("PUT", f"/jobs/{token}/tag", params={"tag": tag})

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

        model_files += self.client.extra_model_files
        model_files = get_relative_paths(model_files, working_directory)

        with zipfile.ZipFile(model_data_zip, "w", zipfile.ZIP_DEFLATED) as model_data:
            for model_file in model_files:
                model_data.write(
                    os.path.join(working_directory, model_file),
                    arcname=model_file,
                )

        model_data_zip.seek(0)

        return model_data_zip

    def _get_params(
        self, model_data_zip: io.BytesIO, gms_file: str, pf_file: str | None
    ) -> tuple[dict, dict]:
        file_params = {}
        query_params = copy.deepcopy(self.engine_options) if self.engine_options else {}

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


class Usage(Endpoint):
    def __init__(self, client: EngineClient) -> None:
        self.client = client

    def get(
        self,
        from_datetime: datetime | str | None = None,
        to_datetime: datetime | str | None = None,
        tokens: list[str] | None = None,
    ) -> dict:
        """
        Get request to /usage/ which returns the usage data of the user.
        A client authenticated with a JWT needs the USERS scope to look up its username.
        Refer to https://engine.gams.com/api/ for more details.

        Parameters
        ----------
        from_datetime : datetime | str | None, optional
            Exclude jobs submitted before this time, by default None
        to_datetime : datetime | str | None, optional
            Exclude jobs submitted after this time, by default None
        tokens : list[str] | None, optional
            Only include these jobs, by default None

        Returns
        -------
        dict
            Usage data with the keys "job_usage", "hypercube_job_usage" and "pool_usage"

        Raises
        ------
        EngineClientException
            If get request has failed.
        """
        params: dict[str, str | list[str]] = {"username": self.client._get_username()}
        for key, value in (
            ("from_datetime", from_datetime),
            ("to_datetime", to_datetime),
        ):
            if value is not None:
                params[key] = value if isinstance(value, str) else value.isoformat()

        if tokens is not None:
            params["token"] = tokens

        return self.client._send("GET", "/usage/", params=params).json()

    def get_quota(self) -> list[dict]:
        """
        Get request to /usage/quota which returns the quotas of the user.
        A client authenticated with a JWT needs the USERS scope to look up its username.
        Refer to https://engine.gams.com/api/ for more details.

        Returns
        -------
        list[dict]
            Quotas that apply to the user, one per user they are inherited from

        Raises
        ------
        EngineClientException
            If get request has failed.
        """
        params = {"username": self.client._get_username()}
        return self.client._send("GET", "/usage/quota", params=params).json()


class Namespaces(Endpoint):
    def __init__(self, client: EngineClient) -> None:
        self.client = client

    def get_models(
        self, namespace: str | None = None, model: str | None = None
    ) -> list[dict]:
        """
        Get request to /namespaces/{namespace} which lists the models registered
        in a namespace. A registered model can be run with
        ``engine_options={"model": <name>}``.
        Refer to https://engine.gams.com/api/ for more details.

        Parameters
        ----------
        namespace : str | None, optional
            Namespace, by default the namespace of the client
        model : str | None, optional
            Only list the model with this name, by default None

        Returns
        -------
        list[dict]
            Registered models

        Raises
        ------
        EngineClientException
            If get request has failed.
        """
        if namespace is None:
            namespace = self.client._engine_config.namespace

        params = {"model": model} if model is not None else None
        return self.client._send(
            "GET", f"/namespaces/{namespace}", params=params
        ).json()


class EngineClient:
    """
    GAMS Engine Client.

    Parameters
    ----------
    host : str
        Host address.
    username : str, optional
        Username
    password : str, optional
        Password
    jwt : str, optional
        JWT Token
    namespace : str, optional
        Namespace, by default "global"
    extra_model_files : list[str], optional
        extra_model_files, by default []
    engine_options : dict | None, optional
        GAMS Engine options, by default None
    remove_results : bool, optional
        Whether to remove the results from GAMS Engine Server, by default False
    is_blocking : bool, optional
        Synchronous (blocking) or asynchronous, by default True

    Examples
    --------
    >>> import gamspy as gp
    >>> client = gp.EngineClient(host="https://engine.gams.com/api/", username="<username>", password="<password>")

    """

    def __init__(
        self,
        host: str,
        username: str = "",
        password: str = "",
        jwt: str = "",
        namespace: str = "global",
        extra_model_files: list[str] | None = None,
        engine_options: dict | None = None,
        *,
        remove_results: bool = False,
        is_blocking: bool = True,
    ) -> None:
        self.host = host
        self.username = username
        self.password = password
        self.jwt = jwt
        self.namespace = namespace
        if extra_model_files is None:
            extra_model_files = []
        self.extra_model_files = extra_model_files
        self.engine_options = engine_options
        self.remove_results = remove_results
        self.is_blocking = is_blocking
        self.tokens: list[str] = []

        self._engine_config = EngineConfiguration(
            self.host,
            self.username,
            self.password,
            self.jwt,
            self.namespace,
        )

        # Endpoints
        self.job = Job(self)
        self.auth = Auth(self)
        self.usage = Usage(self)
        self.namespaces = Namespaces(self)

    def get_request_headers(self) -> dict[str, str]:
        return {
            "Authorization": self._engine_config._get_auth_header(),
            "User-Agent": "GAMSPy EngineClient",
            "Accept": "application/json",
        }

    def get_version(self) -> dict:
        """
        Get request to /version which returns the GAMS Engine and GAMS versions.
        Refer to https://engine.gams.com/api/ for more details.

        Returns
        -------
        dict
            Versions with the keys "version", "gams_version" and "in_kubernetes"

        Raises
        ------
        EngineClientException
            If get request has failed.
        """
        return self._send("GET", "/version").json()

    def _get_username(self) -> str:
        if self._engine_config.username:
            return self._engine_config.username

        r = self._send("GET", "/users/", params={"everyone": False})
        return r.json()[0]["username"]

    def _send(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        headers: dict[str, str] | None = None,
        ok_codes: tuple[int, ...] = (200,),
    ) -> requests.Response:
        import requests

        headers = self.get_request_headers() | (headers or {})
        for attempt_number in range(MAX_REQUEST_ATTEMPS):
            r = requests.request(
                method,
                f"{self._engine_config.host}{path}",
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
            if r.status_code in ok_codes:
                return r

            if r.status_code != 429:
                break

            time.sleep(2**attempt_number)  # retry with exponential backoff

        try:
            message = r.json()["message"]
        except (ValueError, KeyError, TypeError):
            message = r.text

        raise EngineClientException(
            f"{method} {path} failed with status code {r.status_code}: {message}"
        )


class GAMSEngine(backend.Backend):
    def __init__(
        self,
        container: Container,
        options: Options,
        solver: str,
        solver_options: dict | Path | None,
        output: TextIO | None,
        model: Model,
        client: EngineClient,
    ) -> None:
        if solver == "mpsge":
            raise ValidationError(f"`{solver}` is not a valid solver for GAMS Engine.")

        super().__init__(
            "engine",
            container,
            options,
            solver,
            solver_options,
            output,
        )
        self.model = model
        self.client = client

    def is_async(self) -> bool:
        return not self.client.is_blocking

    def run(self) -> pd.DataFrame | None:
        # Run a dummy job to get the restart file to be sent to GAMS Engine
        self._create_restart_file()

        # Generate gams string
        self.model._assign_model_attributes("engine")
        gams_string = self.preprocess()

        self.execute_gams(gams_string)

        if self.is_async():
            return None

        # Synchronize GAMSPy with checkpoint and return a summary
        summary = self.postprocess()

        # Run another dummy job to synchronize GAMS and GAMSPy state
        self._sync()

        return summary

    def execute_gams(self, gams_string: str) -> None:
        hidden_options = {
            "gdx": os.path.basename(self.container._gdx_out),
            "gdxSymbols": "newOrChanged",
            "restart": os.path.basename(self.restart_file),
            "input": os.path.basename(self.gms_file),
        }
        self.options._set_hidden_options(hidden_options)
        self.options._export(self.pf_file, self.output)

        with open(self.gms_file, "w", encoding="utf-8") as file:
            file.write(gams_string)

        try:
            original_extra_files = self.client.extra_model_files
            self.client.extra_model_files = self._append_gamspy_files()
            try:
                token = self.client.job.post(
                    self.container.working_directory,
                    self.gms_file,
                    self.pf_file,
                )
            finally:
                self.client.extra_model_files = original_extra_files

            self.client.tokens.append(token)

            if not self.is_async():
                return_code = self._wait_for_job(token)
                self.client.job.get_results(token, self.container.working_directory)
                if self.client.remove_results:
                    self.client.job.delete_results(token)

                engine_output_path = os.path.join(
                    self.container.working_directory, "log_stdout.txt"
                )

                if self.options.log_file:
                    # logoption = 2 | 4
                    shutil.move(engine_output_path, self.options.log_file)
                else:
                    try:
                        os.remove(engine_output_path)  # logoption = 0 | 3
                    except FileNotFoundError:
                        ...

                if return_code:
                    message = (
                        f"Return code {return_code}. {GAMS_STATUS.get(return_code, '')}"
                        f" Check {self.job_name + '.lst'} for more information."
                        + _customize_exception(self.options, self.job_name, return_code)
                    )
                    self.container._workspace._errors.append(message)
                    raise EngineException(message, return_code=return_code)

                self.model._update_model_attributes()
        finally:
            self.container._unsaved_statements = []
            self.container._delete_autogenerated_symbols()

    def _wait_for_job(self, token: str) -> int | None:
        """
        Polls the job until it finishes and streams its log to the output.
        Returns the GAMS return code of the job.
        """
        logs_finished = self.output is None
        last_message = None

        while True:
            job_status, message, return_code = self.client.job.get(token)

            if job_status in (-1, -3):
                raise EngineException(
                    f"Could not get job results because the job is {message}.",
                    return_code=job_status,
                    status_code=job_status,
                )

            if job_status not in STATUS_MAP:
                raise EngineException(
                    "Unknown job status code! Currently supported job"
                    f" status codes: {STATUS_MAP.keys()}",
                    return_code=job_status,
                    status_code=job_status,
                )

            # Logs of a finished job can still be fetched once.
            if job_status in (1, 2, 10) and not logs_finished:
                assert self.output is not None
                logs, logs_finished = self.client.job.get_logs(token)
                self.output.write(logs)

            if job_status == 10 and logs_finished:
                return return_code

            if job_status in (-10, -2, 0) and message != last_message:
                logger.info(f"Job status is {message}...")
                last_message = message

            time.sleep(JOB_POLL_INTERVAL)

    def postprocess(self) -> pd.DataFrame | None:
        if self.client.remove_results or self.model is None:
            return None

        if not self.is_async():
            self.parse_listings(self.model)
            return self.prepare_summary(self.model)

        return None

    def _prepare_dummy_options(self) -> dict:
        scrdir = self.container._process_directory

        hidden_options = {
            "input": self.gms_file,
            "sysdir": self.container.system_directory,
            "scrdir": scrdir,
            "scriptnext": os.path.join(scrdir, "gamsnext.sh"),
            "logoption": 0,
            "previouswork": 1,
            "license": self.container._license_path,
            "save": self.restart_file,
        }

        if self.container._network_license:
            hidden_options["netlicense"] = os.path.join(scrdir, "gamslice.dat")

        return hidden_options

    def _create_restart_file(self) -> None:
        with open(self.gms_file, "w", encoding="utf-8") as gams_file:
            gams_file.write("")

        options = Options()
        hidden_options = self._prepare_dummy_options()
        options._set_hidden_options(hidden_options)
        options._export(self.pf_file)

        engine = self.container._execution_engine
        if not isinstance(engine, SocketEngine):  # pragma: no cover
            raise ValidationError(
                f"The {self.backend_type} backend requires the socket execution engine."
            )

        engine.send_job(self.job_name, self.pf_file)

    def _sync(self) -> None:
        symbols = gdxio._get_symbol_names_from_gdx(
            self.container.system_directory, self.container._gdx_out
        )
        dirty_str = ",".join(symbols)
        self.container._add_statement(
            f'execute_load "{self.container._gdx_out}", {dirty_str};'
        )
        self.container._synch_with_gams()
        self.container._should_load_from(symbols, source=DataSource.GAMS)

    def _append_gamspy_files(self) -> list[str]:
        extra_model_files = self.client.extra_model_files + [self.restart_file]
        if os.path.exists(self.container._gdx_in):
            extra_model_files.append(self.container._gdx_in)

        if self.solver_options:
            assert self.solver is not None
            extra_model_files.append(
                os.path.join(
                    self.container.working_directory,
                    f"{self.solver.lower()}.opt",
                )
            )

            if self.model._indicator_file is not None:
                extra_model_files.append(self.model._indicator_file)

        if self.options.loadpoint is not None:
            extra_model_files.append(str(self.options.loadpoint))

        if self.model.external_module is not None:
            extra_model_files.append(self.model.external_module)

        return extra_model_files
