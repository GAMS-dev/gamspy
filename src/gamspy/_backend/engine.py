#
# GAMS - General Algebraic Modeling System Python API
#
# Copyright (c) 2023 GAMS Development Corp. <support@gams.com>
# Copyright (c) 2023 GAMS Software GmbH <support@gams.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
from __future__ import annotations

import os
import io
import uuid
import json
import time
import copy
import logging
import tempfile

from typing import List
from typing import Optional
from typing import TYPE_CHECKING

from dataclasses import dataclass
from abc import ABC

import urllib3
import urllib.parse
import certifi
import zipfile

from gams import DebugLevel
from gams import GamsEngineConfiguration
from gams import GamsOptions
from gams import GamsCheckpoint
from gams.control.workspace import GamsException
from gams.control.workspace import GamsExceptionExecution
from gams.core.gmo import gmoProc_nrofmodeltypes
from gams.core.cfg import cfgModelTypeName
from gams.core.opt import optSetStrStr

import gamspy._backend.backend as backend
from gamspy.exceptions import GamspyException
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from gamspy import Container
    from gamspy import Model


logger = logging.getLogger("ENGINE")
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(name)s - %(levelname)s] %(message)s")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


MAX_REQUEST_ATTEMPS = 3
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


@dataclass
class GamsEngineJob:
    token: str
    configuration: GamsEngineConfiguration
    request_headers: dict


class Endpoint(ABC): ...


class Job(Endpoint):
    def __init__(
        self,
        http: urllib3.PoolManager,
        extra_model_files: list,
        engine_config: GamsEngineConfiguration,
        request_headers: dict,
        engine_options: Optional[dict] = None,
    ) -> None:
        self._http = http
        self.extra_model_files = extra_model_files
        self.engine_options = engine_options
        self._engine_config = engine_config
        self._request_headers = request_headers

    def get(self, token: str):
        for attempt_number in range(MAX_REQUEST_ATTEMPS):
            r = self._http.request(
                "GET",
                self._engine_config.host + f"/jobs/{token}",
                headers=self._request_headers,
            )
            response_data = r.data.decode("utf-8", errors="replace")

            if r.status == 200:
                job_status = int(json.loads(response_data)["status"])
                return job_status, STATUS_MAP[job_status]
            elif r.status == 429:
                time.sleep(2**attempt_number)  # retry with exponential backoff
                continue

            raise GamspyException(
                "Creating job on GAMS Engine failed with status code: "
                + str(r.status)
                + ". Message: "
                + response_data
            )
        else:
            raise GamspyException(
                "Creating job on GAMS Engine failed after: "
                + str(MAX_REQUEST_ATTEMPS)
                + " attempts. Message: "
                + response_data
            )

    def post(
        self,
        working_directory: str,
        job_name: str,
        restart_file: GamsCheckpoint,
        options: GamsOptions,
    ):
        model_data_zip = self._create_zip_file(
            working_directory, job_name, restart_file
        )
        query_params, file_params = self._get_params(
            model_data_zip, options, job_name
        )

        for attempt_number in range(MAX_REQUEST_ATTEMPS):
            r = self._http.request(
                "POST",
                self._engine_config.host
                + "/jobs/?"
                + urllib.parse.urlencode(query_params, doseq=True),
                fields=file_params,
                headers=self._request_headers,
            )
            print(r.status)
            response_data = r.data.decode("utf-8", errors="replace")
            if r.status == 201:
                break
            elif r.status == 429:
                time.sleep(2**attempt_number)  # retry with exponential backoff
                continue

            raise GamspyException(
                "Creating job on GAMS Engine failed with status code: "
                + str(r.status)
                + ". Message: "
                + response_data
            )
        else:
            raise GamspyException(
                "Creating job on GAMS Engine failed after: "
                + str(MAX_REQUEST_ATTEMPS)
                + " attempts. Message: "
                + response_data
            )

        return GamsEngineJob(  # type: ignore
            json.loads(response_data)["token"],
            self._engine_config,
            self._request_headers,
        )

    def get_results(self, token: str, working_directory: str):
        r = self._http.request(
            "GET",
            self._engine_config.host + f"/jobs/{token}/result",
            headers=self._request_headers,
            preload_content=False,
        )

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

    def delete_results(self, token: str):
        for attempt_number in range(MAX_REQUEST_ATTEMPS):
            r = self._http.request(
                "DELETE",
                self._engine_config.host + "/jobs/" + token + "/result",
                headers=self._request_headers,
            )
            response_data = r.data.decode("utf-8", errors="replace")
            if r.status in [200, 403]:
                return
            elif r.status == 429:
                time.sleep(2**attempt_number)  # retry with exponential backoff
                continue

            raise GamspyException(
                "Removing job result failed with status code: "
                + str(r.status)
                + ". Message: "
                + response_data
            )
        else:
            raise GamspyException(
                "Removing job result failed after: "
                + str(MAX_REQUEST_ATTEMPS)
                + " attempts. Message: "
                + response_data
            )

    def get_logs(self, token: str) -> int:
        poll_logs_sleep_time = 1

        while True:
            r = self._http.request(
                "DELETE",
                self._engine_config.host + f"/jobs/{token}/unread-logs",
                headers=self._request_headers,
            )
            response_data = r.data.decode("utf-8", errors="replace")
            if r.status == 429:
                # too many requests, slow down
                poll_logs_sleep_time = min(poll_logs_sleep_time + 1, 5)
                time.sleep(poll_logs_sleep_time)
                continue
            elif r.status == 403:
                # job still in queue
                time.sleep(poll_logs_sleep_time)
                continue
            elif r.status != 200:
                raise GamspyException(
                    "Getting logs failed with status code: "
                    + str(r.status)
                    + ". Message: "
                    + response_data
                )
            response_data = json.loads(response_data)
            stdout_data = response_data["message"]
            if stdout_data != "":
                print(stdout_data, end="")

            if response_data["queue_finished"] is True:
                exitcode = response_data["gams_return_code"]
                break
            time.sleep(poll_logs_sleep_time)

        return exitcode

    def _create_zip_file(
        self,
        working_directory: str,
        job_name: str,
        restart_file: GamsCheckpoint,
    ) -> io.BytesIO:
        model_data_zip = tempfile.NamedTemporaryFile(delete=False)
        model_files = {job_name + ".gms", job_name + ".pf"}

        if os.path.exists(restart_file._checkpoint_file_name):
            model_files.add(restart_file._checkpoint_file_name)

        if self.extra_model_files:
            extra_model_files_cleaned = {
                (x if os.path.isabs(x) else os.path.join(working_directory, x))
                for x in self.extra_model_files
            }
            model_files.update(extra_model_files_cleaned)

        with zipfile.ZipFile(
            model_data_zip, "w", zipfile.ZIP_DEFLATED
        ) as model_data:
            for model_file in model_files:
                model_data.write(
                    model_file,
                    arcname=(
                        os.path.relpath(model_file, working_directory)
                        if os.path.isabs(model_file)
                        else model_file
                    ),
                )

        model_data_zip.seek(0)

        return model_data_zip

    def _get_params(self, model_data_zip, options, job_name):
        file_params = {}
        query_params = (
            copy.deepcopy(self.engine_options) if self.engine_options else {}
        )

        query_params["namespace"] = self._engine_config.namespace

        if "data" in query_params or "model_data" in query_params:
            raise GamspyException(
                "`engine_options` must not include keys `data` or "
                "`model_data`. Please use `extra_model_files` to "
                "provide additional files to send to GAMS Engine."
            )

        if "inex_file" in query_params:
            if isinstance(query_params["inex_file"], io.IOBase):
                file_params["inex_file"] = (
                    "inex.json",
                    query_params["inex_file"].read(),
                    "application/json",
                )
            else:
                with open(query_params["inex_file"], "rb") as f:
                    file_params["inex_file"] = (
                        "inex.json",
                        f.read(),
                        "application/json",
                    )
            del query_params["inex_file"]

        if "model" in query_params:
            file_params["data"] = (
                "data.zip",
                model_data_zip.read(),
                "application/zip",
            )
        else:
            query_params["run"] = options._input
            query_params["model"] = os.path.splitext(options._input)[0]
            file_params["model_data"] = (
                "data.zip",
                model_data_zip.read(),
                "application/zip",
            )

        model_data_zip.close()

        if "arguments" in query_params:
            if not isinstance(query_params["arguments"], list):
                query_params["arguments"] = [query_params["arguments"]]
            query_params["arguments"].append(
                f"pf={os.path.basename(job_name)}.pf"
            )
        else:
            query_params["arguments"] = [
                "pf=" + os.path.basename(job_name) + ".pf"
            ]

        return query_params, file_params


class EngineClient:
    def __init__(
        self,
        host: str,
        username: str | None = None,
        password: str | None = None,
        jwt: str | None = None,
        namespace: str = "global",
        extra_model_files: List[str] = [],
        engine_options: Optional[dict] = None,
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
        self._request_headers = {
            "Authorization": self._engine_config._get_auth_header(),
            "User-Agent": "GAMS Python API",
            "Accept": "application/json",
        }

        # Endpoints
        self.job = Job(
            self._http,
            extra_model_files,
            self._engine_config,
            self._request_headers,
            engine_options,
        )

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
        container: "Container",
        client: "EngineClient" | None,
        options: "GamsOptions",
        output: Optional[io.TextIOWrapper] = None,
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
        self.options = options
        self.output = output
        self.model = model
        self.job_name = os.path.join(
            self.container.working_directory, f"_job_{uuid.uuid4()}"
        )
        self.gms_file = self.job_name + ".gms"
        self.pf_path = self.job_name + ".pf"

    def is_async(self):
        return False if self.client.is_blocking else True

    def preprocess(self, keep_flags: bool = False):
        gams_string, dirty_names = super().preprocess(keep_flags)

        self.client.extra_model_files.append(self.gdx_in)

        # Set selected solvers
        for i in range(1, gmoProc_nrofmodeltypes):
            optSetStrStr(
                self.options._opt,
                cfgModelTypeName(self.options._cfg, i),
                self.options._selected_solvers[i],
            )

        # Set save file path
        self.options._save = os.path.basename(
            self.container._save_to._checkpoint_file_name
        )

        # Set restart file path
        if os.path.exists(self.container._restart_from._checkpoint_file_name):
            self.options._restart = os.path.basename(
                self.container._restart_from._checkpoint_file_name
            )

        # Set input file path
        self.options._input = os.path.basename(self.job_name) + ".gms"

        # Set output file path
        if not self.options.output:
            self.options.output = os.path.basename(self.job_name) + ".lst"

        # Export pf file
        self.options.export(os.path.basename(self.pf_path))

        # Export gms file
        with open(self.gms_file, "w", encoding="utf-8") as file:
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
            job = self.client.job.post(
                self.container.working_directory,
                self.job_name,
                self.container._restart_from,
                self.options,
            )
            print(f"{job.token=}")
            self.client.tokens.append(job.token)

            if not self.is_async() and self.model:
                job_status, message = self.client.job.get(job.token)

                if job_status < 0:
                    raise GamspyException(
                        "Could not get job results because the job is"
                        f" {message}."
                    )

                while job_status in [0, 1, 2]:
                    logger.info(f"Job status is {message}...")
                    job_status = self.client.job.get(job.token)

                exit_code = self.client.job.get_logs(job.token)

                if exit_code != 0:
                    raise GamspyException(
                        "Could not get the job logs. Return code is"
                        f" {exit_code}. Check"
                        f" `{os.path.join(self.container.working_directory, self.job_name + '.lst')}`"
                        " for further details."
                    )

                self.client.job.get_results(
                    job.token, self.container.working_directory
                )

                self.model._update_model_attributes()
        except (GamsException, GamsExceptionExecution) as e:
            if self.container._debugging_level == "keep_on_error":
                self.container.workspace._debug = DebugLevel.KeepFiles

            raise GamspyException(str(e))
        finally:
            self.container._unsaved_statements = []
            self.container._delete_autogenerated_symbols()

    def postprocess(self, dirty_names: List[str], is_implicit: bool = False):
        self.container._load_records_from_gdx(
            self.container._gdx_out,
            dirty_names + self.container._import_symbols,
        )
        self.container._swap_checkpoints()

        if (
            self.client.remove_results
            or self.options.traceopt != 3
            or is_implicit
        ):
            return None

        return self.prepare_summary(
            self.container.working_directory, self.options.trace
        )
