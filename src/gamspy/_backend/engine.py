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
import shutil
import uuid
import time
import copy

from typing import List
from typing import Optional
from typing import TYPE_CHECKING

import urllib3
import urllib.parse
import certifi

from gams import DebugLevel
from gams import GamsEngineConfiguration
from gams import GamsJob
from gams import GamsOptions
from gams.control.workspace import GamsException
from gams.control.workspace import GamsExceptionExecution
from pydantic import BaseModel

import gamspy._backend.backend as backend
from gamspy.exceptions import GamspyException
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    import io
    from gamspy import Container
    from gamspy import Model


MAX_REQUEST_ATTEMPS = 3


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
    ):
        self.host = host
        self.username = username
        self.password = password
        self.jwt = jwt
        self.namespace = namespace
        self.extra_model_files = extra_model_files
        self.engine_options = engine_options
        self.remove_results = remove_results

        self._http = urllib3.PoolManager(
            cert_reqs="CERT_REQUIRED", ca_certs=certifi.where()
        )

        self._engine_config = self._get_engine_config()
        self._request_headers = {
            "Authorization": self._engine_config._get_auth_header(),
            "User-Agent": "GAMS Python API",
            "Accept": "application/json",
        }

    def _get_engine_config(self):
        return GamsEngineConfiguration(
            self.host,
            self.username,
            self.password,
            self.jwt,
            self.namespace,
        )

    def _get_params(self):
        query_params = (
            copy.deepcopy(self.engine_options) if self.engine_options else {}
        )
        query_params["namespace"] = self._engine_config.namespace

    def job_post(self):
        query_params, file_params = self._get_params()

        for attempt_number in range(MAX_REQUEST_ATTEMPS):
            r = self._http.request(
                "POST",
                self._engine_config.host
                + "/jobs/?"
                + urllib.parse.urlencode(query_params, doseq=True),
                fields=file_params,
                headers=self._request_headers,
            )
            response_data = r.data.decode("utf-8", errors="replace")
            if r.status == 201:
                break
            elif r.status == 429:
                # retry
                time.sleep(2**attempt_number)
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

    def job_delete(self, token):
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
                # retry
                time.sleep(2**attempt_number)
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


class EngineConfig(BaseModel):
    host: str
    username: Optional[str] = None
    password: Optional[str] = None
    jwt: Optional[str] = None
    namespace: str = "global"
    extra_model_files: List[str] = []
    engine_options: Optional[dict] = None
    remove_results: bool = False

    class Config:
        extra = "forbid"

    def _get_engine_config(self):
        return GamsEngineConfiguration(
            self.host,
            self.username,
            self.password,
            self.jwt,
            self.namespace,
        )


class GAMSEngine(backend.Backend):
    def __init__(
        self,
        container: "Container",
        config: "EngineClient" | None,
        options: "GamsOptions",
        output: Optional[io.TextIOWrapper] = None,
        model: Model | None = None,
    ) -> None:
        if config is None:
            raise ValidationError(
                "`engine_config` must be provided to solve on GAMS Engine"
            )

        super().__init__(
            container,
            os.path.basename(container._gdx_in),
            os.path.basename(container._gdx_out),
        )

        self.client = config
        self.options = options
        self.output = output
        self.model = model

    def is_async(self):
        return False

    def solve(self, is_implicit: bool = False, keep_flags: bool = False):
        # Generate gams string and write modified symbols to gdx
        gams_string, dirty_names = self.preprocess(keep_flags)

        # Run the model
        self.run(gams_string)

        # Synchronize GAMSPy with checkpoint and return a summary
        summary = self.postprocess(dirty_names, is_implicit)

        return summary

    def run(self, gams_string: str):
        extra_model_files = self._preprocess_extra_model_files()

        checkpoint = None
        if os.path.exists(self.container._restart_from._checkpoint_file_name):
            checkpoint = self.container._restart_from

        job = GamsJob(
            self.container.workspace,
            job_name=f"_job_{uuid.uuid4()}",
            source=gams_string,
            checkpoint=checkpoint,
        )

        try:
            self.container._job = job
            job.run_engine(  # type: ignore
                engine_configuration=self.client._get_engine_config(),
                extra_model_files=extra_model_files,
                gams_options=self.options,
                checkpoint=self.container._save_to,
                output=self.output,
                create_out_db=False,
                engine_options=self.client.engine_options,
                remove_results=self.client.remove_results,
            )
            if not self.is_async() and self.model:
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

    def _preprocess_extra_model_files(self) -> List[str]:
        for extra_file in self.client.extra_model_files:
            try:
                shutil.copy(
                    extra_file, self.container.workspace.working_directory
                )
            except shutil.SameFileError:
                # extra file might already be in the working directory
                pass

        extra_model_files = [
            os.path.basename(extra_file)
            for extra_file in self.client.extra_model_files
        ]

        extra_model_files.append(self.gdx_in)

        return extra_model_files
