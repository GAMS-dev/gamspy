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
from typing import List
from typing import Optional
from typing import TYPE_CHECKING

from gams import GamsEngineConfiguration
from gams import GamsJob
from gams import GamsOptions
from gams.control.workspace import GamsException
from gams.control.workspace import GamsExceptionExecution
from pydantic import BaseModel

import gamspy._backend.backend as backend
from gamspy.exceptions import GamspyException

if TYPE_CHECKING:
    import io
    from gamspy import Container


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
        config: "EngineConfig" | None,
        options: "GamsOptions",
        output: Optional[io.TextIOWrapper] = None,
    ) -> None:
        if config is None:
            raise GamspyException(
                "`engine_config` must be provided to solve on GAMS Engine"
            )

        self.container = container
        self.config = config
        self.options = options
        self.output = output
        self.gdx_in = os.path.basename(container._gdx_in)
        self.gdx_out = os.path.basename(container._gdx_out)

    def is_async(self):
        return False

    def preprocess(
        self,
        dirty_names: List[str],
        modified_names: List[str],
    ):
        self.gams_string = self.container._generate_gams_string(
            self.gdx_in, self.gdx_out, dirty_names, modified_names
        )

    def run(self):
        extra_model_files = self._preprocess_extra_model_files()

        checkpoint = None
        if os.path.exists(self.container._restart_from._checkpoint_file_name):
            checkpoint = self.container._restart_from

        job = GamsJob(
            self.container.workspace,
            job_name=f"_job_{uuid.uuid4()}",
            source=self.gams_string,
            checkpoint=checkpoint,
        )

        try:
            self.container._job = job
            job.run_engine(  # type: ignore
                engine_configuration=self.config._get_engine_config(),
                extra_model_files=extra_model_files,
                gams_options=self.options,
                checkpoint=self.container._save_to,
                output=self.output,
                create_out_db=False,
                engine_options=self.config.engine_options,
                remove_results=self.config.remove_results,
            )
        except (GamsException, GamsExceptionExecution) as e:
            raise GamspyException(str(e))
        finally:
            self.container._unsaved_statements = []

    def postprocess(self, is_implicit: bool = False):
        if (
            self.config.remove_results
            or self.options.traceopt != 3
            or is_implicit
        ):
            return None

        return self.prepare_summary(
            self.container.working_directory, self.options.trace
        )

    def _preprocess_extra_model_files(self) -> List[str]:
        for extra_file in self.config.extra_model_files:
            try:
                shutil.copy(
                    extra_file, self.container.workspace.working_directory
                )
            except shutil.SameFileError:
                # extra file might already be in the working directory
                pass

        extra_model_files = [
            os.path.basename(extra_file)
            for extra_file in self.config.extra_model_files
        ]

        extra_model_files.append(self.gdx_in)

        return extra_model_files
