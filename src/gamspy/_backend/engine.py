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
import uuid
from typing import List
from typing import Optional
from typing import TYPE_CHECKING

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

        self.config = config
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
        extra_model_files = self._validate_extra_model_files()

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
                engine_configuration=self.config._get_engine_config(),
                extra_model_files=extra_model_files,
                gams_options=self.options,
                checkpoint=self.container._save_to,
                output=self.output,
                create_out_db=False,
                engine_options=self.config.engine_options,
                remove_results=self.config.remove_results,
            )
            if not self.is_async() and self.model:
                self.model._update_model_attributes()
        except (GamsException, GamsExceptionExecution) as e:
            if self.container._debugging_level == "keep_on_error":
                self.container.workspace._debug = DebugLevel.KeepFiles

            raise GamspyException(str(e)) from e
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
            self.config.remove_results
            or self.options.traceopt != 3
            or is_implicit
        ):
            return None

        return self.prepare_summary(
            self.container.working_directory, self.options.trace
        )

    def _validate_extra_model_files(self) -> List[str]:
        extra_model_files = []
        for extra_file in self.config.extra_model_files:
            relative_path = os.path.relpath(
                extra_file, self.container.working_directory
            )
            if relative_path.startswith("."):
                raise ValidationError(
                    "Extra model file path must be relative to the working"
                    f" directory.The given path: {extra_file}, the working"
                    f" directory: {self.container.working_directory}, the"
                    f" relative path: {relative_path}"
                )

            extra_model_files.append(relative_path)

        extra_model_files.append(
            os.path.join(self.container.working_directory, self.gdx_in)
        )

        return extra_model_files
