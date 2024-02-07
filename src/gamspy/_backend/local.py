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
from typing import TYPE_CHECKING

from gams import DebugLevel
from gams import GamsJob
from gams import GamsOptions
from gams.control.workspace import GamsExceptionExecution

import gamspy._backend.backend as backend
from gamspy.exceptions import customize_exception
from gamspy.exceptions import GamspyException

if TYPE_CHECKING:
    import io
    from gamspy import Container
    from gamspy import Model


class Local(backend.Backend):
    def __init__(
        self,
        container: Container,
        options: GamsOptions,
        output: io.TextIOWrapper | None = None,
        model: Model | None = None,
    ) -> None:
        super().__init__(container, container._gdx_in, container._gdx_out)
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
            job.run(
                gams_options=self.options,
                checkpoint=self.container._save_to,
                create_out_db=False,
                output=self.output,
            )
            if not self.is_async() and self.model:
                self.model._update_model_attributes()
        except GamsExceptionExecution as exception:
            if self.container._debugging_level == "keep_on_error":
                self.container.workspace._debug = DebugLevel.KeepFiles

            message = customize_exception(
                self.container.workspace, self.options, job, exception
            )
            raise GamspyException(message) from exception
        finally:
            self.container._unsaved_statements = []
            self.container._delete_autogenerated_symbols()

    def postprocess(self, dirty_names: list[str], is_implicit: bool = False):
        self.container._load_records_from_gdx(
            self.container._gdx_out,
            dirty_names + self.container._import_symbols,
        )
        self.container._swap_checkpoints()

        if self.options.traceopt == 3 and not is_implicit:
            return self.prepare_summary(
                self.container.working_directory, self.options.trace
            )

        return None
