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

from gams import GamsJob
from gams import GamsOptions
from gams.control.workspace import GamsExceptionExecution

import gamspy._backend.backend as backend
from gamspy.exceptions import customize_exception

if TYPE_CHECKING:
    import io
    from gamspy import Container


class Local(backend.Backend):
    def __init__(
        self,
        container: "Container",
        options: "GamsOptions",
        output: Optional[io.TextIOWrapper] = None,
    ) -> None:
        super().__init__(container, container._gdx_in, container._gdx_out)
        self.options = options
        self.output = output

    def is_async(self):
        return False

    def solve(self, is_implicit: bool = False, keep_flags: bool = False):
        # Generate gams string and write modified symbols to gdx
        gams_string, dirty_names, modified_names = self.preprocess()

        # Run the model
        self.run(gams_string)

        if self.is_async():
            return None

        # Synchronize GAMSPy with checkpoint and return a summary
        summary = self.postprocess(
            dirty_names, modified_names, is_implicit, keep_flags
        )

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
        except GamsExceptionExecution as exception:
            exception = customize_exception(
                self.container.workspace, self.options, job, exception
            )
            raise exception
        finally:
            self.container._unsaved_statements = []

    def postprocess(
        self,
        dirty_names: List[str],
        modified_names: List[str],
        is_implicit: bool = False,
        keep_flags: bool = False,
    ):
        self.container.loadRecordsFromGdx(
            self.container._gdx_out,
            dirty_names + self.container._import_symbols,
        )
        self.container._swap_checkpoints()
        if not keep_flags:
            self.update_modified_state(modified_names)

        if self.options.traceopt == 3 and not is_implicit:
            return self.prepare_summary(
                self.container.working_directory, self.options.trace
            )

        return None
