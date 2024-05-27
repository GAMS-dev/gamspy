from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING

from gams import GamsJob
from gams.control.workspace import GamsExceptionExecution

import gamspy._backend.backend as backend
import gamspy._miro as miro
from gamspy.exceptions import GamspyException, customize_exception

if TYPE_CHECKING:
    import io

    from gamspy import Container, Model, Options


class Local(backend.Backend):
    def __init__(
        self,
        container: Container,
        options: Options,
        output: io.TextIOWrapper | None = None,
        model: Model | None = None,
    ) -> None:
        super().__init__(container, container._gdx_in, container._gdx_out)
        if model is None:
            self.options = options._get_gams_options(self.container.workspace)
        else:
            self.options = options._get_gams_options(
                self.container.workspace, model.problem
            )
        self.options.license = self.container._license_path
        self.options.trace = os.path.join(
            self.container.workspace.working_directory, "trace.txt"
        )
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
            message = customize_exception(
                self.container.workspace, self.options, job, exception
            )
            raise GamspyException(message) from exception
        finally:
            self.container._unsaved_statements = []
            self.container._delete_autogenerated_symbols()

    def postprocess(self, dirty_names: list[str], is_implicit: bool = False):
        symbols = dirty_names + self.container._import_symbols
        if len(symbols) != 0:
            self.container._load_records_from_gdx(
                self.container._gdx_out, symbols
            )

        miro.load_miro_symbol_records(self.container)

        self.container._swap_checkpoints()

        if not is_implicit:
            return self.prepare_summary(
                self.container.working_directory, self.options.trace
            )

        return None
