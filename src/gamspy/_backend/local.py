from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING

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
        super().__init__(
            container, model, container._gdx_in, container._gdx_out
        )
        self.options = options
        self.output = output

    def _prepare_extra_options(self, job_name: str) -> dict:
        trace_file_path = os.path.join(
            self.container.working_directory, "trace.txt"
        )
        scrdir = os.path.join(self.container.working_directory, "225a")

        extra_options = {
            "trace": trace_file_path,
            "input": job_name + ".gms",
            "sysdir": self.container.system_directory,
            "scrdir": scrdir,
            "scriptnext": os.path.join(scrdir, "gamsnext.sh"),
        }

        if self.container._network_license:
            extra_options["netlicense"] = os.path.join(scrdir, "gamslice.dat")

        return extra_options

    def is_async(self):
        return False

    def solve(self, keep_flags: bool = False):
        # Generate gams string and write modified symbols to gdx
        gams_string, dirty_names = self.preprocess(keep_flags)

        # Run the model
        self.run(gams_string)

        # Synchronize GAMSPy with checkpoint and return a summary
        summary = self.postprocess(dirty_names)

        return summary

    def run(self, gams_string: str):
        job_id = "_" + str(uuid.uuid4())
        job_name = os.path.join(self.container.working_directory, job_id)

        # Write gms file
        with open(job_name + ".gms", "w") as gams_file:
            gams_file.write(gams_string)

        # Write pf file
        extra_options = self._prepare_extra_options(job_name)
        self.options._set_extra_options(extra_options)

        pf_file = os.path.join(
            self.container.working_directory, job_name + ".pf"
        )
        self.options.export(pf_file, self.output)

        try:
            self.container._job = job_name
            self.container._send_job(job_name, pf_file, self.output)

            if not self.is_async() and self.model:
                self.model._update_model_attributes()
        except GamspyException as exception:
            self.container.workspace._has_error = True
            message = customize_exception(
                self.container.working_directory,
                self.options,
                job_name,
                exception.rc,
            )

            exception.args = (exception.message + message,)
            raise exception
        finally:
            self.container._unsaved_statements = []
            self.container._delete_autogenerated_symbols()

    def postprocess(self, dirty_names: list[str]):
        symbols = dirty_names + self.container._import_symbols

        if len(symbols) != 0:
            self.container._load_records_from_gdx(
                self.container._gdx_out, symbols
            )

        miro.load_miro_symbol_records(self.container)

        if self.model is not None:
            return self.prepare_summary(self.container.working_directory)

        return None
