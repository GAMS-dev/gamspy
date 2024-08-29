from __future__ import annotations

import os
from typing import TYPE_CHECKING

from gams import DebugLevel

import gamspy._backend.backend as backend
import gamspy._miro as miro
import gamspy.utils as utils
from gamspy.exceptions import GamspyException, _customize_exception

if TYPE_CHECKING:
    import io

    from gamspy import Container, Model, Options
    from gamspy._symbols.symbol import Symbol


class Local(backend.Backend):
    def __init__(
        self,
        container: Container,
        options: Options,
        solver: str | None,
        solver_options: dict | None,
        output: io.TextIOWrapper | None,
        model: Model | None,
        load_symbols: list[Symbol] | None,
    ) -> None:
        super().__init__(
            container,
            model,
            options,
            solver,
            solver_options,
            output,
            load_symbols,
        )
        self.job_name = self.get_job_name()
        self.gms_file = self.job_name + ".gms"
        self.pf_file = self.job_name + ".pf"
        self.trace_file = self.job_name + ".txt"

        if self.container._debugging_level == DebugLevel.KeepFiles:
            self.options.log_file = self.job_name + ".log"
            self.container._gdx_in = self.job_name + "in.gdx"
            self.container._gdx_out = self.job_name + "out.gdx"

    def _prepare_extra_options(self, job_name: str) -> dict:
        scrdir = self.container._process_directory

        extra_options = {
            "gdx": self.container._gdx_out,
            "trace": self.trace_file,
            "input": self.gms_file,
            "output": job_name + ".lst",
            "optdir": self.container.working_directory,
            "sysdir": self.container.system_directory,
            "scrdir": scrdir,
            "scriptnext": os.path.join(scrdir, "gamsnext.sh"),
            "license": utils._get_license_path(
                self.container.system_directory
            ),
        }

        if self.container._network_license:
            extra_options["netlicense"] = os.path.join(scrdir, "gamslice.dat")

        return extra_options

    def is_async(self):
        return False

    def run(self, keep_flags: bool = False):
        if self.model is not None:
            self.model._add_runtime_options(self.options)
            self.model._append_solve_string()
            self.model._create_model_attributes()
            self.options._set_solver_options(
                working_directory=self.container.working_directory,
                solver=self.solver,
                problem=self.model.problem,
                solver_options=self.solver_options,
            )

        # Generate gams string and write modified symbols to gdx
        gams_string = self.preprocess(self.container._gdx_in, keep_flags)

        # Run the model
        self.execute_gams(gams_string)

        # Synchronize GAMSPy with checkpoint and return a summary
        summary = self.postprocess()

        return summary

    def execute_gams(self, gams_string: str):
        # Write gms file
        with open(self.gms_file, "w", encoding="utf-8") as gams_file:
            gams_file.write(gams_string)

        # Write pf file
        extra_options = self._prepare_extra_options(self.job_name)
        self.options._set_extra_options(extra_options)

        self.options._export(self.pf_file, self.output)

        try:
            self.container._job = self.job_name
            self.container._send_job(self.job_name, self.pf_file, self.output)

            if not self.is_async() and self.model:
                self.model._update_model_attributes()
        except GamspyException as exception:
            self.container._workspace._has_error = True
            message = _customize_exception(
                self.options,
                self.job_name,
                exception.return_code,
            )

            exception.args = (exception.message + message,)
            raise exception
        finally:
            self.container._unsaved_statements = []
            self.container._delete_autogenerated_symbols()

    def postprocess(self):
        super().postprocess()
        miro.load_miro_symbol_records(self.container)

        if self.model is not None:
            listing_file = (
                self.options.listing_file
                if self.options.listing_file
                else self.job_name + ".lst"
            )
            if self.options.equation_listing_limit:
                utils._parse_generated_equations(self.model, listing_file)

            if self.options.variable_listing_limit:
                utils._parse_generated_variables(self.model, listing_file)

            return self.prepare_summary(self.trace_file)

        return None
