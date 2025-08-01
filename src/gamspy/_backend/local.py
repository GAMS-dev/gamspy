from __future__ import annotations

import os
from typing import TYPE_CHECKING

import gamspy._backend.backend as backend
import gamspy._miro as miro
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
        self.lst_file = self.job_name + ".lst"
        self.pf_file = self.job_name + ".pf"
        self.trace_file = self.job_name + ".txt"

    def _prepare_extra_options(self, gams_to_gamspy: bool) -> dict:
        scrdir = self.container._process_directory

        extra_options = {
            "trace": self.trace_file,
            "input": self.gms_file,
            "output": self.lst_file,
            "optdir": self.container.working_directory,
            "sysdir": self.container.system_directory,
            "scrdir": scrdir,
            "scriptnext": os.path.join(scrdir, "gamsnext.sh"),
            "license": self.container._license_path,
        }

        if gams_to_gamspy:
            extra_options["gdx"] = self.container._gdx_out
            extra_options["gdxSymbols"] = "newOrChanged"

        if self.container._network_license:
            extra_options["netlicense"] = os.path.join(scrdir, "gamslice.dat")

        return extra_options

    def is_async(self):
        return False

    def run(
        self,
        relaxed_domain_mapping: bool = False,
        gams_to_gamspy: bool = False,
    ):
        # Generate gams string and write modified symbols to gdx
        gams_string = self.preprocess(self.container._gdx_in)

        # Run the model
        self.execute_gams(gams_string, gams_to_gamspy)

        # Synchronize GAMSPy with checkpoint and return a summary
        summary = self.postprocess(relaxed_domain_mapping, gams_to_gamspy)

        return summary

    def execute_gams(self, gams_string: str, gams_to_gamspy: bool):
        # Write gms file
        with open(self.gms_file, "w", encoding="utf-8") as gams_file:
            gams_file.write(gams_string)

        # Write pf file
        extra_options = self._prepare_extra_options(gams_to_gamspy)
        self.options._set_extra_options(extra_options)
        self.options._export(self.pf_file, self.output)

        try:
            self.container._job = self.job_name
            self.container._send_job(self.job_name, self.pf_file, self.output)

            if not self.is_async() and self.model:
                self.model._update_model_attributes()
                self.container._delete_autogenerated_symbols()
        except GamspyException as exception:
            self.container._workspace._errors.append(str(exception))
            message = _customize_exception(
                self.options,
                self.job_name,
                exception.return_code,
            )

            exception.args = (exception.message + message,)
            raise exception
        finally:
            self.container._unsaved_statements = []

    def postprocess(self, relaxed_domain_mapping: bool, gams_to_gamspy: bool):
        if gams_to_gamspy:
            super().load_records(relaxed_domain_mapping)

        miro.load_miro_symbol_records(self.container)

        if self.model is not None:
            self.parse_listings()
            return self.prepare_summary(self.trace_file)

        return None
