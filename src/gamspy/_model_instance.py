from __future__ import annotations

import logging
import os
import sys
import time
import weakref
from collections.abc import Iterable
from typing import TYPE_CHECKING, TextIO, cast

import pandas as pd
from gams.core.cfg import GMS_SSSIZE
from gams.core.gev import (
    gevCreateD,
    gevFree,
    gevGetLShandle,
    gevGetStrOpt,
    gevHandleToPtr,
    gevInitEnvironmentLegacy,
    gevNameLogFile,
    gevNameStaFile,
    gevRestoreLogStat,
    gevRestoreLogStatRewrite,
    gevSwitchLogStat,
    new_gevHandle_tp,
)
from gams.core.gmd import (
    GMD_NRUELS,
    gmdCallSolver,
    gmdCloseLicenseSession,
    gmdInfo,
    gmdInitFromDict,
    gmdInitUpdate,
    gmdUpdateModelSymbol,
)
from gams.core.gmo import (
    gmoCreateD,
    gmoFree,
    gmoGetHeadnTail,
    gmoHandleToPtr,
    gmoHdomused,
    gmoHetalg,
    gmoHiterused,
    gmoHmarginals,
    gmoHobjval,
    gmoHresused,
    gmoLoadDataLegacy,
    gmoModelStat,
    gmoNameOptFileSet,
    gmoOptFileSet,
    gmoRegisterEnvironment,
    gmoSolveStat,
    gmoTmipbest,
    gmoTmipnod,
    new_gmoHandle_tp,
)

import gamspy as gp
import gamspy._symbols.implicits as implicits
import gamspy.utils as utils
from gamspy._backend.backend import backend_factory
from gamspy._database import Database, GamsEquation, GamsParameter, GamsVariable
from gamspy._gmd import (
    get_records,
    get_variable_equation_names,
    update_parameter_records,
)
from gamspy._internals import ATTR_PREFIX
from gamspy.exceptions import GamspyException, ValidationError

if TYPE_CHECKING:
    from pathlib import Path

    from gamspy import Container, Model, Parameter, Variable
    from gamspy._options import FreezeOptions, Options
    from gamspy._symbols.implicits import ImplicitParameter

logger = logging.getLogger("FROZEN MODEL")
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(name)s - %(levelname)s] %(message)s")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


VARIABLE_MAP = {
    "binary": 1,
    "integer": 2,
    "positive": 3,
    "negative": 4,
    "free": 5,
    "sos1": 6,
    "sos2": 7,
    "semicont": 8,
    "semiint": 9,
}


EQUATION_MAP = {
    "eq": 0,
    "geq": 1,
    "leq": 2,
    "nonbinding": 3,
    "external": 4,
}

UPDATE_ACTION_MAP = {
    "up": 1,
    "lo": 2,
    "fx": 3,
    "l": 4,
    "m": 5,
}

UPDATE_TYPE_MAP = {
    "0": 0,
    "base_case": 1,
    "accumulate": 2,
    "inherit": 3,
}


class GamsModifier:
    def __init__(
        self,
        gams_symbol: GamsParameter | GamsVariable | GamsEquation,
        update_action: int | None = None,
        data_symbol: GamsParameter | None = None,
        update_type: int = 3,
    ):
        self.update_action = None
        self.gams_symbol = gams_symbol
        self.update_type = update_type
        self.data_symbol = None

        # update_action and data_symbol specified
        if update_action is not None and data_symbol is not None:
            self._validate_update_action(update_action)

            self.update_action = update_action
            self.data_symbol = data_symbol
        # only the gams_symbol is specified
        elif update_action is None and data_symbol is None:
            ...
        else:
            raise GamspyException(
                "Wrong combination of parameters. Specifying only update_action or data_symbol is not allowed."
            )

    def _validate_update_action(self, update_action: int):
        if update_action in (1, 2, 3):
            if not isinstance(self.gams_symbol, GamsVariable):
                raise GamspyException(
                    f"GAMS Symbol must be GAMSVariable for {update_action}"
                )
        elif update_action in (4, 5):
            if not (isinstance(self.gams_symbol, (GamsVariable, GamsEquation))):
                raise GamspyException(
                    f"GAMS Symbol must be GAMSVariable or GAMSEquation for {update_action}"
                )
        else:
            raise GamspyException(f"Unknown update action {update_action}")


class ModelInstance:
    """
    ModelInstance class provides a controlled way of modifying a model instance
    and solving the resulting problem in the most efficient way, by communicating
    only the changes of the model to the solver and doing a hot start (in case of
    a continuous model like LP) without the use of disk IO.

    Parameters
    ----------
    container : Container
    model : Model
    modifiables : list[Parameter  |  ImplicitParameter]
    freeze_options : Options | None, optional
    """

    def __init__(
        self,
        container: Container,
        model: Model,
        modifiables: list[Parameter | ImplicitParameter],
        freeze_options: Options,
        output: TextIO | None,
    ):
        self.container = container
        self.job_name = container._job
        self.gms_file = self.job_name + ".gms"
        self.lst_file = self.job_name + ".lst"
        self.pf_file = self.job_name + ".pf"
        self.trace_file = self.job_name + ".txt"
        self.solver_control_file = os.path.join(
            self.container._process_directory, "gamscntr.dat"
        )

        self.model = model
        self.output = output

        self.modifiables = self._init_modifiables(modifiables)

        # Names of the symbols whose values a frozen solve changes (variables
        # and equations). Populated once in `instantiate` and used to restrict
        # the post-solve read-back in `_update_main_container`.
        self._solution_symbols: list[str] = []

        self.workspace = container._workspace
        self.sync_db = Database(self.workspace)

        self._gev = new_gevHandle_tp()
        ret = gevCreateD(self._gev, container.system_directory, GMS_SSSIZE)
        if not ret[0]:
            raise GamspyException(ret[1])

        self._gmo = new_gmoHandle_tp()
        ret = gmoCreateD(self._gmo, container.system_directory, GMS_SSSIZE)
        if not ret[0]:
            raise GamspyException(ret[1])

        self.modifiers = self._create_modifiers()
        self.instantiate(model, freeze_options)

        # preallocate summary frame for performance reasons
        HEADER = [
            "Solver Status",
            "Model Status",
            "Objective",
            "Num of Equations",
            "Num of Variables",
            "Model Type",
            "Solver",
            "Solver Time",
        ]
        self.summary = pd.DataFrame(index=range(1), columns=HEADER)

        weakref.finalize(self, self.cleanup, self._gmo, self._gev)

    @staticmethod
    def cleanup(gmo, gev) -> None:
        gmoFree(gmo)
        gevFree(gev)

    def close_license_session(self) -> None:
        gmdCloseLicenseSession(self.sync_db.gmd)

    def _create_modifiers(self) -> list[GamsModifier]:
        modifiers = []

        for symbol in self.modifiables:
            if isinstance(symbol, gp.Parameter):
                modifiers.append(
                    GamsModifier(
                        self.sync_db.add_parameter(
                            symbol.name,
                            symbol.dimension,
                            symbol.description,
                        )
                    )
                )

            elif isinstance(symbol, implicits.ImplicitParameter):
                attribute = symbol.name.split(".")[-1]
                update_action = UPDATE_ACTION_MAP[attribute]

                try:
                    sync_db_symbol = self.sync_db[symbol.parent.name]
                except KeyError:
                    if isinstance(symbol.parent, gp.Variable):
                        sync_db_symbol = self.sync_db.add_variable(
                            symbol.parent.name,
                            symbol.parent.dimension,
                            VARIABLE_MAP[symbol.parent.type],
                        )

                    elif isinstance(symbol.parent, gp.Equation):
                        sync_db_symbol = self.sync_db.add_equation(
                            symbol.parent.name,
                            symbol.parent.dimension,
                            EQUATION_MAP[symbol.parent.type],
                        )

                attr_name = "_".join(symbol.name.split("."))

                data_symbol = self.sync_db.add_parameter(
                    attr_name,
                    symbol.parent.dimension,
                )
                modifiers.append(
                    GamsModifier(sync_db_symbol, update_action, data_symbol)
                )
            else:
                raise ValidationError(
                    f"Symbol type {type(symbol)} cannot be modified in a frozen solve"
                )

        return modifiers

    def instantiate(self, model: Model, options: Options) -> None:
        # Check the gmd state.
        rc, _, _, _ = gmdInfo(self.sync_db.gmd, GMD_NRUELS)
        self.sync_db._check_for_gmd_error(rc, self.workspace)

        # Prepare the required lines to solve with model instance.
        model._add_runtime_options(options)
        scenario_str = self._get_scenario(model)
        self.container._add_statement(scenario_str)
        model._assign_model_attributes()

        # Write pf file
        options._set_extra_options({"solvercntr": self.solver_control_file})
        options.log_file = os.path.join(self.container.working_directory, "gamslog.dat")

        runner = backend_factory(
            self.container,
            options,
            self.model._default_solver,
            model=model,
            output=self.output,
        )
        runner.run()

        # Init environments
        if gevInitEnvironmentLegacy(self._gev, self.solver_control_file) != 0:
            raise GamspyException("Could not initialize model instance")

        gmoRegisterEnvironment(self._gmo, gevHandleToPtr(self._gev))
        ret = gmoLoadDataLegacy(self._gmo)
        if ret[0] != 0:
            raise GamspyException(f"Could not load model instance: {ret[1]}")

        rc = gmdInitFromDict(self.sync_db.gmd, gmoHandleToPtr(self._gmo))
        self.sync_db._check_for_gmd_error(rc, self.workspace)

        # A frozen solve only changes variable/equation values (and the
        # objective, which is an autogenerated variable/equation). Cache their
        # names once so each solve reads back only what actually changes.
        self._solution_symbols = get_variable_equation_names(self.sync_db.gmd)

    def solve(
        self,
        solver: str,
        instance_options: FreezeOptions,
        solver_options: dict | Path | None,
        output: TextIO | None,
    ) -> pd.DataFrame:
        # write solver options file
        option_file = 1 if solver_options else 0

        for symbol in self.modifiables:
            if isinstance(symbol, gp.Parameter):
                update_parameter_records(
                    self.container._gams2np,
                    self.sync_db.gmd,
                    self.sync_db[symbol.name].sym_ptr,
                    self.container[symbol.name].records,
                    symbol.dimension,
                    eps_to_zero=False,
                )

            if (
                isinstance(symbol, implicits.ImplicitParameter)
                and symbol.parent.records is not None
            ):
                parent_name, attr = symbol.name.split(".")
                attr_name = "_".join([parent_name, attr])

                columns = self._get_columns_to_drop(attr)

                update_parameter_records(
                    self.container._gams2np,
                    self.sync_db.gmd,
                    self.sync_db[attr_name].sym_ptr,
                    symbol.parent.records.drop(columns, axis=1),
                    symbol.parent.dimension,
                    eps_to_zero=False,
                )

        ### Legacy code from GAMS Control. TODO: Pay the technical debt of the following legacy code. ###
        rc = gmdInitUpdate(self.sync_db.gmd, gmoHandleToPtr(self._gmo))
        self.sync_db._check_for_gmd_error(rc, self.workspace)

        # Update gmd
        start = time.perf_counter()
        accumulate_no_match_cnt = 0
        no_match_cnt = 0

        for modifier in self.modifiers:
            update_type = UPDATE_TYPE_MAP[instance_options.update_type]
            if modifier.update_type != 3:
                update_type = modifier.update_type

            if isinstance(modifier.gams_symbol, GamsParameter):
                rc, no_match_cnt = gmdUpdateModelSymbol(
                    self.sync_db.gmd,
                    modifier.gams_symbol.sym_ptr,
                    0,
                    modifier.gams_symbol.sym_ptr,
                    update_type,
                    no_match_cnt,
                )
                self.sync_db._check_for_gmd_error(rc, self.workspace)
            else:
                rc, no_match_cnt = gmdUpdateModelSymbol(
                    self.sync_db.gmd,
                    modifier.gams_symbol.sym_ptr,
                    modifier.update_action,
                    modifier.data_symbol.sym_ptr,  # type: ignore
                    update_type,
                    no_match_cnt,
                )
                self.sync_db._check_for_gmd_error(rc, self.workspace)

            accumulate_no_match_cnt += no_match_cnt
            if accumulate_no_match_cnt > instance_options.no_match_limit:
                raise GamspyException(
                    f"Unmatched record limit exceeded while processing modifier {modifier.gams_symbol.name}, for more info check no_match_limit option."
                )

        model_generation_time = time.perf_counter() - start

        # Close Log and status file and remove
        if output:
            gevSwitchLogStat(self._gev, 0, "", False, "", False, None, None, None)
            ls_handle = gevGetLShandle(self._gev)
            gevRestoreLogStatRewrite(self._gev, ls_handle)

        if output == sys.stdout:
            gevSwitchLogStat(
                self._gev,
                3,
                gevGetStrOpt(self._gev, gevNameLogFile),
                False,
                gevGetStrOpt(self._gev, gevNameStaFile),
                False,
                None,
                None,
                ls_handle,
            )
            ls_handle = gevGetLShandle(self._gev)

        if instance_options is not None and instance_options.debug:
            with open(
                os.path.join(self.workspace.working_directory, "convert.opt"),
                "w",
            ) as opt_file:
                opt_file.writelines(
                    [
                        "gams "
                        + os.path.join(self.workspace.working_directory, "gams.gms"),
                        "dumpgdx "
                        + os.path.join(self.workspace.working_directory, "dump.gdx\n"),
                        "dictmap "
                        + os.path.join(self.workspace.working_directory, "dictmap.gdx"),
                    ]
                )

                gmoOptFileSet(self._gmo, 1)
                gmoNameOptFileSet(
                    self._gmo,
                    os.path.join(self.workspace.working_directory, "convert.opt"),
                )
                rc = gmdCallSolver(self.sync_db.gmd, "convert")
                self.sync_db._check_for_gmd_error(rc, self.workspace)

        gmoOptFileSet(self._gmo, option_file)
        gmoNameOptFileSet(
            self._gmo,
            os.path.join(self.workspace.working_directory, solver.lower() + ".opt"),
        )

        rc = gmdCallSolver(self.sync_db.gmd, solver)
        self.sync_db._check_for_gmd_error(rc, self.workspace)

        if output == sys.stdout:
            gevRestoreLogStat(self._gev, ls_handle)

        if output is not None and output != sys.stdout:
            gevSwitchLogStat(self._gev, 0, "", False, "", False, None, None, ls_handle)
            ls_handle = gevGetLShandle(self._gev)
            with open(gevGetStrOpt(self._gev, gevNameLogFile)) as file:
                for line in file.readlines():
                    output.write(line)
                gevRestoreLogStat(self._gev, ls_handle)
        ### end of the legacy code ###

        self._update_main_container()

        # update model attributes
        from gamspy._model import INTERRUPT_STATUS

        self.model._status = gp.ModelStatus(gmoModelStat(self._gmo))
        self.model._solve_status = gp.SolveStatus(gmoSolveStat(self._gmo))
        if self.model._solve_status in INTERRUPT_STATUS:
            logger.warning(
                f"The solve was interrupted! Solve status: {self.model._solve_status.name}. "
                "For further information, see https://gamspy.readthedocs.io/en/latest/reference/gamspy._model.html#gamspy.SolveStatus."
            )
        self.model._model_generation_time = model_generation_time
        self.model._solve_model_time = gmoGetHeadnTail(self._gmo, gmoHresused)
        self.model._num_iterations = gmoGetHeadnTail(self._gmo, gmoHiterused)
        self.model._marginals = gmoGetHeadnTail(self._gmo, gmoHmarginals)
        self.model._algorithm_time = gmoGetHeadnTail(self._gmo, gmoHetalg)
        self.model._objective_estimation = gmoGetHeadnTail(self._gmo, gmoTmipbest)
        self.model._num_nodes_used = gmoGetHeadnTail(self._gmo, gmoTmipnod)
        self.model._num_domain_violations = gmoGetHeadnTail(self._gmo, gmoHdomused)
        self.model._objective_value = gmoGetHeadnTail(self._gmo, gmoHobjval)
        self.summary.loc[0] = [
            str(self.model._solve_status),
            str(self.model._status),
            self.model._objective_value,
            self.model.num_equations,
            self.model.num_variables,
            self.model.used_model_type,
            solver,
            self.model._solve_model_time,
        ]

        return self.summary

    def _get_scenario(self, model: Model) -> str:
        auto_id = "s" + utils._get_unique_name()[:5]
        params = [
            modifier.gams_symbol
            for modifier in self.modifiers
            if isinstance(modifier.gams_symbol, GamsParameter)
        ]
        lines = []
        if params:
            lines.append(f"Set {ATTR_PREFIX}{auto_id}__(*) /'s0'/;")
            for symbol in params:
                declaration = f"Parameter {ATTR_PREFIX}{auto_id}__{symbol.name}({ATTR_PREFIX}{auto_id}__"
                domain = ""
                if symbol.dimension:
                    domain = "," + ",".join("*" * symbol.dimension)
                domain += ")"

                declaration = f"{declaration}{domain};"
                lines.append(declaration)

                domain = f"({ATTR_PREFIX}{auto_id}__"

                if symbol.dimension:
                    domain += ","

                assign_str = (
                    f"{ATTR_PREFIX}{auto_id}__{symbol.name}({ATTR_PREFIX}{auto_id}__"
                )
                if symbol.dimension:
                    assign_str += "," + ",".join(
                        [f"{ATTR_PREFIX}{auto_id}__"] * symbol.dimension
                    )

                assign_str += ") = Eps;"
                lines.append(assign_str)

            scenario = f"Set {ATTR_PREFIX}{auto_id}_dict(*,*,*) / '{ATTR_PREFIX}{auto_id}__'.'scenario'.''"
            for symbol in params:
                scenario += f",\n'{symbol.name}'.'param'.'{ATTR_PREFIX}{auto_id}__{symbol.name}'"
            scenario += "/;"
            lines.append(scenario)

        lines.append(f"{model.name}.justScrDir = 1;")
        solve_string = model._generate_solve_string()

        if params:
            solve_string += f" scenario {ATTR_PREFIX}{auto_id}_dict"

        solve_string += ";"
        lines.append(solve_string)

        return "\n".join(lines)

    def _init_modifiables(
        self, modifiables: list[Parameter | ImplicitParameter]
    ) -> list[Parameter | ImplicitParameter]:
        if not isinstance(modifiables, Iterable):
            raise ValidationError(
                "Modifiables must be iterable (i.e. list, tuple etc.)."
            )

        if any(
            not isinstance(symbol, (gp.Parameter, implicits.ImplicitParameter))
            for symbol in modifiables
        ):
            raise ValidationError(
                "Type of a modifiable must be either Parameter or a Variable attribute (e.g. variable.up)"
            )

        symbols_in_conditions: list[str] = []
        for equation in self.model.equations:
            assert equation._definition is not None
            symbols_in_conditions += equation._definition._find_symbols_in_conditions()

        will_be_modified: list[Parameter | ImplicitParameter] = []
        for symbol in modifiables:
            if isinstance(symbol, implicits.ImplicitParameter):
                if symbol.parent.name in symbols_in_conditions:
                    raise ValidationError(
                        f"Modifiable symbol `{symbol.parent.name}` cannot be in a condition."
                    )

                attr_name = symbol.name.split(".")[-1]

                # If the symbol attr is fx, then modify level, lower and upper.
                if attr_name == "fx":
                    if not utils.isin(symbol.parent.l, will_be_modified):
                        will_be_modified.append(symbol.parent.l)

                    if not utils.isin(symbol.parent.lo, will_be_modified):
                        will_be_modified.append(symbol.parent.lo)

                    if not utils.isin(symbol.parent.up, will_be_modified):
                        will_be_modified.append(symbol.parent.up)
                else:
                    # if fx already added level, lower or upper, do not add again.
                    if not utils.isin(symbol, will_be_modified):
                        will_be_modified.append(symbol)
            else:
                if symbol.name in symbols_in_conditions:
                    raise ValidationError(
                        f"Modifiable symbol `{symbol.name}` cannot be in a condition."
                    )
                will_be_modified.append(symbol)

        return will_be_modified

    def _prepare_hidden_options(self) -> dict:
        scrdir = self.container._process_directory
        hidden_options = {
            "trace": self.trace_file,
            "input": self.gms_file,
            "output": self.lst_file,
            "optdir": self.container.working_directory,
            "sysdir": self.container.system_directory,
            "scrdir": scrdir,
            "scriptnext": os.path.join(scrdir, "gamsnext.sh"),
            "license": self.container._license_path,
            "solvercntr": self.solver_control_file,
        }
        if self.container._network_license:
            hidden_options["netlicense"] = os.path.join(
                self.container._process_directory, "gamslice.dat"
            )

        return hidden_options

    def _get_columns_to_drop(self, attr: str) -> list[str]:
        attr_map = {
            "l": "level",
            "m": "marginal",
            "lo": "lower",
            "up": "upper",
            "scale": "scale",
        }

        columns = []
        for key, value in attr_map.items():
            if key != attr:
                columns.append(value)

        return columns

    def _update_main_container(self) -> None:
        records_dict = get_records(
            self.container._gams2np,
            self.sync_db.gmd,
            symbols=self._solution_symbols,
        )

        prev_state = self.container._options.miro_protect
        modifiable_names = {symbol.name for symbol in self.modifiables}

        # Refresh the symbols a solve changes (variables and equations).
        for name, records in records_dict.items():
            if name in self.container._data:
                self.container._options.miro_protect = False
                symbol = self.container._data[name]
                symbol._records = records
                symbol._should_unload_to_gams = False
                symbol._should_load_from_gams = False
                symbol.domain_labels = symbol.domain_names

        # Maintain the `<name>_var` companion for each modifiable parameter.
        for name in modifiable_names:
            if name not in self.container._data:
                # e.g. attribute modifiers like `x.fx` have no own symbol.
                continue

            generated_var = name + "_var"
            generated_records = records_dict.get(generated_var)

            if generated_var not in self.container._data:
                _ = gp.Variable(
                    self.container,
                    generated_var,
                    domain=self.container._data[name].domain,
                    records=generated_records,
                )
            else:
                symbol = cast("Variable", self.container[generated_var])
                symbol._records = generated_records

        self.container._options.miro_protect = prev_state

        if self.model._objective_variable is not None:
            obj_name = self.model._objective_variable.name
            if obj_name in records_dict:
                obj_records = records_dict[obj_name]
                if not obj_records.empty and "level" in obj_records.columns:
                    self.model._objective_value = float(obj_records["level"].iloc[0])
