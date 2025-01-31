from __future__ import annotations

import io
import os
import sys
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING

import gams.transfer as gt
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
    gmoNameOptFile,
    gmoNameOptFileSet,
    gmoOptFile,
    gmoOptFileSet,
    gmoRegisterEnvironment,
    gmoSolveStat,
    new_gmoHandle_tp,
)

import gamspy as gp
import gamspy._symbols.implicits as implicits
import gamspy.utils as utils
from gamspy._database import (
    Database,
    GamsEquation,
    GamsParameter,
    GamsVariable,
)
from gamspy._options import ModelInstanceOptions, Options
from gamspy.exceptions import GamspyException, ValidationError

if TYPE_CHECKING:
    from gamspy import Container, Model, Parameter
    from gamspy._symbols.implicits import ImplicitParameter


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
            if not (
                isinstance(self.gams_symbol, (GamsVariable, GamsEquation))
            ):
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
    freeze_options : Options | dict | None, optional
    """

    def __init__(
        self,
        container: Container,
        model: Model,
        modifiables: list[Parameter | ImplicitParameter],
        freeze_options: Options,
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
        assert self.model._is_frozen

        self.modifiables = self._init_modifiables(modifiables)
        self.instance_container = gt.Container(
            system_directory=container.system_directory,
        )

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

    def __del__(self):
        if hasattr(self, "_gmo") and self._gmo:
            gmoFree(self._gmo)

        if (
            hasattr(self, "_gev")
            and self._gev
            and gevHandleToPtr(self._gev) is not None
        ):
            gevFree(self._gev)

    def _create_modifiers(self) -> list[GamsModifier]:
        modifiers = []

        for symbol in self.modifiables:
            if isinstance(symbol, gp.Parameter):
                domain = ["*"] * symbol.dimension
                _ = gt.Parameter(self.instance_container, symbol.name, domain)
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

                domain = ["*"] * symbol.parent.dimension
                _ = gt.Parameter(
                    self.instance_container,
                    attr_name,
                    domain=domain,
                )

                data_symbol = self.sync_db.add_parameter(
                    attr_name,
                    symbol.parent.dimension,
                )
                modifiers.append(
                    GamsModifier(sync_db_symbol, update_action, data_symbol)
                )
            else:
                raise ValidationError(
                    f"Symbol type {type(symbol)} cannot be modified in a"
                    " frozen solve"
                )

        return modifiers

    def instantiate(self, model: Model, options: Options) -> None:
        # Check the gmd state.
        rc, _, _, _ = gmdInfo(self.sync_db.gmd, GMD_NRUELS)
        self.sync_db._check_for_gmd_error(rc)

        # Prepare the required lines to solve with model instance
        scenario_str = self._get_scenario(model)
        with open(self.gms_file, "w", encoding="utf-8") as gams_file:
            gams_file.write(scenario_str)

        # Write pf file
        extra_options = self._prepare_gams_options()
        options._set_extra_options(extra_options)
        options.log_file = os.path.join(
            self.container.working_directory, "gamslog.dat"
        )
        options.export(self.pf_file)

        # Run
        try:
            self.container._job = self.job_name
            self.container._send_job(self.job_name, self.pf_file)
        except GamspyException as exception:
            self.container._workspace._has_error = True
            raise exception
        finally:
            self.container._unsaved_statements = []

        # Init environments
        if gevInitEnvironmentLegacy(self._gev, self.solver_control_file) != 0:
            raise GamspyException("Could not initialize model instance")

        gmoRegisterEnvironment(self._gmo, gevHandleToPtr(self._gev))
        ret = gmoLoadDataLegacy(self._gmo)
        if ret[0] != 0:
            raise GamspyException(f"Could not load model instance: {ret[1]}")

        rc = gmdInitFromDict(self.sync_db.gmd, gmoHandleToPtr(self._gmo))
        self.sync_db._check_for_gmd_error(rc)

    def solve(
        self,
        solver: str,
        instance_options: ModelInstanceOptions,
        solver_options: dict | None = None,
        output: io.TextIOWrapper | None = None,
    ) -> pd.DataFrame:
        # write solver options file
        solver_options_file_name = os.path.join(
            self.container.working_directory, f"{solver.lower()}.opt"
        )
        option_file = 0
        if solver_options:
            with open(
                solver_options_file_name, "w", encoding="utf-8"
            ) as solver_file:
                for key, value in solver_options.items():
                    solver_file.write(f"{key} {value}\n")

            option_file = 1

        for symbol in self.modifiables:
            if isinstance(symbol, gp.Parameter):
                self.instance_container[symbol.name].setRecords(
                    self.container[symbol.name].records
                )
                self.instance_container.write(
                    self.sync_db.gmd, symbols=[symbol.name], eps_to_zero=False
                )

            if (
                isinstance(symbol, implicits.ImplicitParameter)
                and symbol.parent.records is not None
            ):
                parent_name, attr = symbol.name.split(".")
                attr_name = "_".join([parent_name, attr])

                columns = self._get_columns_to_drop(attr)

                self.instance_container[attr_name].setRecords(
                    self.container[parent_name].records.drop(columns, axis=1)
                )

                self.instance_container.write(
                    self.sync_db.gmd, symbols=[attr_name], eps_to_zero=False
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
            gevSwitchLogStat(
                self._gev, 0, "", False, "", False, None, None, None
            )
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

        tmp_opt_file = gmoOptFile(self._gmo)
        save_name_opt_file = gmoNameOptFile(self._gmo)
        if instance_options is not None and option_file != 0:
            tmp_opt_file = option_file

        if instance_options is not None and instance_options.debug:
            with open(
                os.path.join(self.workspace.working_directory, "convert.opt"),
                "w",
            ) as opt_file:
                opt_file.writelines(
                    [
                        "gams "
                        + os.path.join(
                            self.workspace.working_directory, "gams.gms"
                        ),
                        "dumpgdx "
                        + os.path.join(
                            self.workspace.working_directory, "dump.gdx\n"
                        ),
                        "dictmap "
                        + os.path.join(
                            self.workspace.working_directory, "dictmap.gdx"
                        ),
                    ]
                )

                gmoOptFileSet(self._gmo, 1)
                gmoNameOptFileSet(
                    self._gmo,
                    os.path.join(
                        self.workspace.working_directory, "convert.opt"
                    ),
                )
                rc = gmdCallSolver(self.sync_db.gmd, "convert")
                self.sync_db._check_for_gmd_error(rc, self.workspace)

        gmoOptFileSet(self._gmo, tmp_opt_file)
        gmoNameOptFileSet(
            self._gmo,
            os.path.join(os.path.dirname(save_name_opt_file), solver + ".opt"),
        )

        rc = gmdCallSolver(self.sync_db.gmd, solver)
        self.sync_db._check_for_gmd_error(rc, self.workspace)

        if output == sys.stdout:
            gevRestoreLogStat(self._gev, ls_handle)

        if output is not None and output != sys.stdout:
            gevSwitchLogStat(
                self._gev, 0, "", False, "", False, None, None, ls_handle
            )
            ls_handle = gevGetLShandle(self._gev)
            with open(gevGetStrOpt(self._gev, gevNameLogFile)) as file:
                for line in file.readlines():
                    output.write(line)
                gevRestoreLogStat(self._gev, ls_handle)
        ### end of the legacy code ###

        self._update_main_container()

        # update model attributes
        self.model._status = gp.ModelStatus(gmoModelStat(self._gmo))
        self.model._solve_status = gp.SolveStatus(gmoSolveStat(self._gmo))
        self.model._model_generation_time = model_generation_time
        self.model._solve_model_time = gmoGetHeadnTail(self._gmo, gmoHresused)
        self.model._num_iterations = gmoGetHeadnTail(self._gmo, gmoHiterused)
        self.model._marginals = gmoGetHeadnTail(self._gmo, gmoHmarginals)
        self.model._algorithm_time = gmoGetHeadnTail(self._gmo, gmoHetalg)
        self.model._num_domain_violations = gmoGetHeadnTail(
            self._gmo, gmoHdomused
        )
        self.model._objective_value = gmoGetHeadnTail(self._gmo, gmoHobjval)
        HEADER = [
            "Solver Status",
            "Model Status",
            "Objective",
            "Solver",
            "Solver Time",
        ]
        summary = pd.DataFrame(
            [
                [
                    str(self.model._solve_status),
                    str(self.model._status),
                    self.model._objective_value,
                    solver,
                    self.model._solve_model_time,
                ]
            ],
            columns=HEADER,
        )

        return summary

    def _get_scenario(self, model: Model) -> str:
        params = [
            modifier.gams_symbol
            for modifier in self.modifiers
            if isinstance(modifier.gams_symbol, GamsParameter)
        ]
        lines = []
        if params:
            lines.append("Set s__(*) /'s0'/;")
            for symbol in params:
                declaration = f"Parameter s__{symbol.name}(s__"
                domain = ""
                if symbol.dimension:
                    domain = "," + ",".join("*" * symbol.dimension)
                domain += ")"

                declaration = f"{declaration}{domain};"
                lines.append(declaration)

                domain = "(s__"

                if symbol.dimension:
                    domain += ","

                assign_str = f"s__{symbol.name}(s__"
                if symbol.dimension:
                    assign_str += "," + ",".join(["s__"] * symbol.dimension)

                assign_str += ") = Eps;"
                lines.append(assign_str)

            scenario = "Set dict(*,*,*) / 's__'.'scenario'.''"
            for symbol in params:
                scenario += f",\n'{symbol.name}'.'param'.'s__{symbol.name}'"
            scenario += "/;"
            lines.append(scenario)

        lines.append(f"{model.name}.justScrDir = 1;")
        solve_string = f"solve {model.name} using {model.problem}"

        if model.problem not in (gp.Problem.MCP, gp.Problem.CNS):
            solve_string += f" {model.sense}"

        if model._objective_variable is not None:
            solve_string += f" {model._objective_variable.gamsRepr()}"

        if params:
            solve_string += " scenario dict"
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
            symbols_in_conditions += (
                equation._definition._find_symbols_in_conditions()
            )

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

    def _prepare_gams_options(self) -> dict:
        scrdir = self.container._process_directory
        extra_options = {
            "trace": self.trace_file,
            "input": self.gms_file,
            "output": self.lst_file,
            "optdir": self.container.working_directory,
            "sysdir": self.container.system_directory,
            "scrdir": scrdir,
            "scriptnext": os.path.join(scrdir, "gamsnext.sh"),
            "license": utils._get_license_path(
                self.container.system_directory
            ),
            "solvercntr": self.solver_control_file,
        }
        if self.container._network_license:
            extra_options["netlicense"] = os.path.join(
                self.container._process_directory, "gamslice.dat"
            )

        return extra_options

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
        temp = gt.Container(
            system_directory=self.container.system_directory,
        )
        temp.read(self.sync_db.gmd)

        prev_state = self.container._options.miro_protect
        for name in temp.data:
            if name in self.container.data:
                self.container._options.miro_protect = False
                self.container[name].records = temp[name].records
                self.container[name].domain_labels = self.container[
                    name
                ].domain_names

            if name in (symbol.name for symbol in self.modifiables):
                _ = gp.Variable(
                    self.container,
                    name + "_var",
                    domain=self.container[name].domain,
                    records=temp[name + "_var"].records,
                )
        self.container._options.miro_protect = prev_state

        if self.model._objective_variable is not None:
            self.model._objective_value = temp[
                self.model._objective_variable.name
            ].toValue()
