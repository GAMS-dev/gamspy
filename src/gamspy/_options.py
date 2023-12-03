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

from typing import Literal
from typing import TYPE_CHECKING

from gams import GamsOptions
from gams import GamsWorkspace
from pydantic import BaseModel

from gamspy.exceptions import GamspyException

if TYPE_CHECKING:
    import io

multi_solve_map = {"replace": 0, "merge": 1, "clear": 2}

# GAMSPy to GAMS Control mapping
option_map = {
    "cns": "cns",
    "dnlp": "dnlp",
    "emp": "emp",
    "lp": "lp",
    "mcp": "mcp",
    "minlp": "minlp",
    "mip": "mip",
    "miqcp": "miqcp",
    "mpec": "mpec",
    "nlp": "nlp",
    "qcp": "qcp",
    "rminlp": "rminlp",
    "rmip": "rmip",
    "rmiqcp": "rmiqcp",
    "rmpec": "rmpec",
    "allow_suffix_in_equation": "suffixalgebravars",
    "allow_suffix_in_limited_variables": "suffixdlvars",
    "basis_detection_threshold": "bratio",
    "compile_error_limit": "cerr",
    "domain_violation_limit": "domlim",
    "job_time_limit": "etlim",
    "job_heap_limit": "heaplimit",
    "hold_fixed_variables": "holdfixed",
    "hold_fixed_variables_async": "holdfixedasync",
    "integer_variable_upper_bound": "intvarup",
    "iteration_limit": "iterlim",
    "keep_temporary_files": "keep",
    "license": "license",
    "variable_listing_limit": "limcol",
    "equation_listing_limit": "limrow",
    "node_limit": "nodlim",
    "absolute_optimality_gap": "optca",
    "relative_optimality_gap": "optcr",
    "profile": "profile",
    "profile_tolerance": "profiletol",
    "time_limit": "reslim",
    "savepoint": "savepoint",
    "seed": "seed",
    "report_solution": "solprint",
    "show_os_memory": "showosmemory",
    "solver_link_type": "solvelink",
    "multi_solve_strategy": "solveopt",
    "step_summary": "stepsum",
    "suppress_compiler_listing": "suppress",
    "report_solver_status": "sysout",
    "threads": "threads",
    "trace_file": "trace",
    "trace_level": "tracelevel",
    "trace_file_format": "traceopt",
    "write_listing_file": "_writeoutput",
    "zero_rounding_threshold": "zerores",
    "report_underflow": "zeroresrep",
}


class Options(BaseModel):
    cns: str | None = None
    dnlp: str | None = None
    emp: str | None = None
    lp: str | None = None
    mcp: str | None = None
    minlp: str | None = None
    mip: str | None = None
    miqcp: str | None = None
    mpec: str | None = None
    nlp: str | None = None
    qcp: str | None = None
    rminlp: str | None = None
    rmip: str | None = None
    rmiqcp: str | None = None
    rmpec: str | None = None
    allow_suffix_in_equation: bool | None = None
    allow_suffix_in_limited_variables: bool | None = None
    basis_detection_threshold: float | None = None
    compile_error_limit: int = 1
    domain_violation_limit: int | None = None
    job_time_limit: float | None = None
    job_heap_limit: float | None = None
    hold_fixed_variables: bool | None = None
    integer_variable_upper_bound: int | None = None
    iteration_limit: int | None = None
    keep_temporary_files: bool = False
    license: str | None = None
    variable_listing_limit: int | None = None
    equation_listing_limit: int | None = None
    node_limit: int | None = None
    absolute_optimality_gap: float | None = None
    relative_optimality_gap: float | None = None
    profile: int | None = None
    profile_tolerance: float | None = None
    time_limit: float | None = None
    savepoint: Literal[0, 1, 2, 3, 4] | None = None
    seed: int | None = None
    report_solution: Literal[0, 1, 2] = 2
    show_os_memory: Literal[0, 1, 2] = 0
    solver_link_type: Literal[0, 1, 2, 3, 4, 5, 6, 7] | None = None
    multi_solve_strategy: Literal["replace", "merge", "clear"] | None = None
    step_summary: bool | None = None
    suppress_compiler_listing: bool = True
    report_solver_status: bool | None = None
    threads: int | None = None
    trace_file: str | None = None
    trace_level: int | None = None
    trace_file_format: Literal[0, 1, 2, 3, 4, 5] | None = None
    write_listing_file: bool = True
    zero_rounding_threshold: float | None = None
    report_underflow: bool | None = None

    class Config:
        extra = "forbid"

    def _getGamsCompatibleOptions(self):
        options_dict = self.model_dump()
        if options_dict["allow_suffix_in_equation"] is not None:
            allows_suffix = options_dict["allow_suffix_in_equation"]
            options_dict["allow_suffix_in_equation"] = (
                "on" if allows_suffix else "off"
            )

        if options_dict["allow_suffix_in_limited_variables"] is not None:
            allows_suffix = options_dict["allow_suffix_in_limited_variables"]
            options_dict["allow_suffix_in_limited_variables"] = (
                "on" if allows_suffix else "off"
            )

        if options_dict["multi_solve_strategy"] is not None:
            strategy = options_dict["multi_solve_strategy"]
            options_dict["multi_solve_strategy"] = multi_solve_map[strategy]

        options_dict = {
            option_map[key]: value for key, value in options_dict.items()  # type: ignore
        }

        return options_dict


def _fix_log_option(
    output: io.TextIOWrapper | None,
    create_log_file: bool,
    options: GamsOptions,
) -> GamsOptions:
    if output is None:
        if create_log_file:
            # Output = None & debug_logfile = True -> logOption = 2
            options._logoption = 2
        else:
            # Output = None & debug_logfile = False -> logOption = 0
            options._logoption = 0

    # Output = writer & debug_logfile = True -> logOption = 4
    # will be implemented once GAMS Control allows it
    if output is not None and create_log_file:
        ...

    return options


def _mapOptions(
    workspace: GamsWorkspace,
    options: Options | None,
    is_seedable: bool = True,
    output: io.TextIOWrapper | None = None,
    create_log_file: bool = False,
) -> GamsOptions:
    """
    Maps given GAMSPy options to GamsOptions

    Parameters
    ----------
    options : Options | None
        GAMSPy options
    is_seedable : bool, optional
        only seedable at first run or in model.solve function, by default True

    Returns
    -------
    GamsOptions

    Raises
    ------
    GamspyException
        when options is not type Options
    GamspyException
        when one of the option names is invalid
    """
    gams_options = GamsOptions(workspace)

    if options is not None:
        if not isinstance(options, Options):
            raise GamspyException(
                f"options must be of type Option but found {type(options)}"
            )

        options_dict = options._getGamsCompatibleOptions()

        for option, value in options_dict.items():
            if option not in option_map.values():
                raise GamspyException(
                    f"Invalid option `{option}`. Possible options:"
                    f" {option_map.keys()}"
                )

            if value is not None:
                if option == "seed" and not is_seedable:
                    continue
                setattr(gams_options, option.lower(), value)

    gams_options = _fix_log_option(output, create_log_file, gams_options)

    return gams_options
