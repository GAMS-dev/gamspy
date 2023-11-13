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
from typing import Literal
from typing import Optional

from pydantic import BaseModel
from pydantic import field_validator

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
    "gams_log_redirection": "_logoption",
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
    cns: Optional[str] = None
    dnlp: Optional[str] = None
    emp: Optional[str] = None
    lp: Optional[str] = None
    mcp: Optional[str] = None
    minlp: Optional[str] = None
    mip: Optional[str] = None
    miqcp: Optional[str] = None
    mpec: Optional[str] = None
    nlp: Optional[str] = None
    qcp: Optional[str] = None
    rminlp: Optional[str] = None
    rmip: Optional[str] = None
    rmiqcp: Optional[str] = None
    rmpec: Optional[str] = None
    allow_suffix_in_equation: Optional[bool] = None
    allow_suffix_in_limited_variables: Optional[bool] = None
    basis_detection_threshold: Optional[float] = None
    compile_error_limit: int = 1
    domain_violation_limit: Optional[int] = None
    gams_log_redirection: Literal[0, 2, 3, 4] = 3
    job_time_limit: Optional[float] = None
    job_heap_limit: Optional[float] = None
    hold_fixed_variables: Optional[bool] = None
    integer_variable_upper_bound: Optional[int] = None
    iteration_limit: Optional[int] = None
    keep_temporary_files: bool = False
    license: Optional[str] = None
    variable_listing_limit: Optional[int] = None
    equation_listing_limit: Optional[int] = None
    node_limit: Optional[int] = None
    absolute_optimality_gap: Optional[float] = None
    relative_optimality_gap: Optional[float] = None
    profile: Optional[int] = None
    profile_tolerance: Optional[float] = None
    time_limit: Optional[float] = None
    savepoint: Optional[Literal[0, 1, 2, 3, 4]] = None
    seed: Optional[int] = None
    report_solution: Literal[0, 1, 2] = 2
    show_os_memory: Literal[0, 1, 2] = 0
    solver_link_type: Optional[Literal[0, 1, 2, 3, 4, 5, 6, 7]] = None
    multi_solve_strategy: Optional[Literal["replace", "merge", "clear"]] = None
    step_summary: Optional[bool] = None
    suppress_compiler_listing: bool = True
    report_solver_status: Optional[bool] = None
    threads: Optional[int] = None
    trace_file: Optional[str] = None
    trace_level: Optional[int] = None
    trace_file_format: Optional[Literal[0, 1, 2, 3, 4, 5]] = None
    write_listing_file: bool = True
    zero_rounding_threshold: Optional[float] = None
    report_underflow: Optional[bool] = None

    class Config:
        extra = "forbid"

    @field_validator("allow_suffix_in_equation")
    @classmethod
    def validate_allow_suffix_in_equation(cls, is_allowing):
        return "on" if is_allowing else "off"

    @field_validator("allow_suffix_in_limited_variables")
    @classmethod
    def validate_allow_suffix_in_limited_variables(cls, is_allowing):
        return "on" if is_allowing else "off"

    @field_validator("hold_fixed_variables")
    @classmethod
    def validate_hold_fixed_variables(cls, is_holding: bool) -> int:
        return 1 if is_holding else 0

    @field_validator("keep_temporary_files")
    @classmethod
    def validate_keep_temporary_files(cls, is_keeping: bool) -> int:
        return 1 if is_keeping else 0

    @field_validator("multi_solve_strategy")
    @classmethod
    def validate_multi_solve_strategy(cls, strategy: str) -> int:
        return multi_solve_map[strategy]

    @field_validator("step_summary")
    @classmethod
    def validate_step_summary(cls, is_summarizing: bool) -> int:
        return 1 if is_summarizing else 0

    @field_validator("report_solver_status")
    @classmethod
    def validate_report_solver_status(cls, is_reporting: bool) -> int:
        return 1 if is_reporting else 0

    @field_validator("report_underflow")
    @classmethod
    def validate_report_underflow(cls, is_reporting: bool) -> int:
        return 1 if is_reporting else 0

    @field_validator("suppress_compiler_listing")
    @classmethod
    def validate_suppress_compiler_listing(cls, is_suppressing: bool) -> int:
        return 1 if is_suppressing else 0

    @field_validator("write_listing_file")
    @classmethod
    def validate_write_listing_file(cls, is_writing: bool) -> int:
        return 1 if is_writing else 0
