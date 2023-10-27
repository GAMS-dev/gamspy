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

ACTIONS = Literal[
    "restart_after_solve",
    "compile_only",
    "execute_only",
    "compile_and_execute",
    "trace_report",
]

action_map = {
    "restart_after_solve": "R",
    "compile_only": "C",
    "execute_only": "E",
    "compile_and_execute": "CE",
    "trace_report": "GT",
}

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
    "action": "action",
    "append_output": "appendout",
    "report_async_solve": "asyncsollst",
    "basis_detection_threshold": "bratio",
    "compile_error_limit": "cerr",
    "decrypt_key": "decryptkey",
    "domain_violation_limit": "domlim",
    "encrypt_key": "encryptkey",
    "time_limit": "etlim",
    "finite_diff_step_size": "fddelta",
    "finite_diff_option": "fdopt",
    "heap_limit": "heaplimit",
    "hold_fixed_variables": "holdfixed",
    "hold_fixed_variables_async": "holdfixedasync",
    "int_variable_upper_bound": "intvarup",
    "iteration_limit": "iterlim",
    "job_trace": "jobtrace",
    "license": "license",
    "variable_column_limit": "limcol",
    "equation_row_limit": "limrow",
    "line_tracing_level": "logline",
    "max_execution_errors": "maxexecerror",
    "node_limit": "nodlim",
    "absolute_termination_tolerance": "optca",
    "relative_termination_tolerance": "optcr",
    "solver_options_directory": "optdir",
    "option_file": "optfile",
    "output_file_name": "output",
    "privacy_license": "plicense",
    "profile": "profile",
    "profile_file": "profilefile",
    "profile_tolerance": "profiletol",
    "reference_file": "reference",
    "reference_line_number": "referencelineno",
    "solver_time_limit": "reslim",
    "savepoint": "savepoint",
    "seed": "seed",
    "report_solution": "solprint",
    "solver_link_option": "solvelink",
    "multi_solve_strategy": "solveopt",
    "step_summary": "stepsum",
    "suppress_compiler_listing": "suppress",
    "symbol_table_file": "symbol",
    "report_solver_status": "sysout",
    "threads": "threads",
    "threads_async": "threadsasync",
    "instruction_time_threshold": "timer",
    "trace_file": "trace",
    "trace_level": "tracelevel",
    "trace_file_format": "traceopt",
    "num_warnings_limit": "warnings",
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
    action: Optional[ACTIONS] = None
    append_output: Optional[bool] = None
    report_async_solve: Optional[bool] = None
    basis_detection_threshold: Optional[float] = None
    compile_error_limit: Optional[int] = None
    decrypt_key: Optional[str] = None
    domain_violation_limit: Optional[int] = None
    encrypt_key: Optional[str] = None
    time_limit: Optional[float] = None
    finite_diff_step_size: Optional[float] = None
    finite_diff_option: Optional[
        Literal[0, 1, 2, 3, 4, 10, 11, 12, 13, 14]
    ] = None
    heap_limit: Optional[float] = None
    hold_fixed_variables: Optional[bool] = None
    hold_fixed_variables_async: Optional[bool] = None
    int_variable_upper_bound: Optional[int] = None
    iteration_limit: Optional[int] = None
    job_trace: Optional[str] = None
    license: Optional[str] = None
    variable_column_limit: Optional[int] = None
    equation_row_limit: Optional[int] = None
    line_tracing_level: Optional[Literal[0, 1, 2]] = None
    max_execution_errors: Optional[int] = None
    node_limit: Optional[int] = None
    absolute_termination_tolerance: Optional[float] = None
    relative_termination_tolerance: Optional[float] = None
    solver_options_directory: Optional[str] = None
    option_file: Optional[int] = None
    output_file_name: Optional[str] = None
    privacy_license: Optional[str] = None
    profile: Optional[int] = None
    profile_file: Optional[str] = None
    profile_tolerance: Optional[float] = None
    reference_file: Optional[str] = None
    reference_line_number: Optional[Literal["actual", "start"]] = None
    solver_time_limit: Optional[float] = None
    savepoint: Optional[Literal[0, 1, 2, 3, 4]] = None
    seed: Optional[int] = None
    report_solution: Optional[bool] = None
    solver_link_option: Optional[Literal[0, 1, 2, 3, 4, 5, 6, 7]] = None
    multi_solve_strategy: Optional[Literal["replace", "merge", "clear"]] = None
    step_summary: Optional[bool] = None
    suppress_compiler_listing: Optional[bool] = None
    symbol_table_file: Optional[str] = None
    report_solver_status: Optional[bool] = None
    threads: Optional[int] = None
    threads_async: Optional[int] = None
    instruction_time_threshold: Optional[int] = None
    trace_file: Optional[str] = None
    trace_level: Optional[int] = None
    trace_file_format: Optional[Literal[0, 1, 2, 3, 4, 5]] = None
    num_warnings_limit: Optional[int] = None
    zero_rounding_threshold: Optional[float] = None
    report_underflow: Optional[bool] = None

    @field_validator("action")
    @classmethod
    def validate_action(cls, action: str) -> str:
        return action_map[action]

    @field_validator("append_output")
    @classmethod
    def validate_append_output(cls, is_appending: bool) -> int:
        return 1 if is_appending else 0

    @field_validator("report_async_solve")
    @classmethod
    def validate_report_async_solve(cls, is_reporting: bool) -> int:
        return 1 if is_reporting else 0

    @field_validator("hold_fixed_variables")
    @classmethod
    def validate_hold_fixed_variables(cls, is_holding: bool) -> int:
        return 1 if is_holding else 0

    @field_validator("hold_fixed_variables_async")
    @classmethod
    def validate_hold_fixed_variables_async(cls, is_holding: bool) -> int:
        return 1 if is_holding else 0

    @field_validator("report_solution")
    @classmethod
    def validate_report_solution(cls, is_reporting: bool) -> int:
        return 1 if is_reporting else 0

    @field_validator("multi_solve_strategy")
    @classmethod
    def validate_multi_solve_strategy(cls, strategy: str) -> int:
        return multi_solve_map[strategy]

    @field_validator("step_summary")
    @classmethod
    def validate_step_summary(cls, is_summarizing: bool) -> int:
        return 1 if is_summarizing else 0

    @field_validator("suppress_compiler_listing")
    @classmethod
    def validate_suppress_compiler_listing(cls, is_surpressing: bool) -> int:
        return 1 if is_surpressing else 0

    @field_validator("report_solver_status")
    @classmethod
    def validate_report_solver_status(cls, is_reporting: bool) -> int:
        return 1 if is_reporting else 0

    @field_validator("report_underflow")
    @classmethod
    def validate_report_underflow(cls, is_reporting: bool) -> int:
        return 1 if is_reporting else 0
