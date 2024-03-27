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
"""Exception classes for GAMSPy"""

from __future__ import annotations

import os

from gams import GamsOptions, GamsWorkspace
from gams.control.workspace import GamsExceptionExecution


class GamspyException(Exception):
    """Plain Gamspy exception."""

    def __init__(self, message: str, return_code: int | None = None) -> None:
        self.message = message
        self.rc = return_code

    def __str__(self) -> str:
        return self.message


class NeosClientException(Exception):
    """NeosClient exception."""


class EngineClientException(Exception):
    """EngineClient exception"""


class NeosException(GamspyException):
    """NEOS Server execution exception"""


class EngineException(GamspyException):
    def __init__(
        self,
        message: str,
        return_code: int,
        status_code: int | None = None,
        gams_exit_code: int | None = None,
    ) -> None:
        self.message = message
        self.return_code = return_code
        self.status_code = status_code
        self.gams_exit_code = gams_exit_code

    def __str__(self) -> str:
        return self.message


class ValidationError(Exception):
    """An error while validating data."""


error_codes = {
    1: "Solver is to be called, the system should never return this number",
    2: "There was a compilation error",
    3: "There was an execution error",
    4: "System limits were reached",
    5: "There was a file error",
    6: "There was a parameter error",
    7: "There was a licensing error",
    8: "There was a GAMS system error",
    9: "GAMS could not be started",
    10: "Out of memory",
    11: "Out of disk",
    109: "Could not create process/scratch directory",
    110: "Too many process/scratch directories",
    112: "Could not delete the process/scratch directory",
    113: "Could not write the script gamsnext",
    114: "Could not write the parameter file",
    115: "Could not read environment variable",
    126: "Driver error: internal error: cannot load option handling library",
    144: "Could not spawn the GAMS language compiler (gamscmex)",
    145: "Current directory (curdir) does not exist",
    146: "Cannot set current directory (curdir)",
    148: "Blank in system directory (UNIX only)",
    149: "Blank in current directory (UNIX only)",
    150: "Blank in scratch extension (scrext)",
    151: "Unexpected cmexRC",
    152: "Could not find the process directory (procdir)",
    153: "CMEX library not be found (experimental)",
    154: "Entry point in CMEX library could not be found (experimental)",
    155: "Blank in process directory (UNIX only)",
    156: "Blank in scratch directory (UNIX only)",
    160: (
        "Driver error: internal error: GAMS compile and execute module not"
        " found"
    ),
    184: "Driver error: problems getting current directory",
    208: "Driver error: internal error: cannot install interrupt handler",
    232: "Driver error: incorrect command line parameters for gams",
    400: "Could not spawn the GAMS language compiler (gamscmex)",
    401: "Current directory (curdir) does not exist",
    402: "Cannot set current directory (curdir)",
    404: "Blank in system directory (UNIX only)",
    405: "Blank in current directory (UNIX only)",
    406: "Blank in scratch extension (scrext)",
    407: "Unexpected cmexRC",
    408: "Could not find the process directory (procdir)",
    409: "CMEX library not be found (experimental)",
    410: "Entry point in CMEX library could not be found (experimental)",
    411: "Blank in process directory (UNIX only)",
    412: "Blank in scratch directory (UNIX only)",
    909: (
        "Cannot add path / unknown UNIX environment / cannot set environment"
        " variable"
    ),
    1000: "Driver error: incorrect command line parameters for gams",
    2000: "Driver error: internal error: cannot install interrupt handler",
    3000: "Driver error: problems getting current directory",
    4000: (
        "Driver error: internal error: GAMS compile and execute module not"
        " found"
    ),
    5000: "Driver error: internal error: cannot load option handling library",
}


def customize_exception(
    workspace: GamsWorkspace,
    options: GamsOptions,
    job: str,
    exception: GamsExceptionExecution,
) -> str:
    error_message = ""
    if not options._writeoutput:
        exception.value = error_message
        return exception

    header = "=" * 80
    footer = "=" * 80
    message_format = "\n\n{header}\nError Summary\n{footer}\n{message}\n"

    if options.output:
        lst_path = (
            options.output
            if os.path.isabs(options.output)
            else os.path.join(workspace._working_directory, options.output)
        )
    else:
        lst_path = job + ".lst"

    with open(lst_path, encoding="utf-8") as lst_file:
        all_lines = lst_file.readlines()
        num_lines = len(all_lines)

        index = 0
        while index < num_lines:
            line = all_lines[index]

            if line.startswith("****"):
                error_lines = [all_lines[index - 1]]
                temp_index = index

                while (
                    all_lines[temp_index].startswith("****")
                    and temp_index < len(all_lines) - 1
                ):
                    error_lines.append(all_lines[temp_index])
                    temp_index += 1

                error_message = message_format.format(
                    message="".join(error_lines),
                    header=header,
                    footer=footer,
                    return_code=exception.rc,
                    meaning=error_codes[exception.rc],
                )
                break

            index += 1

    explanation = (
        f"\nMeaning of return code {exception.rc}: {error_codes[exception.rc]}"
    )

    return error_message + explanation
