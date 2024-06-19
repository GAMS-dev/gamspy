"""Exception classes for GAMSPy"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gamspy import Options


class GamspyException(Exception):
    """Plain Gamspy exception."""

    def __init__(self, message: str, return_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.rc = return_code


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
    13: "System error",
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
    working_directory: str,
    options: Options,
    job_name: str,
    return_code: int | None,
) -> str:
    if options.write_listing_file is False or return_code is None:
        return ""

    header = "=" * 14
    footer = "=" * 14
    message_format = "\n\n{header}\nError Summary\n{footer}\n{message}\n"

    if options.listing_file:
        lst_path = (
            options.listing_file
            if os.path.isabs(options.listing_file)
            else os.path.join(working_directory, options.listing_file)
        )
    else:
        lst_path = job_name + ".lst"

    try:
        with open(lst_path, encoding="utf-8") as lst_file:
            all_lines = lst_file.readlines()
            num_lines = len(all_lines)

            index = 0
            while index < num_lines:
                line = all_lines[index]

                if line.startswith("****"):
                    error_lines = [all_lines[index - 1]]
                    temp_index = index

                    try:
                        while any(
                            "****" in err_line
                            for err_line in all_lines[
                                temp_index : temp_index + 8
                            ]
                        ):
                            for offset in range(8):
                                error_lines.append(
                                    all_lines[temp_index + offset]
                                )

                            temp_index += 8
                    except IndexError:
                        ...

                    error_message = message_format.format(
                        message="".join(error_lines),
                        header=header,
                        footer=footer,
                        return_code=return_code,
                        meaning=error_codes[return_code],
                    )
                    break

                index += 1
    except FileNotFoundError:
        return ""

    explanation = (
        f"\nMeaning of return code {return_code}: {error_codes[return_code]}"
    )

    return error_message + explanation
