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
        self.return_code = return_code


class NeosClientException(Exception):
    """NeosClient exception."""


class EngineClientException(Exception):
    """EngineClient exception"""


class NeosException(GamspyException):
    """NEOS Server execution exception"""


class LatexException(GamspyException):
    """To latex execution exception"""


class EngineException(GamspyException):
    def __init__(
        self,
        message: str,
        return_code: int,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message, return_code)
        self.status_code = status_code


class ValidationError(Exception):
    """An error while validating data."""


CHUNK_SIZE = 8
FRAME = "=" * 14


def _parse_errors(lines: list[str], index: int) -> str:
    error_lines = [lines[index - 1]]
    temp_index = index

    try:
        chunk = lines[temp_index : temp_index + CHUNK_SIZE]

        while any("****" in err_line for err_line in chunk):
            chunk = lines[temp_index : temp_index + CHUNK_SIZE]
            error_lines += chunk
            temp_index += CHUNK_SIZE
    except IndexError:
        ...

    error_message = (
        f'\n\n{FRAME}\nError Summary\n{FRAME}\n{"".join(error_lines)}'
    )

    return error_message


def customize_exception(
    working_directory: str,
    options: Options,
    job_name: str,
    return_code: int | None,
) -> str:
    error_message = ""
    if options.write_listing_file is False or return_code is None:
        return ""

    if options.listing_file:
        lst_path = (
            options.listing_file
            if os.path.isabs(options.listing_file)
            else os.path.join(working_directory, options.listing_file)
        )
    else:
        lst_path = job_name + ".lst"

    with open(lst_path, encoding="utf-8") as lst_file:
        all_lines = lst_file.readlines()

    num_lines = len(all_lines)
    index = 0
    while index < num_lines:
        line = all_lines[index]

        if line.startswith("****"):
            error_message = _parse_errors(all_lines, index)
            break

        index += 1

    return error_message
