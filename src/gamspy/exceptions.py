"""Exception classes for GAMSPy"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gamspy import Options


class FatalError(Exception):
    """Indicates an error that cannot be recovered from. This error should never be caught."""


class GamspyException(Exception):
    """
    Plain Gamspy exception. This exception can be caught and GAMSPy should be able to continue.

    Parameters
    ----------
    message : str
    return_code : int | None
    """

    def __init__(self, message: str, return_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.return_code = return_code


class ValidationError(Exception):
    """An error while validating data."""


class NeosClientException(Exception):
    """NeosClient exception."""


class EngineClientException(Exception):
    """EngineClient exception"""


class NeosException(GamspyException):
    """NEOS Server execution exception"""


class LatexException(GamspyException):
    """To latex execution exception"""


class EngineException(GamspyException):
    """
    GAMS Engine execution exception.

    Parameters
    ----------
    message : str
    return_code : int
    status_code : int | None, optional
    """

    def __init__(
        self,
        message: str,
        return_code: int,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message, return_code)
        self.status_code = status_code


CHUNK_SIZE = 8
FRAME = "=" * 13


def _parse_errors(lines: list[str], index: int) -> str:
    error_lines = [lines[index - 1]]
    temp_index = index

    try:
        chunk = lines[temp_index : temp_index + CHUNK_SIZE]

        while any("****" in err_line for err_line in chunk):
            error_lines += chunk
            temp_index += CHUNK_SIZE
            chunk = lines[temp_index : temp_index + CHUNK_SIZE]
    except IndexError:
        ...

    error_message = (
        f"\n\n{FRAME}\nError Summary\n{FRAME}\n{''.join(error_lines)}"
    )

    return error_message.rstrip()


def _customize_exception(
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
            else os.path.join(os.getcwd(), options.listing_file)
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
