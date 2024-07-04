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


class EngineException(GamspyException):
    def __init__(
        self,
        message: str,
        return_code: int,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message, return_code)
        self.status_code = status_code

    def __str__(self) -> str:
        return self.message


class ValidationError(Exception):
    """An error while validating data."""


def customize_exception(
    working_directory: str,
    options: Options,
    job_name: str,
    return_code: int | None,
) -> str:
    error_message = ""
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
                    )
                    break

                index += 1
    except FileNotFoundError:
        return error_message

    return error_message
