from __future__ import annotations

import os
import shutil
import tempfile

from gamspy.exceptions import ValidationError

DEBUGGING_LEVELS = ["delete", "keep_on_error", "keep"]


def validate_arguments(
    working_directory: str | None,
    debugging_level: str,
):
    # Validate working_directory
    if working_directory == "":
        raise ValidationError("`working_directory` cannot be an empty string.")

    # Validate debug_level
    if (
        not isinstance(debugging_level, str)
        or debugging_level not in DEBUGGING_LEVELS
    ):
        raise ValidationError(
            f"debugging level must be one of {DEBUGGING_LEVELS}"
        )


class Workspace:
    def __init__(
        self, debugging_level: str, working_directory: str | None = None
    ):
        validate_arguments(working_directory, debugging_level)

        self.debugging_level = debugging_level
        self.using_tmp_working_dir = False
        self._has_error = False
        self._first_try = True

        if working_directory is None:
            self.using_tmp_working_dir = True
            self.working_directory = tempfile.mkdtemp()
        else:
            self.working_directory = os.path.abspath(working_directory)
            os.makedirs(self.working_directory, exist_ok=True)

    def __del__(self):
        if (
            hasattr(self, "using_tmp_working_dir")
            and self.using_tmp_working_dir
        ):
            try:
                if self.debugging_level == "delete":
                    shutil.rmtree(self.working_directory)

                if (
                    self.debugging_level == "keep_on_error"
                    and not self._has_error
                ):
                    shutil.rmtree(self.working_directory)
            except PermissionError:  # pragma: no cover
                ...
