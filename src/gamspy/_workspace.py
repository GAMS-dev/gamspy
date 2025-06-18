from __future__ import annotations

import os
import shutil
import tempfile
import weakref

from gamspy.exceptions import ValidationError

DEBUGGING_LEVELS = ("delete", "keep_on_error", "keep")


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
        self,
        debugging_level: str,
        working_directory: str | os.PathLike | None = None,
    ):
        if isinstance(working_directory, os.PathLike):
            working_directory = os.fspath(working_directory)

        validate_arguments(working_directory, debugging_level)

        self.debugging_level = debugging_level
        self.using_tmp_working_dir = False
        self._errors: list[str] = []

        if working_directory is None:
            self.using_tmp_working_dir = True
            self.working_directory = tempfile.mkdtemp()
        else:
            self.working_directory = os.path.abspath(working_directory)
            os.makedirs(self.working_directory, exist_ok=True)

        weakref.finalize(
            self,
            self.cleanup,
            self.using_tmp_working_dir,
            self.debugging_level,
            self.working_directory,
            self._errors,
        )

    @staticmethod
    def cleanup(
        using_tmp_working_dir: bool,
        debugging_level: str,
        working_directory: str,
        errors: list[str],
    ):
        if using_tmp_working_dir:
            try:  # in case working directory has already been deleted.
                if debugging_level == "delete":
                    shutil.rmtree(working_directory)

                if debugging_level == "keep_on_error" and len(errors) == 0:
                    shutil.rmtree(working_directory)
            except (FileNotFoundError, PermissionError):
                ...
