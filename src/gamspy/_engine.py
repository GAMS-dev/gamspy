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

import os
import shutil
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union

from gams import GamsEngineConfiguration
from gams import GamsOptions
from gams import GamsWorkspace
from gams.control.workspace import GamsException
from gams.control.workspace import GamsExceptionExecution
from pydantic import BaseModel

from gamspy.exceptions import GamspyException

if TYPE_CHECKING:
    import io
    from gamspy import Container


class EngineConfig(BaseModel):
    host: str
    username: Optional[str] = None
    password: Optional[str] = None
    jwt: Optional[str] = None
    namespace: str = "global"
    extra_model_files: List[str] = []
    engine_options: Optional[dict] = None
    remove_results: bool = False

    class Config:
        extra = "forbid"

    def _get_engine_config(self):
        return GamsEngineConfiguration(
            self.host,
            self.username,
            self.password,
            self.jwt,
            self.namespace,
        )


def _preprocess_extra_model_files(
    config: EngineConfig, workspace: GamsWorkspace, gdx_path: str
) -> List[str]:
    # copy provided extra model files to working directory
    for extra_file in config.extra_model_files:
        try:
            shutil.copy(extra_file, workspace.working_directory)
        except shutil.SameFileError:
            # extra file might already be in the working directory
            pass

    extra_model_files = [
        os.path.basename(extra_file) for extra_file in config.extra_model_files
    ]

    extra_model_files.append(os.path.basename(gdx_path))

    return extra_model_files


def run(
    container: Container,
    options: GamsOptions,
    output: Union[io.TextIOWrapper, None],
    engine_config: EngineConfig,
):
    options.previouswork = 1  # In case GAMS version differs on Engine

    extra_model_files = _preprocess_extra_model_files(
        engine_config, container.workspace, container._gdx_in
    )

    try:
        container._job.run_engine(  # type: ignore
            engine_configuration=engine_config._get_engine_config(),
            extra_model_files=extra_model_files,
            gams_options=options,
            checkpoint=container._save_to,
            output=output,
            create_out_db=False,
            engine_options=engine_config.engine_options,
            remove_results=engine_config.remove_results,
        )
    except (GamsException, GamsExceptionExecution) as e:
        raise GamspyException(str(e))
    finally:
        container._unsaved_statements = []
        options.previouswork = 0
