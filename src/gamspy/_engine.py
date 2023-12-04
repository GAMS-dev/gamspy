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

from gams import GamsEngineConfiguration
from gams import GamsWorkspace
from pydantic import BaseModel


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
        self, workspace: GamsWorkspace, gdx_path: str
    ) -> List[str]:
        # copy provided extra model files to working directory
        for extra_file in self.extra_model_files:
            try:
                shutil.copy(extra_file, workspace.working_directory)
            except shutil.SameFileError:
                # extra file might already be in the working directory
                pass

        # trim path and keep only the names of the files
        extra_model_files = [
            os.path.basename(extra_file)
            for extra_file in self.extra_model_files
        ]

        # add name of the gdx file
        extra_model_files.append(os.path.basename(gdx_path))

        return extra_model_files
