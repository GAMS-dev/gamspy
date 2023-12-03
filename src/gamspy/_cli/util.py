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
import platform
from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import Tuple
from typing import Union

__all__ = ["SolverInfo", "add_solver_entry", "remove_solver_entry"]

platform_to_capabilities_file = {
    "windows": "gmscmpNT.txt",
    "linux": "gmscmpun.txt",
    "mac_x86_64": "gmscmpun.txt",
    "mac_arm64": "gmscmpun.txt",
}


def get_platform() -> str:
    operating_system = platform.system().lower()
    architecture = platform.machine()

    if operating_system == "darwin":
        return f"mac_{architecture}"

    return operating_system


@dataclass
class SolverInfo:
    solver_id: str
    file_type: str
    dict_type: str
    lic_codes: str
    default_ok_flag: str
    hidden_flag: str
    lines_to_follow: str
    model_types: list = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"{self.solver_id} {self.file_type} {self.dict_type} "
            f"{self.lic_codes} {self.default_ok_flag} {self.hidden_flag} "
            f"{self.lines_to_follow} {' '.join(self.model_types)}"
        )


def check_solver_exists(
    capabilities_file: str, solver_name: str
) -> Union[Tuple[int, int], None]:
    with open(capabilities_file) as capabilities:
        lines = capabilities.readlines()
        lines = [line for line in lines if line != "\n" and line[0] != "*"]
        idx = 0
        while True:
            start_idx = idx
            line1 = lines[idx]
            if line1 == "DEFAULTS\n":
                break

            idx += 2
            splitted_line = line1.split(" ")
            solver = SolverInfo(*splitted_line[:7])  # type: ignore
            idx += int(solver.lines_to_follow)
            if solver.solver_id.lower() == solver_name.lower():
                return start_idx, 2 + int(solver.lines_to_follow)

        return None


def get_capabilities_filename() -> str:
    current_platform = get_platform()
    return platform_to_capabilities_file[current_platform]


def add_solver_entry(
    gamspy_base_location: str,
    solver_name: str,
    verbatims: List[str],
):
    capabilities_file = (
        gamspy_base_location + os.sep + get_capabilities_filename()
    )

    if check_solver_exists(capabilities_file, solver_name):
        if solver_name == "scip":
            if check_solver_exists(capabilities_file, "mosek"):
                print(
                    "Solver already exists in the capabilities file, skipping"
                )
                return
        else:
            print("Solver already exists in the capabilities file, skipping")
            return

    with open(capabilities_file) as f:
        string = f.read()

    for verbatim in verbatims:
        string = f"{verbatim}\n\n{string}"

        with open(capabilities_file, "w") as f:
            f.write(string)


def remove_solver_entry(gamspy_base_location: str, solver_name: str):
    capabilities_file = (
        gamspy_base_location + os.sep + get_capabilities_filename()
    )
    solver_tuple = check_solver_exists(capabilities_file, solver_name)

    if not solver_tuple:
        print("Solver is not in the capabilities file, skipping")
        return

    line_num, line_count = solver_tuple
    with open(capabilities_file) as f:
        lines = f.readlines()

    for _ in range(line_count + 1):
        lines.pop(line_num)

    with open(capabilities_file, "w") as f:
        f.writelines(lines)
