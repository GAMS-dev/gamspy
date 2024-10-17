from __future__ import annotations

import os
from dataclasses import dataclass, field

import gamspy.utils as utils

__all__ = ["SolverInfo", "add_solver_entry", "remove_solver_entry"]


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


def add_solver_entry(
    system_directory: str, solver_name: str, verbatims: list[str]
):
    capabilities_path = os.path.join(system_directory, utils.CAPABILITIES_FILE)
    installed_solvers = utils.getInstalledSolvers(system_directory)
    if solver_name.upper() in installed_solvers:
        print(
            f"`{solver_name}` already exists in the capabilities file, skipping"
        )
        return

    with open(capabilities_path, encoding="utf-8") as f:
        string = f.read()

    for verbatim in verbatims:
        string = f"{verbatim}\n\n{string}"

        with open(capabilities_path, "w", encoding="utf-8") as f:
            f.write(string)


def find_bounds(system_directory: str, solver_name: str) -> tuple[int, int]:
    capabilities_path = os.path.join(system_directory, utils.CAPABILITIES_FILE)
    with open(capabilities_path, encoding="utf-8") as file:
        lines = file.readlines()

    start_idx = 0
    line_count = 0
    while True:
        line = lines.pop(0)
        if line.startswith("*") or line == "" or line == "\n":
            start_idx += 1
            continue
        if line == "DEFAULTS":
            break

        start_idx += 1
        solver, _, _, _, _, _, num_lines, *_ = line.split()
        for _ in range(int(num_lines) + 2):
            _ = lines.pop(0)

        start_idx += int(num_lines) + 2

        if solver == solver_name.upper():
            line_count = int(num_lines)
            start_idx -= int(num_lines) + 3
            break

    return start_idx, line_count


def remove_solver_entry(system_directory: str, solver_name: str):
    capabilities_path = os.path.join(system_directory, utils.CAPABILITIES_FILE)
    installed_solvers = utils.getInstalledSolvers(system_directory)

    if solver_name.upper() not in installed_solvers:
        print("Solver is not in the capabilities file, skipping")
        return

    line_num, line_count = find_bounds(system_directory, solver_name)

    with open(capabilities_path, encoding="utf-8") as f:
        lines = f.readlines()

    for _ in range(line_count + 3):
        lines.pop(line_num)

    with open(capabilities_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
