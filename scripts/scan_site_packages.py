"""
Fails if the interpreter running this script would execute code at startup
through .pth import lines, .start entry points, or sitecustomize/usercustomize
modules, unless the file is allowlisted by its exact content. Run it with
``python -I -S`` so that the startup code it looks for is not executed.
"""

from __future__ import annotations

import os
import site
import sys

ALLOWED: dict[str, bytes] = {
    # coverage 7.x
    "a1_coverage.pth": (
        b'import sys; exec(\'import os\\n\\nif os.getenv("COVERAGE_PROCESS_START")'
        b' or os.getenv("COVERAGE_PROCESS_CONFIG"):\\n try:\\n  import coverage\\n'
        b' except:\\n  pass\\n else:\\n  coverage.process_startup(slug="pth")\')\n'
    ),
}

CUSTOMIZE_MODULES = ("sitecustomize", "usercustomize")


def site_dirs() -> list[str]:
    dirs = site.getsitepackages()
    include_system = True
    cfg = os.path.join(sys.prefix, "pyvenv.cfg")
    if sys.prefix != sys.base_prefix and os.path.isfile(cfg):
        with open(cfg, encoding="utf-8") as f:
            for line in f:
                key, _, value = line.partition("=")
                if key.strip().lower() == "include-system-site-packages":
                    include_system = value.strip().lower() == "true"
        if include_system:
            dirs += site.getsitepackages([sys.base_prefix])

    if include_system:
        dirs.append(site.getusersitepackages())

    return [d for d in dict.fromkeys(dirs) if os.path.isdir(d)]


def read_lines(content: bytes) -> list[str]:
    text = content.decode("utf-8-sig", errors="replace")
    return [line.strip() for line in text.splitlines()]


def is_code(line: str) -> bool:
    return bool(line) and not line.startswith("#")


def scan_dir(directory: str) -> tuple[list[str], list[str]]:
    problems = []
    path_entries = []
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if name.partition(".")[0] in CUSTOMIZE_MODULES:
            problems.append(f"{path}: runs at interpreter startup")
            continue

        if not name.endswith((".pth", ".start")) or not os.path.isfile(path):
            continue

        with open(path, "rb") as f:
            content = f.read()
        # Installers on Windows may write the file with CRLF line endings.
        if ALLOWED.get(name) == content.replace(b"\r\n", b"\n"):
            continue

        for lineno, line in enumerate(read_lines(content), 1):
            if name.endswith(".start") and is_code(line):
                problems.append(f"{path}:{lineno}: entry point {line!r}")
            elif line.startswith(("import ", "import\t")):
                problems.append(f"{path}:{lineno}: import line {line[:80]!r}")
            elif name.endswith(".pth") and is_code(line):
                path_entries.append(os.path.normpath(os.path.join(directory, line)))

    return problems, path_entries


def main() -> int:
    problems = []
    for directory in site_dirs():
        dir_problems, path_entries = scan_dir(directory)
        problems += dir_problems
        for entry in path_entries:
            if not os.path.isdir(entry):
                continue
            problems += [
                f"{os.path.join(entry, name)}: runs at interpreter startup"
                for name in sorted(os.listdir(entry))
                if name.partition(".")[0] in CUSTOMIZE_MODULES
            ]

    for problem in problems:
        print(problem)

    if problems:
        print(
            f"\n{sys.executable} runs the code above at startup. Remove it, or "
            "add the file to ALLOWED in scripts/scan_site_packages.py after "
            "reviewing it."
        )

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
