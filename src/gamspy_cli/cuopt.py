"""
Installation of the GAMS solver link for the NVIDIA cuOpt solver. cuOpt is not
distributed as a gamspy-<solver_name> package but as a release archive of
https://github.com/GAMS-dev/cuoptlink-builder.
"""

from __future__ import annotations

import ctypes
import os
import platform
import re
import shutil
import tempfile
import zipfile

import typer

SOLVER_NAME = "cuopt"
REPOSITORY = "GAMS-dev/cuoptlink-builder"
RELEASES_URL = f"https://api.github.com/repos/{REPOSITORY}/releases"
RELEASES_PAGE = f"https://github.com/{REPOSITORY}/releases"
DEFAULT_VERSION = "v0.0.8b"
CUDA_VERSIONS = ("12", "13")
DEFAULT_CUDA_VERSION = "13"
CONFIG_FILE = "gamsconfig.yaml"
SOLVER_CONFIG_FILE = "gamsconfig_cuopt.yaml"
SOLVER_CONFIG_SECTION = "solverConfig"
BACKUP_FILE = "gamsconfig.yaml.gamspy_backup"
MARKER_BEGIN = f"# begin {SOLVER_NAME} solver configuration of GAMSPy"
MARKER_END = f"# end {SOLVER_NAME} solver configuration of GAMSPy"
MANIFEST_FILE = "cuopt_files.txt"

CUOPT_VERSION_ENV = "GAMSPY_CUOPT_VERSION"
CUDA_VERSION_ENV = "GAMSPY_CUDA_VERSION"
CUDA_RUNTIME_ENV = "GAMSPY_CUDA_RUNTIME"


def _get_version() -> str:
    version = os.getenv(CUOPT_VERSION_ENV, "").strip()
    if not version:
        return DEFAULT_VERSION

    return version if version.startswith("v") else f"v{version}"


def _get_cuda_version() -> str | None:
    cuda_version = os.getenv(CUDA_VERSION_ENV, "").strip()
    if not cuda_version:
        return None

    if cuda_version not in CUDA_VERSIONS:
        typer.echo(
            f"Invalid `{CUDA_VERSION_ENV}` ({cuda_version}). `{SOLVER_NAME}` supports "
            f"{' and '.join(CUDA_VERSIONS)}."
        )
        raise typer.Exit(code=1)

    return cuda_version


def _install_cuda_runtime() -> bool:
    value = os.getenv(CUDA_RUNTIME_ENV, "").strip()
    if not value:
        return True

    if value not in ("0", "1"):
        typer.echo(f"Invalid `{CUDA_RUNTIME_ENV}` ({value}). It must be 0 or 1.")
        raise typer.Exit(code=1)

    return value == "1"


def _get_manifest_path() -> str:
    import gamspy.utils as utils

    return os.path.join(utils.DEFAULT_DIR, MANIFEST_FILE)


def get_installed_files() -> list[str]:
    try:
        with open(_get_manifest_path(), encoding="utf-8") as file:
            return [line for line in file.read().splitlines() if line]
    except FileNotFoundError:
        return []


def _set_installed_files(files: list[str]) -> None:
    import gamspy.utils as utils

    os.makedirs(utils.DEFAULT_DIR, exist_ok=True)
    with open(_get_manifest_path(), "w", encoding="utf-8") as file:
        file.write("\n".join(sorted(files)) + "\n")


def _get_architecture() -> str:
    if platform.system() != "Linux":
        typer.echo(
            f"`{SOLVER_NAME}` is only available on Linux. On Windows, install it in a "
            "WSL2 distribution."
        )
        raise typer.Exit(code=1)

    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x86_64"

    if machine in ("aarch64", "arm64"):
        return "arm64"

    typer.echo(
        f"`{SOLVER_NAME}` is only available for x86_64 and arm64 but found {machine}."
    )
    raise typer.Exit(code=1)


def _detect_cuda_version() -> str | None:
    for version in reversed(CUDA_VERSIONS):
        for library in (f"libcudart.so.{version}", f"libcublas.so.{version}"):
            try:
                _ = ctypes.CDLL(library)
            except OSError:
                continue

            return version

    return None


def _get_asset_urls(names: list[str], version: str) -> list[str]:
    import requests

    url = f"{RELEASES_URL}/tags/{version}"
    try:
        response = requests.get(url, timeout=30)
    except requests.RequestException as e:
        typer.echo(
            f"Could not reach {url}. Please check your internet connection. "
            f"Here is the error message: {e}"
        )
        raise typer.Exit(code=1) from e

    if response.status_code == 404:
        typer.echo(
            f"{REPOSITORY} has no release `{version}`. Set `{CUOPT_VERSION_ENV}` to one of "
            f"the versions listed on {RELEASES_PAGE} or unset it to install "
            f"{DEFAULT_VERSION}."
        )
        raise typer.Exit(code=1)

    if response.status_code != 200:
        typer.echo(
            f"Could not get the `{version}` `{SOLVER_NAME}` release. Request status: "
            f"{response.status_code}. Reason: {response.text}."
        )
        raise typer.Exit(code=1)

    release = response.json()
    assets = {
        asset["name"]: asset["browser_download_url"] for asset in release["assets"]
    }

    urls = []
    for name in names:
        if name not in assets:
            typer.echo(
                f"Release `{release['tag_name']}` of {REPOSITORY} does not contain "
                f"`{name}`. Available archives: {sorted(assets)}"
            )
            raise typer.Exit(code=1)

        urls.append(assets[name])

    typer.echo(f"Installing `{SOLVER_NAME}` from release {release['tag_name']}...")
    return urls


def _download(url: str, path: str) -> None:
    import requests
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
    )

    name = os.path.basename(path)
    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length", 0))

            with (
                Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    DownloadColumn(),
                    TimeRemainingColumn(),
                ) as progress,
                open(path, "wb") as file,
            ):
                task = progress.add_task(name, total=total or None)
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    _ = file.write(chunk)
                    progress.update(task, advance=len(chunk))
    except requests.RequestException as e:
        typer.echo(
            f"Could not download {url}. Please check your internet connection. "
            f"Here is the error message: {e}"
        )
        raise typer.Exit(code=1) from e


def _extract(path: str, directory: str) -> list[str]:
    names = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue

            # Do not allow escaping the system directory
            name = os.path.basename(info.filename)
            if not name:
                continue

            with (
                archive.open(info) as source,
                open(os.path.join(directory, name), "wb") as target,
            ):
                shutil.copyfileobj(source, target)

            mode = (info.external_attr >> 16) & 0o777
            if mode:
                os.chmod(os.path.join(directory, name), mode)

            names.append(name)

    return names


def _backup_config(gamspy_base_directory: str, installed_files: list[str]) -> None:
    config_path = os.path.join(gamspy_base_directory, CONFIG_FILE)
    if not os.path.isfile(config_path) or CONFIG_FILE in installed_files:
        return

    backup_path = os.path.join(gamspy_base_directory, BACKUP_FILE)
    _ = shutil.copy2(config_path, backup_path)
    typer.echo(
        f"`{config_path}` is extended with the `{SOLVER_NAME}` solver "
        f"configuration. A backup was written to `{backup_path}`."
    )


def _load_config(path: str) -> dict:
    import yaml

    with open(path, encoding="utf-8") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            typer.echo(f"`{path}` is not a valid YAML file. Here is the error: {e}")
            raise typer.Exit(code=1) from e

    if config is None:
        return {}

    if not isinstance(config, dict):
        typer.echo(
            f"`{path}` must map configuration sections such as "
            f"`{SOLVER_CONFIG_SECTION}` to their entries."
        )
        raise typer.Exit(code=1)

    return config


def _find_section(lines: list[str], section: str, path: str) -> int | None:
    """The index of the line that opens `section` as a block of entries."""
    for index, line in enumerate(lines):
        if not re.match(rf"{re.escape(section)}[ \t]*:", line):
            continue

        if not re.match(rf"{re.escape(section)}[ \t]*:[ \t]*(#.*)?$", line):
            typer.echo(
                f"`{section}` of `{path}` is not written as a block of entries, "
                f"hence GAMSPy cannot add `{SOLVER_NAME}` to it. Please add the "
                f"entry of `{SOLVER_CONFIG_FILE}` to it by hand."
            )
            raise typer.Exit(code=1)

        return index

    return None


def _get_section_end(lines: list[str], start: int) -> int:
    end = start
    for index in range(start, len(lines)):
        line = lines[index]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # The entries of a block sequence may be indented like their section.
        is_entry = stripped == "-" or stripped.startswith("- ")
        if not line[0].isspace() and not is_entry:
            break

        end = index + 1

    return end


def _get_document_end(lines: list[str]) -> int:
    for index, line in enumerate(lines):
        if re.match(r"\.\.\.([ \t]|$)", line):
            return index

    return len(lines)


def _get_entry_indent(entries: list[str]) -> str | None:
    for entry in entries:
        match = re.match(r"([ \t]*)-([ \t]|$)", entry)
        if match is not None:
            return match.group(1)

    return None


def _reindent(entries: list[str], indent: str) -> list[str]:
    base = _get_entry_indent(entries) or ""
    return [
        indent + entry[len(base) :] if entry.startswith(base) else entry
        for entry in entries
    ]


def _get_solver_config(gamspy_base_directory: str) -> list[str]:
    """The entries that the release archive registers in gamsconfig_cuopt.yaml"""
    path = os.path.join(gamspy_base_directory, SOLVER_CONFIG_FILE)
    if list(_load_config(path)) != [SOLVER_CONFIG_SECTION]:
        typer.echo(
            f"`{path}` of the release archive holds more than a "
            f"`{SOLVER_CONFIG_SECTION}` section, which GAMSPy cannot merge into "
            f"`{CONFIG_FILE}`. Please add its content to `{CONFIG_FILE}` by hand."
        )
        raise typer.Exit(code=1)

    with open(path, encoding="utf-8") as file:
        lines = file.read().splitlines()

    index = _find_section(lines, SOLVER_CONFIG_SECTION, path)
    if index is None:
        typer.echo(f"`{path}` of the release archive does not register a solver.")
        raise typer.Exit(code=1)

    return lines[index + 1 : _get_section_end(lines, index + 1)]


def _merge_solver_config(text: str, entries: list[str], path: str) -> str:
    lines = text.splitlines()
    index = _find_section(lines, SOLVER_CONFIG_SECTION, path)
    if index is None:
        block = [f"{SOLVER_CONFIG_SECTION}:", *entries]
        end = _get_document_end(lines)
    else:
        end = _get_section_end(lines, index + 1)
        indent = _get_entry_indent(lines[index + 1 : end])
        block = entries if indent is None else _reindent(entries, indent)

    merged = [*lines[:end], MARKER_BEGIN, *block, MARKER_END, *lines[end:]]
    return "\n".join(merged) + "\n"


def _strip_solver_config(text: str) -> str:
    """Take the entries of a previous installation of GAMSPy back out of `text`."""
    remaining = []
    is_contributed = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == MARKER_BEGIN:
            is_contributed = True
        elif stripped == MARKER_END:
            is_contributed = False
        elif not is_contributed:
            remaining.append(line)

    return "\n".join(remaining) + "\n" if remaining else ""


def _register_solver(gamspy_base_directory: str, installed_files: list[str]) -> bool:
    """Merge the gamsconfig_cuopt.yaml of the release archive into the gamsconfig.yaml"""
    entries = _get_solver_config(gamspy_base_directory)
    config_path = os.path.join(gamspy_base_directory, CONFIG_FILE)

    text = ""
    is_created_by_gamspy = CONFIG_FILE in installed_files
    try:
        with open(config_path, encoding="utf-8") as file:
            text = _strip_solver_config(file.read())
    except FileNotFoundError:
        is_created_by_gamspy = True

    with open(config_path, "w", encoding="utf-8") as file:
        _ = file.write(_merge_solver_config(text, entries, config_path))

    return is_created_by_gamspy


def _unregister_solver(gamspy_base_directory: str) -> None:
    """Take the entries of the release archive back out of the gamsconfig.yaml"""
    config_path = os.path.join(gamspy_base_directory, CONFIG_FILE)
    try:
        with open(config_path, encoding="utf-8") as file:
            remaining = _strip_solver_config(file.read())
    except FileNotFoundError:
        return

    if not remaining.strip():
        os.unlink(config_path)
        return

    with open(config_path, "w", encoding="utf-8") as file:
        _ = file.write(remaining)


def install() -> list[str]:
    import gamspy.utils as utils

    gamspy_base_directory = utils._get_gamspy_base_directory()
    architecture = _get_architecture()
    version = _get_version()
    cuda_version = _get_cuda_version()
    install_cuda_runtime = _install_cuda_runtime()

    if cuda_version is None:
        cuda_version = _detect_cuda_version()
        if cuda_version is None:
            if not install_cuda_runtime:
                typer.echo(
                    "Could not find a CUDA runtime on your machine. Install the CUDA "
                    "runtime (https://developer.nvidia.com/cuda-downloads) and specify "
                    f"its major version with `{CUDA_VERSION_ENV}="
                    f"<{'|'.join(CUDA_VERSIONS)}>`, or let GAMSPy install the required "
                    f"CUDA runtime libraries by unsetting `{CUDA_RUNTIME_ENV}`."
                )
                raise typer.Exit(code=1)

            cuda_version = DEFAULT_CUDA_VERSION
        else:
            typer.echo(f"Found CUDA runtime {cuda_version} on your machine.")

    archives = [f"cuopt-link-release-cu{cuda_version}-{architecture}.zip"]
    if install_cuda_runtime:
        archives.append(f"cu{cuda_version}-runtime-{architecture}.zip")

    urls = _get_asset_urls(archives, version)
    installed_files = get_installed_files()
    _backup_config(gamspy_base_directory, installed_files)

    files = set(installed_files)
    with tempfile.TemporaryDirectory() as directory:
        for name, url in zip(archives, urls):
            archive_path = os.path.join(directory, name)
            _download(url, archive_path)
            files.update(_extract(archive_path, gamspy_base_directory))

    if _register_solver(gamspy_base_directory, installed_files):
        files.add(CONFIG_FILE)
    else:
        files.discard(CONFIG_FILE)

    _set_installed_files(sorted(files))
    return sorted(files)


def uninstall() -> None:
    import gamspy.utils as utils

    gamspy_base_directory = utils._get_gamspy_base_directory()
    installed_files = get_installed_files()
    if not installed_files:
        typer.echo(
            f"`{SOLVER_NAME}` was not installed with GAMSPy, hence it cannot be "
            "uninstalled with GAMSPy."
        )
        raise typer.Exit(code=1)

    # gamsconfig.yaml is an installed file only if GAMSPy created it. One that was
    # already there may configure more than cuopt, so only its entries are removed.
    if CONFIG_FILE not in installed_files:
        _unregister_solver(gamspy_base_directory)

    for name in installed_files:
        try:
            os.unlink(os.path.join(gamspy_base_directory, name))
        except FileNotFoundError:
            ...

    try:
        os.unlink(os.path.join(gamspy_base_directory, BACKUP_FILE))
    except FileNotFoundError:
        ...

    try:
        os.unlink(_get_manifest_path())
    except FileNotFoundError:
        ...
