import json
import os
import subprocess
from argparse import ArgumentParser, Namespace

from gamspy import __version__


def update_pyproject(args: Namespace) -> None:
    with open("pyproject.toml") as file:
        content = file.read()

    pattern = f'version = "{__version__}"'
    new_pattern = f'version = "{args.new_version}"'
    new_content = content.replace(pattern, new_pattern)

    with open("pyproject.toml", "w") as file:
        file.write(new_content)

    print("pyproject.toml has been updated!")


def update_switcher(args: Namespace) -> None:
    switcher_path = os.path.join("docs", "_static", "switcher.json")
    with open(switcher_path) as file:
        switcher = json.loads(file.read())

    switcher[1]["name"] = switcher[1]["name"].removesuffix(" (stable)")
    new_version = {
        "name": f"{args.new_version} (stable)",
        "version": f"v{args.new_version}",
        "url": f"https://gamspy.readthedocs.io/en/v{args.new_version}/",
    }

    if switcher[1]["name"] != args.new_version:
        switcher.insert(1, new_version)

    with open(switcher_path, "w") as file:
        json.dump(switcher, file)

    print("switcher.json has been updated!")


def update_version_test(args: Namespace) -> None:
    test_path = os.path.join("tests", "test_gamspy.py")
    with open(test_path) as file:
        content = file.read()

    pattern = f'gp.__version__ == "{__version__}"'
    new_pattern = f'gp.__version__ == "{args.new_version}"'
    new_content = content.replace(pattern, new_pattern)

    with open(test_path, "w") as file:
        file.write(new_content)

    print("version test has been updated!")


def update_release_notes(args: Namespace) -> None:
    # update the release notes index
    release_index_path = os.path.join("docs", "release", "index.rst")
    with open(release_index_path) as file:
        lines = file.readlines()

    index = -1
    for idx, line in enumerate(lines):
        if line.endswith(f"release_{__version__}\n"):
            index = idx

    current_release = lines[index]
    new_release = current_release.replace(__version__, args.new_version)
    lines.insert(index, new_release)

    with open(release_index_path, "w") as file:
        file.write("".join(lines))

    # create a release note file for the new release
    release_path = os.path.join("docs", "release", f"release_{args.new_version}.rst")
    process = subprocess.run(
        ["towncrier", "build", "--draft", "--version", args.new_version],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    with open(release_path, "w") as file:
        file.write(process.stdout)

    print("release notes has been updated!")


def update_changelog(args: Namespace) -> None:
    subprocess.run(
        ["towncrier", "build", "--yes", "--version", args.new_version],
        check=True,
    )


def main():
    parser = ArgumentParser()
    parser.add_argument("new_version")
    args = parser.parse_args()
    assert args.new_version

    update_pyproject(args)
    update_switcher(args)
    update_version_test(args)
    update_release_notes(args)
    update_changelog(args)


if __name__ == "__main__":
    main()
