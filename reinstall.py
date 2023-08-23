import os
import subprocess
import platform
import glob


def find_wheel_path():
    paths = [path.split(os.sep)[-1] for path in glob.glob("wheels/*")]
    user_os = platform.system().lower()

    for path in paths:
        if user_os in path and platform.machine() in path:
            return path

    print(paths)
    print(user_os, platform.machine())

    raise Exception("Couldn't find the path")


def install_transfer():
    wheel_path = find_wheel_path()

    command = [
        "pip",
        "install",
        "gams",
        "--find-links",
        "wheels" + os.sep + wheel_path,
        "--user",
    ]

    subprocess.run(command)


def install_gamspy():
    subprocess.run(["python", "setup.py", "bdist_wheel"])

    command = [
        "pip",
        "install",
        "gamspy",
        "--find-links",
        "dist" + os.sep,
        "--force-reinstall",
        "--user",
    ]

    subprocess.run(command)


if __name__ == "__main__":
    install_transfer()
    install_gamspy()
