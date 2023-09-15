import os
import platform

from setuptools import setup


with open("requirements.txt") as f:
    requirements = f.read().splitlines()


def get_minigams_path():
    user_os = platform.system().lower()
    minigams_path = "minigams" + os.sep + user_os

    if user_os == "darwin":
        minigams_path += f"_{platform.machine()}"

    minigams_path += os.sep + "*"
    return minigams_path


setup(
    packages=["gamspy"],
    package_dir={"gamspy": "src/gamspy"},
    package_data={"gamspy": [get_minigams_path()]},
    install_requires=requirements,
)
