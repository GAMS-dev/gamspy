from setuptools import setup, Distribution
import platform
import os


with open("requirements.txt") as f:
    requirements = f.read().splitlines()


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


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
    distclass=BinaryDistribution,
    install_requires=requirements,
)
