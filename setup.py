from setuptools import setup, Distribution
import platform
import os


with open("requirements.txt") as f:
    requirements = f.read().splitlines()


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(
    packages=["gamspy"],
    package_dir={"gamspy": "src/gamspy"},
    distclass=BinaryDistribution,
    install_requires=requirements,
)
