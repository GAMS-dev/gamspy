from setuptools import setup
import platform
import os



setup(
    packages=["gamspy"],
    package_dir={"gamspy": "src/gamspy"},
    install_requires=[
        "gamsapi==45.0.0",
        "gamspy_base==45.0.0"
    ],
)
