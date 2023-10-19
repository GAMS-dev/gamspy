from setuptools import setup


setup(
    packages=["gamspy"],
    package_dir={"gamspy": "src/gamspy"},
    install_requires=[
        "gamsapi[transfer,control]>=45.1.0",
        "gamspy_base>=45.1.0",
    ],
)
