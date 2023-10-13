from setuptools import setup


setup(
    packages=["gamspy"],
    package_dir={"gamspy": "src/gamspy"},
    install_requires=["gamsapi[transfer]==45.0.0", "gamspy_base==45.0.0"],
)
