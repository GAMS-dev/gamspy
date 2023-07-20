import os
import subprocess
import sys

if not os.path.exists("gams-transfer-python"):
    process = subprocess.run(
        "git clone --recurse-submodules"
        " git@git.gams.com:devel/gams-transfer-python.git",
        shell=True,
    )
    process = subprocess.run(
        "cd gams-transfer-python && python setup.py bdist_wheel && pip install"
        " gams[transfer] --find-links dist/",
        shell=True,
    )

try:
    import gams.transfer  # noqa: F401
except Exception:
    sys.exit("Gotta install Gams Transfer first bruh")

process = subprocess.run(["python", "-m", "build", "--wheel"])

process = subprocess.run(
    [
        "pip",
        "install",
        "dist" + os.sep + "gamspy-0.0.1-py3-none-any.whl[dev,test]",
        "--force-reinstall",
    ]
)
