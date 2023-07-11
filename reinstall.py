import os
import subprocess
import sys

process = subprocess.run(
    (
        "git clone --recurse-submodules"
        " git@git.gams.com:devel/gams-transfer-python.git"
    ),
    shell=True,
)
process = subprocess.run(
    "cd gams-transfer-python && python reinstall.py", shell=True
)

try:
    import gams.transfer  # noqa: F401
except Exception:
    sys.exit("Gotta install Gams Transfer first bruh")

process = subprocess.run(["pip", "install", "-r", "requirements.txt"])
process = subprocess.run(["python", "-m", "build"])

process = subprocess.run(
    [
        "pip",
        "install",
        "dist" + os.sep + "gamspy-0.0.1-py3-none-any.whl",
        "--force-reinstall",
    ]
)
