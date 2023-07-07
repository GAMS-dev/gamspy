import os
import subprocess
import sys

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
