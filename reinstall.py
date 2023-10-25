import os
import subprocess
import tempfile

if os.path.exists(os.getcwd() + os.sep + ".env"):
    from dotenv import load_dotenv

    load_dotenv(os.getcwd() + os.sep + ".env")


def install_gamspy():
    subprocess.run(["python", "-m", "build"])

    command = [
        "pip",
        "install",
        "dist/gamspy-0.10.0-py3-none-any.whl[dev,test]",
        "--force-reinstall",
    ]

    subprocess.run(command, check=True)


def install_gams_license():
    lice = os.environ["GAMS_LICENSE"]
    command = ["gamspy", "install", "license"]

    try:
        f = tempfile.NamedTemporaryFile(mode="wt", suffix=".txt", delete=False)
        f.write(lice)
        f.close()
        command.append(f.name)
        subprocess.run(command, check=True)
    finally:
        os.unlink(f.name)


if __name__ == "__main__":
    install_gamspy()
    install_gams_license()
