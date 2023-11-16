import argparse
import os
import subprocess
import tempfile

if os.path.exists(os.getcwd() + os.sep + ".env"):
    from dotenv import load_dotenv

    load_dotenv(os.getcwd() + os.sep + ".env")


def get_args():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--from-pypi", action="store_true")
    return argument_parser.parse_args()


def install_gamspy(args):
    if args.from_pypi:
        subprocess.run(
            ["pip", "install", "gamspy[dev,test]", "--force-reinstall"]
        )
        return

    subprocess.run(["python", "-m", "build"])

    command = [
        "pip",
        "install",
        ".[dev,test]",
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
    args = get_args()
    install_gamspy(args)
    install_gams_license()
