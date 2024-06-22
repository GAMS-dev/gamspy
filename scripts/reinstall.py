import argparse
import os
import subprocess

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
    command = ["gamspy", "install", "license", os.environ["LOCAL_LICENSE"]]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    args = get_args()
    install_gamspy(args)
    install_gams_license()
