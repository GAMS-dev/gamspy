import shutil
import argparse
import gamspy.utils as utils


def get_args():
    parser = argparse.ArgumentParser(prog="gamspy")
    parser.add_argument("command", choices=["install"], type=str)
    parser.add_argument("component", choices=["license", "solver"], type=str)
    parser.add_argument("name", type=str)

    return vars(parser.parse_args())


def install_license(args: dict):
    minigams_dir = utils._getSystemDirectory()
    shutil.copy(args["name"], minigams_dir)


def install_solver(args: dict):
    ...


def install(args: dict):
    if args["component"] == "license":
        install_license(args)
    elif args["component"] == "solver":
        install_solver(args)


def main():
    """
    Entry point for gamspy command line application.
    """
    args = get_args()

    if args["command"] == "install":
        install(args)
