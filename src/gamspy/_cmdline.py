import shutil
import argparse
import gamspy.utils as utils


def get_args():
    parser = argparse.ArgumentParser(prog="gamspy")
    parser.add_argument("command", choices=["install"], type=str)
    parser.add_argument("component", choices=["license", "solver"], type=str)
    parser.add_argument("name", type=str)

    return parser.parse_args()


def install_license(args: argparse.Namespace):
    minigams_dir = utils._getSystemDirectory()
    shutil.copy(args.name, minigams_dir)


def install_solver(args: argparse.Namespace):
    ...


def install(args: argparse.Namespace):
    if args.component == "license":
        install_license(args)
    elif args.component == "solver":
        install_solver(args)


def main():
    """
    Entry point for gamspy command line application.

    Parameters
    ----------
    argv : Sequence[str], optional
        Arguments passed to the command line application
    """
    args = get_args()

    if args.command == "install":
        install(args)
