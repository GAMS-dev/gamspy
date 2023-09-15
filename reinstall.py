import subprocess


def install_transfer():
    command = [
        "pip",
        "install",
        "gams",
        "--find-links",
        "wheels",
        "--force-reinstall",
        "--user",
    ]

    subprocess.run(command)


def install_gamspy():
    subprocess.run(["python", "setup.py", "bdist_wheel"])

    command = [
        "pip",
        "install",
        "gamspy[dev,test]",
        "--find-links",
        "dist",
        "--force-reinstall",
        "--user",
    ]

    subprocess.run(command)


if __name__ == "__main__":
    install_transfer()
    install_gamspy()
