import os
import platform
import subprocess
import sys
import tempfile


platform_to_job_prefix = {
    "windows": "test-wei-",
    "linux": "test-leg-",
    "mac_x86_64": "test-deg-",
    "mac_arm64": "test-dac-",
}


def get_default_platform():
    operating_system = platform.system().lower()
    architecture = platform.machine()

    if operating_system == "darwin":
        return f"mac_{architecture}"

    return operating_system


def get_job_name():
    job_prefix = platform_to_job_prefix[get_default_platform()]
    major = sys.version_info.major
    minor = sys.version_info.minor
    return f"{job_prefix}{major}.{minor}"


def get_artifacts():
    token = os.environ["gamspy_base_token"]
    repo = os.environ["gamspy_base_artifacts"]
    job_name = get_job_name()
    repo = repo.replace("job_name", job_name)

    command = [
        "curl",
        "--location",
        "--output",
        "artifacts.zip",
        "--header",
        f"PRIVATE-TOKEN: {token}",
        repo,
    ]

    subprocess.run(command, check=True)

    command = ["unzip", "artifacts.zip"]
    subprocess.run(command, check=True)


def install_transfer():
    command = [
        "pip",
        "install",
        "gamsapi[transfer, connect]",
        "--find-links",
        ".",
        "--force-reinstall",
    ]

    subprocess.run(command, check=True)


def install_gamspy_base():
    command = [
        "pip",
        "install",
        "gamspy_base",
        "--find-links",
        "gamspy_base/dist",
        "--force-reinstall",
    ]

    subprocess.run(command, check=True)


def install_gamspy():
    subprocess.run(["python", "setup.py", "sdist"])

    command = [
        "pip",
        "install",
        "gamspy[dev,test]",
        "--find-links",
        "dist",
        "--force-reinstall",
        "--no-deps",
        "--no-cache",
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


def install_development_dependencies():
    command = ["pip", "install", "-r", "dev_requirements.txt"]

    subprocess.run(command, check=True)


if __name__ == "__main__":
    install_development_dependencies()
    get_artifacts()
    install_transfer()
    install_gamspy_base()
    install_gamspy()
    install_gams_license()
