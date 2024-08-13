import fcntl
import platform
import subprocess
import sys
import time
from contextlib import contextmanager


@contextmanager
def locker(filename: str):
    start_time = time.time()
    with open(filename, "w") as file:
        fcntl.lockf(file, fcntl.LOCK_EX)
        yield
        fcntl.lockf(file, fcntl.LOCK_UN)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(
        f"Start time: {start_time}, End time: {end_time}, Elapsed time: {elapsed_time} seconds"
    )


def main():
    operating_system = platform.system().lower()
    if operating_system != "darwin":
        sys.stderr.write(f"Expected OS: darwin, found: {operating_system}\n")
        sys.exit(1)

    if len(sys.argv) != 3:
        sys.stderr.write("Expected arguments: env_name and python=x\n")
        sys.stderr.write("python conda_create.py myenv python=3.9\n")
        sys.exit(1)

    with locker("/tmp/conda_lock"):
        subprocess.run(
            [
                "conda",
                "env",
                "create",
                "-y",
                "-f",
                f"scripts/env_{sys.argv[2].split('=')[-1]}.yml",
                "-p",
                sys.argv[1],
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
