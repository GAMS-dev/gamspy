import fcntl
import os
import platform
import subprocess
import sys


def lock_file(f):
    fcntl.lockf(f, fcntl.LOCK_EX)


def unlock_file(f):
    fcntl.lockf(f, fcntl.LOCK_UN)


class AtomicOpen:
    def __init__(self, path, *args, **kwargs):
        self.file = open(path, *args, **kwargs)
        lock_file(self.file)

    def __enter__(self, *args, **kwargs):
        return self.file

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        self.file.flush()
        os.fsync(self.file.fileno())
        unlock_file(self.file)
        self.file.close()

        if exc_type is not None:
            return False
        else:
            return True


def main():
    operating_system = platform.system().lower()
    if operating_system != "darwin":
        sys.stderr.write(f"Expected OS: darwin, found: {operating_system}\n")
        sys.exit(1)

    if len(sys.argv) != 3:
        sys.stderr.write("Expected arguments: env_name and python=x\n")
        sys.stderr.write("python conda_create.py myenv python=3.9\n")
        sys.exit(1)

    with AtomicOpen("/tmp/conda_lock", "w"):
        subprocess.run(
            ["conda", "create", "-y", "-p", sys.argv[1], sys.argv[2]],
            check=True,
        )


if __name__ == "__main__":
    main()
