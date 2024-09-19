from __future__ import annotations

import os
import platform
import shutil
import subprocess
import unittest

from gamspy import Container, Set

try:
    from dotenv import load_dotenv

    load_dotenv(os.getcwd() + os.sep + ".env")
except Exception:
    pass

user_dir = os.path.expanduser("~")
if platform.system() == "Linux":
    DEFAULT_DIR = os.path.join(user_dir, ".local", "share", "GAMSPy")
elif platform.system() == "Darwin":
    DEFAULT_DIR = os.path.join(
        user_dir, "Library", "Application Support", "GAMSPy"
    )
elif platform.system() == "Windows":
    DEFAULT_DIR = os.path.join(user_dir, "Documents", "GAMSPy")


class CmdSuite(unittest.TestCase):
    def test_install_license(self):
        m = Container()
        self.assertFalse(m._network_license)

        # Test network license
        _ = subprocess.run(
            [
                "gamspy",
                "install",
                "license",
                os.environ["NETWORK_LICENSE"],
            ],
            check=True,
        )

        gamspy_license_path = os.path.join(DEFAULT_DIR, "gamspy_license.txt")

        self.assertTrue(os.path.exists(gamspy_license_path))

        m = Container()
        self.assertTrue(m._network_license)

        _ = Set(m, "i", records=["bla"])
        m.close()

        # Test invalid access code / license
        with self.assertRaises(subprocess.CalledProcessError):
            _ = subprocess.run(
                ["gamspy", "install", "license", "blabla"],
                check=True,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )

        # Test installing a license from a file path.
        tmp_license_path = os.path.join("tmp", "gamspy_license.txt")
        shutil.copy(gamspy_license_path, tmp_license_path)

        _ = subprocess.run(
            [
                "gamspy",
                "install",
                "license",
                tmp_license_path,
            ],
            check=True,
        )

        self.assertTrue(os.path.exists(gamspy_license_path))

    def test_install_solver(self):
        with self.assertRaises(subprocess.CalledProcessError):
            _ = subprocess.run(
                ["gamspy", "install", "solver", "bla"],
                check=True,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )

        process = subprocess.run(
            ["gamspy", "install", "solver", "minos", "mosek"],
            capture_output=True,
            text=True,
        )
        print(process.stdout, process.stderr)
        self.assertTrue(process.returncode == 0)

        with self.assertRaises(subprocess.CalledProcessError):
            _ = subprocess.run(
                ["gamspy", "uninstall", "solver", "bla"],
                check=True,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )

        process = subprocess.run(
            ["gamspy", "install", "solver", "--install-all-solvers"],
            capture_output=True,
            text=True,
        )
        print(process.stdout, process.stderr)
        self.assertTrue(process.returncode == 0)

        process = subprocess.run(
            ["gamspy", "uninstall", "solver", "minos", "mosek"],
            capture_output=True,
            text=True,
        )
        print(process.stdout, process.stderr)
        self.assertTrue(process.returncode == 0)

        process = subprocess.run(
            ["gamspy", "uninstall", "solver", "--uninstall-all-solvers"],
            capture_output=True,
            text=True,
        )
        print(process.stdout, process.stderr)
        self.assertTrue(process.returncode == 0)

        process = subprocess.run(
            ["gamspy", "install", "solver", "mpsge", "scip"],
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )
        self.assertTrue(process.returncode == 0)

    def test_list_solvers(self):
        process = subprocess.run(
            ["gamspy", "list", "solvers"],
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )

        self.assertTrue(process.returncode == 0)

        process = subprocess.run(
            ["gamspy", "list", "solvers", "-a"],
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )

        self.assertTrue(process.returncode == 0)

    def test_show_license(self):
        process = subprocess.run(
            ["gamspy", "show", "license"],
            stderr=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            text=True,
        )

        self.assertTrue(process.returncode == 0)
        self.assertTrue(isinstance(process.stdout, str))

    def test_show_base(self):
        process = subprocess.run(
            ["gamspy", "show", "base"],
            stderr=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            text=True,
        )

        import gamspy_base

        self.assertTrue(process.returncode == 0)
        self.assertEqual(gamspy_base.directory, process.stdout.strip())

    def test_probe(self):
        node_info_path = os.path.join("tmp", "info.json")
        process = subprocess.run(
            ["gamspy", "probe", "-j", node_info_path],
            capture_output=True,
            text=True,
        )

        self.assertTrue(process.returncode == 0)

        process = subprocess.run(
            [
                "gamspy",
                "retrieve",
                "license",
                os.environ["LOCAL_LICENSE"],
                "-i",
                node_info_path,
                "-o",
                node_info_path[:-5] + ".txt",
            ],
            capture_output=True,
            text=True,
        )

        print(process.stderr, process.stdout)
        self.assertTrue(process.returncode == 0)

    @classmethod
    def tearDown(cls):
        _ = subprocess.run(
            ["gamspy", "install", "license", os.environ["LOCAL_LICENSE"]],
            check=True,
        )


def cmd_suite():
    suite = unittest.TestSuite()
    tests = [
        CmdSuite(name) for name in dir(CmdSuite) if name.startswith("test_")
    ]
    suite.addTests(tests)

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(cmd_suite())
