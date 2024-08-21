from __future__ import annotations

import os
import shutil
import subprocess
import time
import unittest

import gamspy.utils as utils
from gamspy import Container

try:
    from dotenv import load_dotenv

    load_dotenv(os.getcwd() + os.sep + ".env")
except Exception:
    pass


class CmdSuite(unittest.TestCase):
    def test_install_license(self):
        gamspy_base_directory = utils._get_gamspy_base_directory()

        _ = subprocess.run(
            [
                "gamspy",
                "install",
                "license",
                os.environ["NETWORK_LICENSE"],
            ],
            check=True,
        )

        user_license_path = os.path.join(
            gamspy_base_directory, "user_license.txt"
        )

        self.assertTrue(os.path.exists(user_license_path))

        m = Container()
        self.assertTrue(m._network_license)
        m.close()
        time.sleep(1)
        _ = subprocess.run(
            ["gamspy", "install", "license", os.environ["LOCAL_LICENSE"]],
            check=True,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )

        m = Container()
        self.assertFalse(m._network_license)

        with self.assertRaises(subprocess.CalledProcessError):
            _ = subprocess.run(
                ["gamspy", "install", "license", "blabla"],
                check=True,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )

        tmp_license_path = os.path.join("tmp", "user_license.txt")
        shutil.copy(user_license_path, tmp_license_path)

        _ = subprocess.run(
            [
                "gamspy",
                "uninstall",
                "license",
            ],
            check=True,
        )

        _ = subprocess.run(
            [
                "gamspy",
                "install",
                "license",
                tmp_license_path,
            ],
            check=True,
        )

        self.assertTrue(os.path.exists(user_license_path))

    def test_uninstall_license(self):
        gamspy_base_directory = utils._get_gamspy_base_directory()

        _ = subprocess.run(
            ["gamspy", "uninstall", "license"],
            check=True,
        )

        self.assertFalse(
            os.path.exists(
                os.path.join(gamspy_base_directory, "user_license.txt")
            )
        )

        # Recover the license
        subprocess.run(
            [
                "gamspy",
                "install",
                "license",
                os.environ["LOCAL_LICENSE"],
            ]
        )

    def test_install_solver(self):
        with self.assertRaises(subprocess.CalledProcessError):
            _ = subprocess.run(
                ["gamspy", "install", "solver", "bla"],
                check=True,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )

    def test_uninstall_solver(self):
        with self.assertRaises(subprocess.CalledProcessError):
            _ = subprocess.run(
                ["gamspy", "uninstall", "solver", "bla"],
                check=True,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )

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
            ["gamspy", "probe", "-o", node_info_path],
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

        self.assertTrue(process.returncode == 0)


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
