from __future__ import annotations

import os
import subprocess
import unittest

import gamspy.utils as utils
from gamspy.exceptions import ValidationError

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
                os.environ["LOCAL07"],
            ],
            check=True,
        )

        self.assertTrue(
            os.path.exists(
                os.path.join(gamspy_base_directory, "user_license.txt")
            )
        )

        _ = subprocess.run(
            [
                "gamspy",
                "install",
                "license",
                os.environ["LOCAL07"],
                "--node-specific",
            ],
            check=True,
        )

        self.assertTrue(
            os.path.exists(
                os.path.join(gamspy_base_directory, "node_info.json")
            )
        )

        with self.assertRaises(ValidationError):
            _ = subprocess.run(
                ["gamspy", "install", "license", "blabla"],
                check=True,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )

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
                os.environ["LOCAL07"],
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
