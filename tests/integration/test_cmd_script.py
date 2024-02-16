from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path

import gamspy.utils as utils


class CmdSuite(unittest.TestCase):
    def test_install_license(self):
        this_folder = str(Path(__file__).parent)
        gamspy_base_directory = utils._get_gamspy_base_directory()

        license = "dummy license"

        with open(
            this_folder + os.sep + "my_license.txt", "w"
        ) as license_file:
            license_file.write(license)

        _ = subprocess.run(
            [
                "gamspy",
                "install",
                "license",
                this_folder + os.sep + "my_license.txt",
            ],
            check=True,
        )

        self.assertTrue(
            os.path.exists(
                os.path.join(gamspy_base_directory, "user_license.txt")
            )
        )

        with self.assertRaises(subprocess.CalledProcessError):
            _ = subprocess.run(
                ["gamspy", "install", "license", "blabla"],
                check=True,
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

    def test_install_solver(self):
        with self.assertRaises(subprocess.CalledProcessError):
            _ = subprocess.run(
                ["gamspy", "install", "solver", "bla"],
                check=True,
            )

    def test_uninstall_solver(self):
        with self.assertRaises(subprocess.CalledProcessError):
            _ = subprocess.run(
                ["gamspy", "uninstall", "solver", "bla"],
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
