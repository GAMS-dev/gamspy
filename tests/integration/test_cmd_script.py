import unittest
import subprocess
import os
from pathlib import Path
import gamspy.utils as utils


class CmdSuite(unittest.TestCase):
    def test_install_license(self):
        gamspy_base_dir = utils._getGAMSPyBaseDirectory()
        old_license_modified_time = os.path.getmtime(
            gamspy_base_dir + os.sep + "gamslice.txt"
        )

        this_folder = str(Path(__file__).parent)

        process = subprocess.run(
            " ".join(
                [
                    "gamspy",
                    "install",
                    "license",
                    os.path.abspath(this_folder + os.sep + "gamslice.txt"),
                ]
            ),
            shell=True,
            capture_output=True,
        )

        print(process.stderr)
        print(process.stdout)

        new_license_modified_time = os.path.getmtime(
            gamspy_base_dir + os.sep + "gamslice.txt"
        )

        print(new_license_modified_time, old_license_modified_time)
        self.assertTrue(new_license_modified_time > old_license_modified_time)


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
