import os
import shutil
import subprocess
import unittest
from pathlib import Path

import gamspy.utils as utils


class CmdSuite(unittest.TestCase):
    def test_install_license(self):
        this_folder = str(Path(__file__).parent)

        license = "dummy license"

        with open(this_folder + os.sep + "gamslice.txt", "w") as license_file:
            license_file.write(license)

        gamspy_base_dir = utils._getGAMSPyBaseDirectory()

        # copy existing license to recover later
        shutil.copy(
            gamspy_base_dir + os.sep + "gamslice.txt",
            this_folder + os.sep + "backup.txt",
        )

        old_license_modified_time = os.path.getmtime(
            gamspy_base_dir + os.sep + "gamslice.txt"
        )

        _ = subprocess.run(
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

        new_license_modified_time = os.path.getmtime(
            gamspy_base_dir + os.sep + "gamslice.txt"
        )

        self.assertTrue(new_license_modified_time > old_license_modified_time)

        # recover the original license
        shutil.copyfile(
            this_folder + os.sep + "backup.txt",
            gamspy_base_dir + os.sep + "gamslice.txt",
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
