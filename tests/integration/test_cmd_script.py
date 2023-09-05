import unittest
import subprocess
import os
from pathlib import Path
import gamspy.utils as utils


class CmdSuite(unittest.TestCase):
    def test_install_license(self):
        minigams_dir = utils._getMinigamsDirectory()
        old_license_modified_time = os.path.getmtime(
            minigams_dir + os.sep + "gamslice.txt"
        )

        this_folder = str(Path(__file__).parent)

        process = subprocess.run(
            [
                "gamspy",
                "install",
                "license",
                this_folder + os.sep + "gamslice.txt",
            ],
        )

        self.assertTrue(process.returncode == 0)

        new_license_modified_time = os.path.getmtime(
            minigams_dir + os.sep + "gamslice.txt"
        )
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
