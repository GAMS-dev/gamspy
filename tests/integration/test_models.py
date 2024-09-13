import glob
import os
import subprocess
import unittest
from pathlib import Path


class ModelsSuite(unittest.TestCase):
    def test_full_models(self):
        print("Running the models with the following license:")
        process = subprocess.run(
            ["gamspy", "show", "license"], capture_output=True
        )
        self.assertTrue(process.returncode == 0)
        print(process.stdout)

        paths = glob.glob(
            str(Path(__file__).parent) + os.sep + "models" + os.sep + "*.py"
        )

        print()
        for idx, path in enumerate(paths):
            print(f"[{idx + 1}/{len(paths)}] {path.split(os.sep)[-1]}")
            process = subprocess.run(["python", path], capture_output=True)
            print(process.stdout.decode())
            print(process.stderr.decode())

            self.assertTrue(process.returncode == 0)


def gams_models_suite():
    suite = unittest.TestSuite()
    tests = [
        ModelsSuite(name)
        for name in dir(ModelsSuite)
        if name.startswith("test_")
    ]
    suite.addTests(tests)

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(gams_models_suite())
