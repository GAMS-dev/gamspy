import unittest
import subprocess
import os
import glob
from pathlib import Path


class ModelsSuite(unittest.TestCase):
    def test_full_models(self):
        paths = glob.glob(
            str(Path(__file__).parent) + os.sep + "models" + os.sep + "*.py"
        )

        print()
        for idx, path in enumerate(paths):
            print(f"[{idx + 1}/{len(paths)}] {path.split(os.sep)[-1]}")
            try:
                process = subprocess.run(
                    ["python", path], check=True, capture_output=True
                )

                self.assertTrue(process.returncode == 0)
            except subprocess.CalledProcessError as e:
                print("(x)")
                print(f"Output: {e.stderr.decode('utf-8')}")
                exit(1)


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
