from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import unittest

from gamspy import Container


class MiroSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container(delayed_execution=True)

    def test_miro(self):
        directory = str(pathlib.Path(__file__).parent.resolve())
        current_environment = os.environ.copy()
        current_environment["MIRO"] = "1"

        try:
            subprocess.run(
                [sys.executable, directory + os.sep + "miro.py"],
                env=current_environment,
                check=True,
                capture_output=True,
            )
        except Exception as e:
            print(e)

        # Test default.gdx
        new_container = Container()
        new_container.read(f"{directory}{os.sep}data_miro{os.sep}default.gdx")

        # Miro input d
        self.assertTrue("d" in new_container.data.keys())
        self.assertEqual(
            new_container["d"].records.values.tolist(),
            [
                ["seattle", "new-york", 2.5],
                ["seattle", "chicago", 1.7],
                ["seattle", "topeka", 1.8],
                ["san-diego", "new-york", 2.5],
                ["san-diego", "chicago", 1.8],
                ["san-diego", "topeka", 1.4],
            ],
        )

        # Miro scalar input f
        self.assertTrue("f" in new_container.data.keys())
        self.assertEqual(new_container["f"].records.value.item(), 90.0)

        # Miro output x
        self.assertTrue("x" in new_container.data.keys())
        self.assertEqual(
            new_container["x"].records.values.tolist(),
            [
                ["seattle", "new-york", 50.0, 0.0, 0.0, float("inf"), 1.0],
                ["seattle", "chicago", 300.0, 0.0, 0.0, float("inf"), 1.0],
                [
                    "seattle",
                    "topeka",
                    0.0,
                    0.036000000000000004,
                    0.0,
                    float("inf"),
                    1.0,
                ],
                ["san-diego", "new-york", 275.0, 0.0, 0.0, float("inf"), 1.0],
                [
                    "san-diego",
                    "chicago",
                    0.0,
                    0.009000000000000008,
                    0.0,
                    float("inf"),
                    1.0,
                ],
                ["san-diego", "topeka", 275.0, 0.0, 0.0, float("inf"), 1.0],
            ],
        )

        # Miro output z
        self.assertTrue("z" in new_container.data.keys())
        self.assertEqual(new_container["z"].records.level.item(), 153.675)

        # Test generated json
        with open(f"{directory}{os.sep}conf_miro{os.sep}miro_io.json") as file:
            contract = json.load(file)
            self.assertEqual(
                contract,
                {
                    "modelTitle": "GAMSPy App",
                    "inputSymbols": {
                        "d": {
                            "alias": "d",
                            "symtype": "parameter",
                            "headers": {
                                "i": {"type": "string", "alias": "i"},
                                "j": {"type": "string", "alias": "j"},
                                "value": {"type": "numeric", "alias": "value"},
                            },
                        },
                        "_scalars": {
                            "alias": "Input Scalars",
                            "symnames": ["model_type", "f"],
                            "symtext": ["model_type", "f"],
                            "symtypes": ["set", "parameter"],
                            "headers": {
                                "scalar": {
                                    "type": "string",
                                    "alias": "Scalar Name",
                                },
                                "description": {
                                    "type": "string",
                                    "alias": "Scalar Description",
                                },
                                "value": {
                                    "type": "string",
                                    "alias": "Scalar Value",
                                },
                            },
                        },
                    },
                    "outputSymbols": {
                        "x": {
                            "alias": "x",
                            "symtype": "variable",
                            "headers": {
                                "i": {"type": "string", "alias": "i"},
                                "j": {"type": "string", "alias": "j"},
                                "level": {"type": "numeric", "alias": "level"},
                                "marginal": {
                                    "type": "numeric",
                                    "alias": "marginal",
                                },
                                "lower": {"type": "numeric", "alias": "lower"},
                                "upper": {"type": "numeric", "alias": "upper"},
                                "scale": {"type": "numeric", "alias": "scale"},
                            },
                        },
                        "_scalarsve_out": {
                            "alias": "Output Variable/Equation Scalars",
                            "symnames": ["z"],
                            "symtext": ["z"],
                            "symtypes": ["variable"],
                            "headers": {
                                "scalar": {
                                    "type": "string",
                                    "alias": "Scalar Name",
                                },
                                "description": {
                                    "type": "string",
                                    "alias": "Scalar Description",
                                },
                                "level": {"type": "numeric", "alias": "Level"},
                                "marginal": {
                                    "type": "numeric",
                                    "alias": "Marginal",
                                },
                                "lower": {"type": "numeric", "alias": "Lower"},
                                "upper": {"type": "numeric", "alias": "Upper"},
                                "scale": {"type": "numeric", "alias": "Scale"},
                            },
                        },
                    },
                },
            )


def miro_suite():
    suite = unittest.TestSuite()
    tests = [
        MiroSuite(name) for name in dir(MiroSuite) if name.startswith("test_")
    ]
    suite.addTests(tests)

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(miro_suite())
