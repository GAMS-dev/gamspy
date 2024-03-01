from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import unittest

from gamspy import Container, Set, Parameter


class MiroSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
            delayed_execution=int(os.getenv("DELAYED_EXECUTION", False)),
        )

    def test_domain_forwarding(self):
        # Only the parameter is miro input
        m = Container()
        i = Set(m, name="i")
        p = Parameter(
            m,
            name="p",
            domain=[i],
            domain_forwarding=True,
            records=[["i1", 1]],
            is_miro_input=True,
        )
        self.assertEqual(i.records.values.tolist()[0][0], "i1")
        self.assertEqual(p.records.values[0][1], 1)

        # Both are miro input
        m2 = Container()
        i2 = Set(m2, name="i2", is_miro_input=True)
        p2 = Parameter(
            m2,
            name="p2",
            domain=[i2],
            domain_forwarding=True,
            records=[["i2", 1]],
            is_miro_input=True,
        )
        self.assertEqual(i2.records.values.tolist()[0][0], "i2")
        self.assertEqual(p2.records.values[0][1], 1)

    def test_miro(self):
        directory = str(pathlib.Path(__file__).parent.resolve())
        current_environment = os.environ.copy()
        current_environment["MIRO"] = "1"

        try:
            subprocess.run(
                [
                    sys.executable,
                    directory + os.sep + "miro_models" + os.sep + "miro.py",
                ],
                env=current_environment,
                check=True,
                capture_output=True,
            )
        except Exception as e:
            exit(e)

        # Test default.gdx
        new_container = Container()
        new_container.read(
            f"{directory}{os.sep}miro_models{os.sep}data_miro{os.sep}default.gdx"
        )

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
        with open(
            f"{directory}{os.sep}miro_models{os.sep}conf_miro{os.sep}miro_io.json"
        ) as file:
            contract = json.load(file)
            self.assertEqual(
                contract,
                {
                    "modelTitle": "GAMSPy App",
                    "inputSymbols": {
                        "d": {
                            "alias": "distance in thousands of miles",
                            "symtype": "parameter",
                            "headers": {
                                "i": {
                                    "type": "string",
                                    "alias": "canning plants",
                                },
                                "new-york": {
                                    "type": "numeric",
                                    "alias": "new-york",
                                },
                                "chicago": {
                                    "type": "numeric",
                                    "alias": "chicago",
                                },
                                "topeka": {
                                    "type": "numeric",
                                    "alias": "topeka",
                                },
                            },
                        },
                        "_scalars": {
                            "alias": "Input Scalars",
                            "symnames": ["model_type", "f"],
                            "symtext": [
                                "model_type",
                                (
                                    "freight in dollars per case per thousand"
                                    " miles"
                                ),
                            ],
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
                            "alias": "shipment quantities in cases",
                            "symtype": "variable",
                            "headers": {
                                "i": {
                                    "type": "string",
                                    "alias": "canning plants",
                                },
                                "j": {"type": "string", "alias": "markets"},
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
                            "symtext": [
                                "total transportation costs in thousands of"
                                " dollars"
                            ],
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

    def test_contract(self):
        directory = str(pathlib.Path(__file__).parent.resolve())
        current_environment = os.environ.copy()
        current_environment["MIRO"] = "1"
        try:
            subprocess.run(
                [
                    sys.executable,
                    directory + os.sep + "miro_models" + os.sep + "miro2.py",
                ],
                env=current_environment,
                check=True,
                capture_output=True,
            )
        except Exception as e:
            exit(e)

        with open(
            f"{directory}{os.sep}miro_models{os.sep}conf_miro2{os.sep}miro2_io.json"
        ) as file:
            contract = json.load(file)
            self.assertEqual(
                contract,
                {
                    "modelTitle": "GAMSPy App",
                    "inputSymbols": {
                        "b": {
                            "alias": "demand at market j in cases",
                            "symtype": "parameter",
                            "headers": {
                                "j": {"type": "string", "alias": "markets"},
                                "value": {"type": "numeric", "alias": "value"},
                            },
                        },
                        "d": {
                            "alias": "distance in thousands of miles",
                            "symtype": "parameter",
                            "headers": {
                                "I": {
                                    "type": "string",
                                    "alias": "canning plants",
                                },
                                "j": {"type": "string", "alias": "markets"},
                                "value": {"type": "numeric", "alias": "value"},
                            },
                        },
                        "jlocdata": {
                            "alias": "Market location information",
                            "symtype": "parameter",
                            "headers": {
                                "j": {"type": "string", "alias": "markets"},
                                "lat": {"type": "numeric", "alias": "lat"},
                                "lnG": {"type": "numeric", "alias": "lnG"},
                            },
                        },
                        "a": {
                            "alias": "capacity of plant i in cases",
                            "symtype": "parameter",
                            "headers": {
                                "I": {
                                    "type": "string",
                                    "alias": "canning plants",
                                },
                                "value": {"type": "numeric", "alias": "value"},
                            },
                        },
                        "ilocdata": {
                            "alias": "Plant location information",
                            "symtype": "parameter",
                            "headers": {
                                "I": {
                                    "type": "string",
                                    "alias": "canning plants",
                                },
                                "lat": {"type": "numeric", "alias": "lat"},
                                "lnG": {"type": "numeric", "alias": "lnG"},
                            },
                        },
                        "k": {
                            "alias": "k",
                            "symtype": "set",
                            "headers": {
                                "uni": {"type": "string", "alias": "uni"},
                                "element_text": {
                                    "type": "string",
                                    "alias": "element_text",
                                },
                            },
                        },
                        "_scalars": {
                            "alias": "Input Scalars",
                            "symnames": ["type", "f", "mins", "beta"],
                            "symtext": [
                                "selected model type",
                                (
                                    "freight in dollars per case per thousand"
                                    " miles"
                                ),
                                "minimum shipment (MIP- and MINLP-only)",
                                "beta (MINLP-only)",
                            ],
                            "symtypes": [
                                "set",
                                "parameter",
                                "parameter",
                                "parameter",
                            ],
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
                        "schedule": {
                            "alias": "shipment quantities in cases",
                            "symtype": "parameter",
                            "headers": {
                                "I": {
                                    "type": "string",
                                    "alias": "canning plants",
                                },
                                "j": {"type": "string", "alias": "markets"},
                                "lngP": {"type": "numeric", "alias": "lngP"},
                                "latP": {"type": "numeric", "alias": "latP"},
                                "lngM": {"type": "numeric", "alias": "lngM"},
                                "latM": {"type": "numeric", "alias": "latM"},
                                "cap": {"type": "numeric", "alias": "cap"},
                                "demand": {
                                    "type": "numeric",
                                    "alias": "demand",
                                },
                                "quantities": {
                                    "type": "numeric",
                                    "alias": "quantities",
                                },
                            },
                        },
                        "_scalars_out": {
                            "alias": "Output Scalars",
                            "symnames": ["total_cost"],
                            "symtext": [
                                "total transportation costs in thousands of"
                                " dollars"
                            ],
                            "symtypes": ["parameter"],
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
                },
            )

    def test_table_columns(self):
        directory = str(pathlib.Path(__file__).parent.resolve())
        miro_gdx_in = os.path.join(directory, "_test_miro_gdxin_.gdx")
        miro_gdx_out = os.path.join(directory, "_test_miro_gdxout_.gdx")
        model_path = os.path.join(directory, "miro_models", "miro3.py")

        subprocess_env = os.environ.copy()
        subprocess_env["MIRO"] = "1"
        subprocess_env["MIRO_MODEL_PATH"] = model_path
        subprocess_env["MIRO_MODE"] = "base"
        subprocess_env["MIRO_DEV_MODE"] = "true"
        subprocess_env["MIRO_USE_TMP"] = "false"
        subprocess_env["PYTHON_EXEC_PATH"] = sys.executable
        subprocess_env["GAMS_IDC_GDX_INPUT"] = miro_gdx_in
        subprocess_env["GAMS_IDC_GDX_OUTPUT"] = miro_gdx_out

        try:
            subprocess.run(
                ["python", model_path], env=subprocess_env, check=True
            )
        except subprocess.CalledProcessError:
            self.fail("Columns are not as expected.")


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
