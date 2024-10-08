from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

from gamspy import Container, Options, Parameter, Set, Variable
from gamspy._miro import MiroJSONEncoder
from gamspy.exceptions import ValidationError

pytestmark = pytest.mark.integration


@pytest.fixture
def data():
    # Arrange
    m = Container()

    # Act and assert
    yield m

    # Cleanup
    m.close()
    if os.path.exists("miro.log"):
        os.remove("miro.log")


def test_domain_forwarding(data):
    m = data
    # Only the parameter is miro input
    i = Set(m, name="i")
    p = Parameter(
        m,
        name="p",
        domain=[i],
        domain_forwarding=True,
        records=[["i1", 1]],
        is_miro_input=True,
    )
    assert i.records.values.tolist()[0][0] == "i1"
    assert p.records.values[0][1] == 1

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
    assert i2.records.values.tolist()[0][0] == "i2"
    assert p2.records.values[0][1] == 1


def test_domain_forwarding_2():
    directory = str(pathlib.Path(__file__).parent.resolve())
    miro_gdx_in = os.path.join(directory, "miro_models", "_miro4_gdxin_.gdx")
    miro_gdx_out = os.path.join(directory, "miro_models", "_miro4_gdxout_.gdx")
    model_path = os.path.join(directory, "miro_models", "miro4.py")

    subprocess_env = os.environ.copy()
    subprocess_env["GAMS_IDC_GDX_INPUT"] = miro_gdx_in
    subprocess_env["GAMS_IDC_GDX_OUTPUT"] = miro_gdx_out

    try:
        subprocess.run(["python", model_path], env=subprocess_env, check=True)
    except subprocess.CalledProcessError:
        pytest.fail("Records are not as expected.")


def test_miro():
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
        pytest.fail(e)

    # Test default.gdx
    new_container = Container()
    new_container.read(
        f"{directory}{os.sep}miro_models{os.sep}data_miro{os.sep}default.gdx"
    )

    # Miro input d
    assert "d" in new_container.data
    assert new_container["d"].records.values.tolist() == [
        ["seattle", "new-york", 2.5],
        ["seattle", "chicago", 1.7],
        ["seattle", "topeka", 1.8],
        ["san-diego", "new-york", 2.5],
        ["san-diego", "chicago", 1.8],
        ["san-diego", "topeka", 1.4],
    ]

    # Miro scalar input f
    assert "f" in new_container.data
    assert new_container["f"].records.value.item() == 90.0

    # Miro output x
    assert "x" in new_container.data
    assert new_container["x"].records.values.tolist() == [
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
    ]

    # Miro output z
    assert "z" in new_container.data
    assert new_container["z"].records.level.item() == 153.675

    # Test generated json
    with open(
        f"{directory}{os.sep}miro_models{os.sep}conf_miro{os.sep}miro_io.json"
    ) as file:
        contract = json.load(file)
        assert contract == {
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
                        ("freight in dollars per case per thousand" " miles"),
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
                        "total transportation costs in thousands of" " dollars"
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
        }


def test_contract():
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
        assert contract == {
            "modelTitle": "GAMSPy App",
            "inputSymbols": {
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
                "jlocdata": {
                    "alias": "Market location information",
                    "symtype": "parameter",
                    "headers": {
                        "j": {"type": "string", "alias": "markets"},
                        "lat": {"type": "numeric", "alias": "lat"},
                        "lnG": {"type": "numeric", "alias": "lnG"},
                    },
                },
                "_scalars": {
                    "alias": "Input Scalars",
                    "symnames": ["type", "f", "mins", "beta"],
                    "symtext": [
                        "selected model type",
                        "freight in dollars per case per thousand miles",
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
                            "alias": "satisfy demand at market j",
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
                        "total transportation costs in thousands of dollars"
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
        }


def test_table_columns():
    directory = str(pathlib.Path(__file__).parent.resolve())
    miro_gdx_in = os.path.join(directory, "miro_models", "_miro3_gdxin_.gdx")
    miro_gdx_out = os.path.join(directory, "miro_models", "_miro3_gdxout_.gdx")
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
        subprocess.run(["python", model_path], env=subprocess_env, check=True)
    except subprocess.CalledProcessError:
        pytest.fail("Columns are not as expected.")


def test_miro_encoder(data):
    m = data
    # Prepare data
    distances = [
        ["seattle", "new-york", 2.5],
        ["seattle", "chicago", 1.7],
        ["seattle", "topeka", 1.8],
        ["san-diego", "new-york", 2.5],
        ["san-diego", "chicago", 1.8],
        ["san-diego", "topeka", 1.4],
    ]

    capacities = [["seattle", 350], ["san-diego", 600]]
    demands = [["new-york", 325], ["chicago", 300], ["topeka", 275]]

    i = Set(
        m,
        name="i",
        records=["seattle", "san-diego"],
        description="canning plants",
    )
    j = Set(
        m,
        name="j",
        records=["new-york", "chicago", "topeka"],
        description="markets",
    )
    _ = Set(m, name="k", is_miro_input=True)
    _ = Set(
        m,
        name="model_type",
        records=["lp"],
        is_singleton=True,
        is_miro_input=True,
    )

    _ = Parameter(
        m,
        name="a",
        domain=[i],
        records=capacities,
        description="capacity of plant i in cases",
    )
    _ = Parameter(
        m,
        name="b",
        domain=[j],
        records=demands,
        description="demand at market j in cases",
    )
    d = Parameter(
        m,
        name="d",
        domain=[i, j],
        records=distances,
        description="distance in thousands of miles",
        is_miro_input=True,
        is_miro_table=True,
    )
    _ = Parameter(
        m,
        name="table_without_records",
        domain=[i, j],
        is_miro_input=True,
        is_miro_table=True,
    )
    c = Parameter(
        m,
        name="c",
        domain=[i, j],
        description="transport cost in thousands of dollars per case",
    )
    f = Parameter(
        m,
        name="f",
        records=90,
        description="freight in dollars per case per thousand miles",
        is_miro_input=True,
    )
    c[i, j] = f * d[i, j] / 1000

    _ = Variable(
        m,
        name="x",
        domain=[i, j],
        type="Positive",
        description="shipment quantities in cases",
        is_miro_output=True,
    )
    _ = Variable(
        m,
        name="z",
        description="total transportation costs in thousands of dollars",
        is_miro_output=True,
    )

    encoder = MiroJSONEncoder(m)
    generated_json = encoder.write_json()
    assert generated_json == {
        "modelTitle": "GAMSPy App",
        "inputSymbols": {
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
            "d": {
                "alias": "distance in thousands of miles",
                "symtype": "parameter",
                "headers": {
                    "i": {"type": "string", "alias": "canning plants"},
                    "new-york": {
                        "type": "numeric",
                        "alias": "new-york",
                    },
                    "chicago": {"type": "numeric", "alias": "chicago"},
                    "topeka": {"type": "numeric", "alias": "topeka"},
                },
            },
            "table_without_records": {
                "alias": "table_without_records",
                "symtype": "parameter",
                "headers": {
                    "i": {"type": "string", "alias": "canning plants"},
                    "new-york": {
                        "type": "numeric",
                        "alias": "new-york",
                    },
                    "chicago": {"type": "numeric", "alias": "chicago"},
                    "topeka": {"type": "numeric", "alias": "topeka"},
                },
            },
            "_scalars": {
                "alias": "Input Scalars",
                "symnames": ["model_type", "f"],
                "symtext": [
                    "model_type",
                    "freight in dollars per case per thousand miles",
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
                    "i": {"type": "string", "alias": "canning plants"},
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
                    "total transportation costs in thousands of dollars"
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
    }

    _ = Parameter(
        m,
        name="table_with_domain_forwarding",
        domain=[i, j],
        records=distances,
        is_miro_input=True,
        is_miro_table=True,
        domain_forwarding=True,
    )
    with pytest.raises(ValidationError):
        encoder.write_json()

    m2 = Container()
    i2 = Set(
        m2,
        name="i2",
        records=["seattle", "san-diego"],
        description="canning plants",
    )
    j2 = Set(
        m2,
        name="j2",
        records=["new-york", "chicago", "topeka"],
        description="markets",
        is_miro_input=True,
    )
    _ = Parameter(
        m2,
        name="last_item_miro_input",
        domain=[i2, j2],
        records=distances,
        is_miro_input=True,
        is_miro_table=True,
    )
    encoder = MiroJSONEncoder(m2)
    with pytest.raises(ValidationError):
        encoder.write_json()

    m3 = Container()
    i3 = Set(m3, "i3", records=["i3"])
    _ = Parameter(
        m3,
        name="dimension_small",
        domain=i3,
        records=[("i3", 2)],
        is_miro_input=True,
        is_miro_table=True,
    )
    encoder = MiroJSONEncoder(m3)
    with pytest.raises(ValidationError):
        encoder.write_json()

    m4 = Container()
    i4 = Set(m4, "i4", records=["i4"])
    _ = Parameter(
        m4,
        name="last_domain_str",
        domain=[i4, "bla"],
        records=[("i4", "bla", 2)],
        is_miro_input=True,
        is_miro_table=True,
    )
    encoder = MiroJSONEncoder(m4)
    with pytest.raises(ValidationError):
        encoder.write_json()

    m5 = Container()
    i5 = Set(m5, "i5", records=["i1", "i2"])
    _ = Set(m5, "i6", domain=i5, records=["i1"], is_miro_input=True)
    encoder = MiroJSONEncoder(m5)
    generated_json = encoder.write_json()
    assert generated_json == {
        "modelTitle": "GAMSPy App",
        "inputSymbols": {
            "i6": {
                "alias": "i6",
                "symtype": "set",
                "headers": {
                    "i5": {"type": "string", "alias": "i5"},
                    "element_text": {
                        "type": "string",
                        "alias": "element_text",
                    },
                },
            }
        },
        "outputSymbols": {},
    }


def test_non_init():
    directory = str(pathlib.Path(__file__).parent.resolve())
    miro_gdx_in = os.path.join(directory, "miro_models", "_miro5_gdxin_.gdx")
    miro_gdx_out = os.path.join(directory, "miro_models", "_miro5_gdxout_.gdx")

    subprocess_env = os.environ.copy()
    subprocess_env["MIRO"] = "0"
    subprocess_env["GAMS_IDC_GDX_INPUT"] = miro_gdx_in
    subprocess_env["GAMS_IDC_GDX_OUTPUT"] = miro_gdx_out

    try:
        subprocess.run(
            [
                sys.executable,
                directory + os.sep + "miro_models" + os.sep + "miro5.py",
            ],
            env=subprocess_env,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(e)


def test_miro_in():
    directory = str(pathlib.Path(__file__).parent.resolve())
    miro_gdx_in = os.path.join(directory, "miro_models", "_miro5_gdxin_.gdx")

    subprocess_env = os.environ.copy()
    subprocess_env["GAMS_IDC_GDX_INPUT"] = miro_gdx_in

    # m.in_miro = True
    try:
        subprocess.run(
            [
                sys.executable,
                directory + os.sep + "miro_models" + os.sep + "miro6.py",
            ],
            env=subprocess_env,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(e)

    subprocess_env = os.environ.copy()

    # m.in_miro = False
    try:
        subprocess.run(
            [
                sys.executable,
                directory + os.sep + "miro_models" + os.sep + "miro6.py",
            ],
            env=subprocess_env,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(e)


def test_miro_protect(data):
    m = data
    m = Container()

    i = Set(m, name="i", is_miro_input=True)
    with pytest.raises(ValidationError):
        i.setRecords(["i1", "i2"])

    f = Parameter(
        m,
        name="f",
        description="supply of commodity at plant i (in cases)",
        records=5 if not m.in_miro else None,
        is_miro_input=True,
    )

    with pytest.raises(ValidationError):
        f.setRecords(6)

    with pytest.raises(ValidationError):
        f[...] = 5

    m = Container(options=Options(miro_protect=False))
    f = Parameter(
        m,
        name="f",
        description="supply of commodity at plant i (in cases)",
        records=5 if not m.in_miro else None,
        is_miro_input=True,
    )
    f.setRecords(6)
