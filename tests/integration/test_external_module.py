from __future__ import annotations

import glob
import math
import os
import pathlib
import platform
import shutil
import sys

import pytest

import gamspy as gp
from gamspy import EngineClient
from gamspy.exceptions import ValidationError

pytestmark = pytest.mark.integration
try:
    from dotenv import load_dotenv

    load_dotenv(os.getcwd() + os.sep + ".env")
except Exception:
    pass


@pytest.fixture
def data():
    # Arrange
    m = gp.Container()
    directory = str(pathlib.Path(__file__).parent.resolve())
    external_module = os.path.relpath(
        os.path.join(
            directory, "external_module", "build", "libsimple_ext_module"
        ),
        os.getcwd(),
    )

    if platform.system() == "Darwin":
        if platform.machine() == "arm64":
            external_module += "_arm64"

        external_module += ".dylib"
    elif platform.system() == "Linux":
        external_module += ".so"
    elif platform.system() == "Windows":
        external_module += ".dll"

    # Act and assert
    yield m, external_module

    # Clean up
    files = glob.glob("_*")
    for file in files:
        if os.path.isfile(file):
            os.remove(file)
    tmp_dirs = glob.glob("tmp*")
    for tmp_dir in tmp_dirs:
        shutil.rmtree(tmp_dir)
    m.close()


def test_sin_cos_example(data):
    if platform.system() == "Linux" and platform.machine() == "aarch64":
        return

    m, external_module = data
    y1 = gp.Variable(m, "y1")
    y2 = gp.Variable(m, "y2")
    x1 = gp.Variable(m, "x1")
    x2 = gp.Variable(m, "x2")

    eq1 = gp.Equation(m, "eq1", type="external")
    eq2 = gp.Equation(m, "eq2", type="external")

    eq1[...] = 1 * x1 + 3 * y1 == 1
    eq2[...] = 2 * x2 + 4 * y2 == 2

    model = gp.Model(
        container=m,
        name="sincos",
        equations=m.getEquations(),
        problem="NLP",
        sense="min",
        objective=y1 + y2,
        external_module=external_module,
    )

    model.solve(output=sys.stdout, solver="conopt")

    assert math.isclose(y1.toDense(), -1)
    assert math.isclose(y2.toDense(), -1)


def test_sin_cos_example2(data):
    if platform.system() == "Linux" and platform.machine() == "aarch64":
        return

    m, _ = data
    y1 = gp.Variable(m, "y1")
    x1 = gp.Variable(m, "x1")

    eq1 = gp.Equation(m, "eq1", type="external")

    with pytest.raises(ValidationError):
        eq1[...] = 1 * x1 + 3 * y1 >= 1


def test_external_equation_on_engine(data):
    if platform.system() == "Linux" and platform.machine() == "aarch64":
        return

    m, external_module = data
    if platform.system() == "Linux":
        m = gp.Container(working_directory=".")
        y1 = gp.Variable(m, "y1")
        y2 = gp.Variable(m, "y2")
        x1 = gp.Variable(m, "x1")
        x2 = gp.Variable(m, "x2")

        eq1 = gp.Equation(m, "eq1", type="external")
        eq2 = gp.Equation(m, "eq2", type="external")

        eq1[...] = 1 * x1 + 3 * y1 == 1
        eq2[...] = 2 * x2 + 4 * y2 == 2

        model = gp.Model(
            container=m,
            name="sincos",
            equations=m.getEquations(),
            problem="NLP",
            sense="min",
            objective=y1 + y2,
            external_module=external_module,
        )

        client = EngineClient(
            host=os.environ["ENGINE_URL"],
            username=os.environ["ENGINE_USER"],
            password=os.environ["ENGINE_PASSWORD"],
            namespace=os.environ["ENGINE_NAMESPACE"],
        )

        model.solve(
            output=sys.stdout,
            solver="conopt",
            backend="engine",
            client=client,
        )
        assert y1.toDense() == -1.0
