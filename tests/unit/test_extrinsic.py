from __future__ import annotations

import os
import platform

import pytest

from gamspy import Container, Equation, Parameter, Set, Sum, Variable
from gamspy.exceptions import ValidationError

pytestmark = pytest.mark.unit


def get_default_platform():
    operating_system = platform.system().lower()
    architecture = platform.machine()

    if operating_system == "darwin":
        return f"mac_{architecture}"
    elif operating_system == "linux" and architecture == "aarch64":
        return f"linux_{architecture}"

    return operating_system


@pytest.fixture
def data():
    m = Container()
    yield m
    m.close()


def test_extrinsic_functions(data):
    m = data
    user_platform = get_default_platform()

    if user_platform == "linux_aarch64":
        return

    # Invalid path
    with pytest.raises(FileNotFoundError):
        _ = m.importExtrinsicLibrary(
            "blablablalib.so",
            functions={
                "myPi": "Pi",
                "myCos": "Cosine",
            },
        )

    names = {
        "linux": "libtricclib64.so",
        "mac_x86_64": "libtricclib64x86.dylib",
        "mac_arm64": "libtricclib64arm.dylib",
        "windows": "tricclib64.dll",
    }
    directory = os.path.dirname(os.path.abspath(__file__))
    shared_object = os.path.join(directory, names[user_platform])

    # This is a library which contains myNum=1.
    trilib = m.importExtrinsicLibrary(
        shared_object,
        functions={
            "myPi": "Pi",
            "myCos": "Cosine",
        },
    )

    # if attribute is not in functions, call the default getattr function of Python
    with pytest.raises(AttributeError):
        _ = trilib.bla

    # Test extrinsic function with no argument
    d = Parameter(m, "d")
    d[...] = trilib.myPi
    assert d.getAssignment() == "d = myPi;"
    assert d.toValue() == 3.141592653589793238462643

    # Test extrinsic function with one argument
    d2 = Parameter(m, "d2")
    d2[...] = trilib.myCos(90)
    assert d2.getAssignment() == "d2 = myCos(90);"
    assert int(d2.toValue()) == 0
    assert trilib.myCos(0).records.values.item() == 1
    assert trilib.myCos(0).toValue() == 1

    # External functions do not accept keyword arguments
    with pytest.raises(ValidationError):
        d2[...] = trilib.myCos(degree=90)

    # Test the interaction with other components
    d3 = Parameter(m, "d3")
    d3[...] = trilib.myCos(90, 1) * 3
    assert d3.getAssignment() == "d3 = myCos(90,1) * 3;"
    assert int(d3.toValue()) == 0

    v = Variable(m, "v")

    # Make sure that Number(value) keeps the order
    assert (v == trilib.myCos(0)).gamsRepr() == "v =e= myCos(0)"
    assert (trilib.myCos(0) == v).gamsRepr() == "myCos(0) =e= v"

    i = Set(m, name="i", records=[f"i{j}" for j in range(10)])
    x = Variable(m, name="x", domain=i)
    e = Equation(m, name="e", definition=Sum(i, x[i] * x[i]) == trilib.myCos(90))
    assert e.getDefinition() == "e .. sum(i,x(i) * x(i)) =e= myCos(90);"
    f = Equation(m, name="f", definition=trilib.myCos(90) == Sum(i, x[i] * x[i]))
    assert f.getDefinition() == "f .. myCos(90) =e= sum(i,x(i) * x(i));"

    m.close()
