from __future__ import annotations

import os
import platform

import pytest
from gamspy import Container, Parameter
from gamspy.exceptions import ValidationError

pytestmark = pytest.mark.unit


def get_default_platform():
    operating_system = platform.system().lower()
    architecture = platform.machine()

    if operating_system == "darwin":
        return f"mac_{architecture}"

    return operating_system


@pytest.fixture
def data():
    m = Container()
    yield m
    m.close()


def test_extrinsic_functions(data):
    m = data
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
    shared_object = os.path.join(directory, names[get_default_platform()])

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
    assert d.toValue() == 3.141592653589793238462643

    # Test extrinsic function with one argument
    d2 = Parameter(m, "d2")
    d2[...] = trilib.myCos(90)
    assert int(d2.toValue()) == 0

    # External functions do not accept keyword arguments
    with pytest.raises(ValidationError):
        d2[...] = trilib.myCos(degree=90)

    # Test the interaction with other components
    d3 = Parameter(m, "d3")
    d3[...] = trilib.myCos(90, 1) * 3
    assert int(d3.toValue()) == 0
