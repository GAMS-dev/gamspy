import pytest
from gams.core import gdx as gams_gdx

import gamspy as gp
import gamspy._gdx as gdx
from gamspy.exceptions import GdxException


@pytest.mark.unit
def test_opening_non_existent_file():
    m = gp.Container()

    with pytest.raises(
        GdxException, match=r"Could not open GDX file `.*?` for reading:"
    ):
        with gdx.open_gdx(m.system_directory, "nonexistent.gdx"):
            ...


@pytest.fixture
def gdx_file(tmp_path):
    m = gp.Container()
    _ = gp.Set(m, "i", records=["a", "b"])
    path = str(tmp_path / "test.gdx")
    m.write(path)
    return m.system_directory, path


@pytest.fixture
def close_spy(monkeypatch):
    """Records whether the GDX handle was released."""
    calls = {"close": 0, "free": 0}

    for name in ("close", "free"):
        original = getattr(gams_gdx, f"gdx{name.capitalize()}")

        def spy(handle, _original=original, _name=name):
            calls[_name] += 1
            return _original(handle)

        monkeypatch.setattr(gams_gdx, f"gdx{name.capitalize()}", spy)

    return calls


@pytest.mark.unit
def test_handle_released_on_base_exception(gdx_file, close_spy):
    """KeyboardInterrupt must not leak the GDX handle."""
    system_directory, path = gdx_file

    with pytest.raises(KeyboardInterrupt):
        with gdx.open_gdx(system_directory, path):
            raise KeyboardInterrupt

    assert close_spy["close"] == 1
    assert close_spy["free"] == 1


@pytest.mark.unit
def test_handle_released_on_exception(gdx_file, close_spy):
    system_directory, path = gdx_file

    with pytest.raises(ValueError):
        with gdx.open_gdx(system_directory, path):
            raise ValueError("error in the body")

    assert close_spy["close"] == 1
    assert close_spy["free"] == 1


@pytest.mark.unit
def test_body_exception_is_not_relabeled(gdx_file):
    system_directory, path = gdx_file

    with pytest.raises(ValueError, match=r"^error in the body$"):
        with gdx.open_gdx(system_directory, path):
            raise ValueError("error in the body")


@pytest.mark.unit
def test_invalid_mode(gdx_file, close_spy):
    system_directory, path = gdx_file

    with pytest.raises(GdxException, match="Invalid mode `a`"):
        with gdx.open_gdx(system_directory, path, mode="a"):
            ...

    assert close_spy["free"] == 1
