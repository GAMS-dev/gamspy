import pytest

import gamspy as gp
import gamspy._gdx as gdx
from gamspy.exceptions import GdxException


def test_opening_non_existent_file():
    m = gp.Container()

    with pytest.raises(
        GdxException, match=r"Could not open GDX file `.*?` for reading:"
    ):
        with gdx.open_gdx(m.system_directory, "nonexistent.gdx"):
            ...
