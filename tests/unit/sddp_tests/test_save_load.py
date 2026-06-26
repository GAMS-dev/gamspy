from __future__ import annotations

import pytest

from gamspy.exceptions import ValidationError
from gamspy.formulations import SDDP


@pytest.mark.unit
def test_save_requires_sddp_extension(clearlake_built, tmp_path):
    with pytest.raises(ValidationError):
        clearlake_built.sddp.save(str(tmp_path / "model.zip"))


@pytest.mark.unit
def test_load_bad_extension():
    with pytest.raises(ValidationError):
        SDDP.load("model.zip")


@pytest.mark.unit
def test_load_missing_file(tmp_path):
    with pytest.raises(ValidationError):
        SDDP.load(str(tmp_path / "does_not_exist.sddp"))


@pytest.mark.requires_license
def test_save_load_roundtrip(clearlake_trained, tmp_path):
    c = clearlake_trained
    before = c.sddp.policy(stage="mar", state=180, noise=100, report=[c.rel, c.lev])

    path = str(tmp_path / "clearlake.sddp")
    c.sddp.save(path)

    loaded = SDDP.load(path)

    after = loaded.policy(
        stage="mar",
        state=180,
        noise=100,
        report=[loaded.container["R"], loaded.container["L"]],
    )

    assert after.approx_cost_to_go == before.approx_cost_to_go
    assert after.decisions == before.decisions


@pytest.mark.requires_license
def test_loaded_instance_is_read_only(clearlake_trained, tmp_path):
    c = clearlake_trained
    path = str(tmp_path / "clearlake.sddp")
    c.sddp.save(path)

    loaded = SDDP.load(path)
    with pytest.raises(ValidationError):
        loaded.train(n_iter=1)
