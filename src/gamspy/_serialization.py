from __future__ import annotations

import json
import os
import shutil

from gamspy import Container


def serialize(container: Container, path: str) -> None:
    os.makedirs(path)
    g00_path = os.path.join(path, "gams_state.g00")
    json_path = os.path.join(path, "dict.json")

    # Dump the GAMS State to disc
    container._options._set_debug_options({"save": g00_path})
    container._synch_with_gams()

    # Serialize symbols
    info = dict()
    for name, symbol in container.data.items():
        info[name] = symbol._serialize()

    # Serialize models
    for model in container.models:
        info[model.name] = model._serialize()

    with open(json_path, "w") as file:
        json.dump(info, file)

    shutil.make_archive(path, "zip", path)


def deserialize(path: str) -> Container:
    shutil.unpack_archive(path, path, "zip")
    g00_path = os.path.join(path, "gams_state.g00")
    json_path = os.path.join(path, "dict.json")

    container = Container(load_from=g00_path)

    with open(json_path) as file:
        info = json.load(file)

    # Deserialize symbols
    for name, symbol in container.data.items():
        symbol._deserialize(info[name])

    # Deserialize models
    for model in container.models:
        model._deserialize(info[name])

    return container
