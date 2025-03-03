from __future__ import annotations

import json
import os
import shutil

from gamspy import Container
from gamspy._model import ATTRIBUTE_MAP


def serialize(container: Container, path: str) -> None:
    os.makedirs(path)
    g00_path = os.path.join(path, "gams_state.g00")
    json_path = os.path.join(path, "dict.json")

    # Dump the GAMS State to disc
    container._options._set_debug_options({"save": g00_path})
    container._synch_with_gams()
    container._options._set_debug_options(dict())

    # Serialize symbols
    info = dict()
    for name, symbol in container.data.items():
        info[name] = symbol._serialize()

    # Serialize models
    models = dict()
    for model in container.models.values():
        models[model.name] = model._serialize()
        info["models"] = models

    with open(json_path, "w") as file:
        json.dump(info, file)

    shutil.make_archive(path, "zip", path)


def deserialize(path: str) -> Container:
    src = path
    dst = path[:-4]
    shutil.unpack_archive(src, dst, "zip")
    g00_path = os.path.join(dst, "gams_state.g00")
    json_path = os.path.join(dst, "dict.json")

    with open(json_path) as file:
        info = json.load(file)

    container = Container(load_from=g00_path)

    # Deserialize symbols
    for name, symbol in container.data.items():
        symbol._deserialize(info[name])

    # Deserialize models
    models = info["models"]
    for name, model in models.items():
        equations = container.getEquations()
        equations = [
            equation
            for equation in equations
            if equation.name in model["equations"]
        ]
        matches = model.get("_matches", None)
        objective_variable = model.get("_objective_variable", None)
        if objective_variable is not None:
            objective_variable = container[objective_variable]

        deserialized_model = container.addModel(
            name=name,
            problem=model["problem"],
            sense=model["sense"],
            equations=equations,
            matches=matches,
            objective=objective_variable,
        )
        for attribute in ATTRIBUTE_MAP.values():
            setattr(deserialized_model, attribute, model[attribute])

    return container
