from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from collections.abc import Sequence

from gamspy import Container
from gamspy._model import ATTRIBUTE_MAP
from gamspy.exceptions import ValidationError


def serialize(container: Container, path: str) -> None:
    """
    Serializes the given Container into a zip file.

    Parameters
    ----------
    container : Container
        Container to be serialized.
    path : str
        Path to the zip file.

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=range(3))
    >>> gp.serialize(m, "serialization_path.zip")

    """
    if not path.endswith(".zip"):
        raise ValidationError(f"The path must end with .zip but found {path}")

    if not isinstance(container, Container):
        raise ValidationError(
            f"`container` must be of type Container but found {type(container)}"
        )

    with tempfile.TemporaryDirectory() as tmpdir_name:
        g00_path = os.path.join(tmpdir_name, "gams_state.g00")
        json_path = os.path.join(tmpdir_name, "dict.json")

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

        shutil.make_archive(path[:-4], "zip", tmpdir_name)


def deserialize(path: str) -> Container:
    """
    Deserializes the given zip file into a Container.

    Parameters
    ----------
    path : str
        Path to the zip file.

    Returns
    -------
    Container
        Deserialized Container.

    Raises
    ------
    ValidationError
        In case the given path is not a zip file.
    """
    if not zipfile.is_zipfile(path):
        raise ValidationError(f"`{path}` is not a zip file.")

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

        deserialized_matches: dict[str, str | Sequence[str]] = model.get(
            "_matches", None
        )
        matches = dict()
        if deserialized_matches is not None:
            for key, value in deserialized_matches.items():
                if isinstance(value, str):
                    matches[container[key]] = container[value]
                else:
                    matches[container[key]] = [
                        container[var_name] for var_name in value
                    ]

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
