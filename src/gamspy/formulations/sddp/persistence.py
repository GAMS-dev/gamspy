"""
Save / load helpers for the sddp module.

Public entry points are ``SDDP.save()`` and ``SDDP.load()`` in
``core.py``; the helpers in this module are private to the sddp package.
"""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from typing import TYPE_CHECKING, Any

import numpy as np

import gamspy as gp
from gamspy._guss import GUSSScenarioDict
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from gamspy.formulations.sddp.core import SDDP

# Layout inside the .sddp archive (a zip under the hood).
_CONTAINER_ARCHIVE = "container.gpz"
_METADATA_FILE = "sddp_metadata.json"
_FILE_FORMAT_VERSION = "0.5.0"


# File-format pack / unpack


def _pack(
    container: gp.Container, metadata: dict[str, Any], path: str | os.PathLike
) -> None:
    """Serialize `container` + `metadata` into the .sddp file at `path`."""
    path_str = os.fspath(path)

    with tempfile.TemporaryDirectory() as tmpdir:
        nested = os.path.join(tmpdir, "_container_inner.zip")
        gp.serialize(container, nested)

        sidecar = os.path.join(tmpdir, _METADATA_FILE)
        with open(sidecar, "w") as f:
            json.dump(metadata, f, indent=2)

        with zipfile.ZipFile(path_str, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(nested, arcname=_CONTAINER_ARCHIVE)
            zf.write(sidecar, arcname=_METADATA_FILE)


def _unpack(path: str | os.PathLike) -> tuple[gp.Container, dict[str, Any]]:
    """Open a .sddp file and return the (container, metadata) pair."""
    path_str = os.fspath(path)

    if not os.path.exists(path_str):
        raise ValidationError(f"No file at `{path_str}`.")
    if not zipfile.is_zipfile(path_str):
        raise ValidationError(f"`{path_str}` is not a zip archive.")

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(path_str, "r") as zf:
            names = set(zf.namelist())
            for required in (_CONTAINER_ARCHIVE, _METADATA_FILE):
                if required not in names:
                    raise ValidationError(
                        f"`{path_str}` is missing `{required}` - "
                        f"not a valid sddp save file."
                    )
            zf.extractall(tmpdir)

        with open(os.path.join(tmpdir, _METADATA_FILE)) as f:
            try:
                metadata = json.load(f)
            except json.JSONDecodeError as e:
                raise ValidationError(
                    f"Sidecar `{_METADATA_FILE}` is not valid JSON: {e}"
                ) from e

        # gp.deserialize takes a path to its own zip archive.
        container = gp.deserialize(os.path.join(tmpdir, _CONTAINER_ARCHIVE))

    return container, metadata


# Sidecar build / validate


def _collect_sddp_metadata(sddp: SDDP) -> dict[str, Any]:
    """Build the sidecar dict from a built sddp instance."""
    if not sddp._built:
        raise ValidationError("Cannot save() a sddp instance that was not built. ")
    if sddp._noise is None:
        raise ValidationError("Cannot save() a sddp instance with no registered noise.")

    state_vars = [
        {
            "name": state_var.variable.name,
            "lower_bound": float(state_var.lower_bound),
            "upper_bound": float(state_var.upper_bound),
            "initial_state": (
                float(state_var.initial_state)
                if state_var.initial_state is not None
                else None
            ),
        }
        for state_var in sddp._states
    ]

    active_iteration_records = (
        sddp._active_cut_iteration_set.records
        if sddp._active_cut_iteration_set is not None
        else None
    )
    iterations_completed = (
        0 if active_iteration_records is None else len(active_iteration_records)
    )

    return {
        "sddp_version": _FILE_FORMAT_VERSION,
        "constructor": {
            "stage_set_name": sddp._stage_set.name,
            "time_set_name": sddp._time_set.name,
            "n_trials": int(sddp._n_trials),
            "seed": int(sddp._seed),
            "verbose": bool(sddp._verbose),
        },
        "build_args": {
            "stage_cost_var_name": sddp._stage_cost_var.name,
        },
        "state_vars": state_vars,
        "noise": {
            "parameter_name": sddp._noise.parameter.name,
            "has_probabilities": sddp._noise.probabilities is not None,
        },
        "sim_call_count": int(sddp._sim_call_count),
        "training_summary": {
            "iterations_completed": iterations_completed,
        },
    }


def _validate_metadata(metadata: dict[str, Any]) -> None:
    """Schema + major-version checks on a loaded sidecar."""
    if not isinstance(metadata, dict):
        raise ValidationError(f"Sidecar is not a dict (got {type(metadata).__name__}).")

    required = (
        "sddp_version",
        "constructor",
        "build_args",
        "state_vars",
        "noise",
        "sim_call_count",
    )
    missing = [k for k in required if k not in metadata]
    if missing:
        raise ValidationError(f"Sidecar is missing required field(s): {missing}")

    saved_version = str(metadata["sddp_version"])
    current_version = _FILE_FORMAT_VERSION
    saved_major = saved_version.split(".", maxsplit=1)[0]
    current_major = current_version.split(".")[0]
    if saved_major != current_major:
        raise ValidationError(
            f"sddp version mismatch: file was saved with `{saved_version}` "
            f"but the current sddp module is `{current_version}`. Major "
            f"versions must match. Retrain from scratch to migrate."
        )

    def _minor_tuple(version: str) -> tuple[int, ...]:
        parts: list[int] = []
        for piece in version.split("."):
            try:
                parts.append(int(piece))
            except ValueError:
                break
        return tuple(parts[:2])

    if _minor_tuple(saved_version) < (0, 5):
        raise ValidationError(
            f"sddp save file version `{saved_version}` predates the internal "
            f"symbol naming refactor (0.5.0). Files using the older symbol "
            f"layout cannot be loaded by sddp `{current_version}`; retrain "
            f"from scratch to migrate."
        )


# Reattach - turn (container, metadata) into a usable SDDP instance


def _reattach_sddp(container: gp.Container, metadata: dict[str, Any]) -> SDDP:
    """Construct an SDDP instance from a deserialized container + sidecar."""
    from gamspy.formulations.sddp.core import SDDP
    from gamspy.formulations.sddp.noise import NoiseConfig
    from gamspy.formulations.sddp.state import StateVar

    constructor = metadata["constructor"]
    build_args = metadata["build_args"]
    state_specs = metadata["state_vars"]
    noise_spec = metadata["noise"]

    def _lookup(name: str) -> Any:
        if name not in container.data:
            raise ValidationError(
                f"Loaded container is missing symbol `{name}` - the saved "
                f"sddp instance may be corrupted or was saved by an "
                f"incompatible version."
            )
        return container.data[name]

    # User-supplied symbols
    stage_set = _lookup(constructor["stage_set_name"])
    time_set = _lookup(constructor["time_set_name"])
    noise_param = _lookup(noise_spec["parameter_name"])
    stage_cost_var = _lookup(build_args["stage_cost_var_name"])

    # Allocate an SDDP without going through __init__ (we are not
    # creating any GAMSPy symbols - they're all already in `container`).
    sddp = SDDP.__new__(SDDP)
    sddp._container = container
    sddp._stage_set = stage_set
    sddp._time_set = time_set
    sddp._n_trials = int(constructor["n_trials"])
    sddp._seed = int(constructor["seed"])
    sddp._verbose = bool(constructor["verbose"])
    sddp._sim_call_count = int(metadata["sim_call_count"])
    sddp._built = True
    sddp._loaded_from_save = True
    sddp._trained = True

    # sddp-internal symbols (all under the sddp_ prefix)
    sddp._active_stage_set = _lookup("sddp_active_stage")
    sddp._iteration_set = _lookup("sddp_iteration")
    sddp._active_cut_iteration_set = _lookup("sddp_active_cut_iteration")
    sddp._alpha = _lookup("sddp_alpha")
    sddp._approx_cost = _lookup("sddp_approx_cost")
    sddp._obj_approx_eq = _lookup("sddp_obj_approx")
    sddp._cuts_eq = _lookup("sddp_cuts")
    sddp._time_alias = _lookup("sddp_time_alias")
    sddp._stage_end_map = _lookup("sddp_stage_end")
    sddp._previous_stage_end_map = _lookup("sddp_previous_stage_end")
    sddp._guss_options = _lookup("sddp_guss_options")
    sddp._scenario_prob = _lookup("sddp_prob")
    sddp._stage_scenario_noise = _lookup("sddp_stage_scenario_noise")

    if "sddp_model" not in container.models:
        raise ValidationError(
            "Loaded container is missing model `sddp_model` - the saved "
            "sddp instance may be corrupted."
        )
    sddp._gp_model = container.models["sddp_model"]

    # sddp-owned composite trial set + shared Benders cut intercept.
    sddp._trial_set = _lookup("sddp_trial")
    sddp._cut_intercept = _lookup("sddp_cut_intercept")

    # State variables
    sddp._states = []
    for state_spec in state_specs:
        variable = _lookup(state_spec["name"])
        state_var = StateVar(  # type: ignore[call-arg]
            variable=variable,
            lower_bound=float(state_spec["lower_bound"]),
            upper_bound=float(state_spec["upper_bound"]),
            initial_state=(
                float(state_spec["initial_state"])
                if state_spec["initial_state"] is not None
                else None
            ),
        )
        state_var.trial_values = _lookup(f"sddp_trial_values_{state_var.name}")
        state_var.cut_slope = _lookup(f"sddp_cut_slope_{state_var.name}")
        sddp._states.append(state_var)

    # Rebuild scenario_data and probabilities arrays
    scenario_set = _lookup("sddp_scenario")
    scenario_data = _read_scenario_data(
        sddp._stage_scenario_noise, stage_set, scenario_set
    )
    probabilities = (
        _read_probabilities(sddp._scenario_prob)
        if noise_spec["has_probabilities"]
        else None
    )
    noise_config = NoiseConfig(  # type: ignore[call-arg]
        parameter=noise_param,
        scenario_data=scenario_data,
        probabilities=probabilities,
    )
    noise_config.scenario_set = scenario_set
    sddp._noise = noise_config

    # GUSS scenario dicts - wrappers reconstructed via from_existing
    sddp._backward_scenario_dict = GUSSScenarioDict.from_existing(
        container, "sddp_backward_scenario_dict"
    )
    sddp._forward_scenario_dict = GUSSScenarioDict.from_existing(
        container, "sddp_forward_scenario_dict"
    )
    sddp._initial_stage_scenario_dict = GUSSScenarioDict.from_existing(
        container, "sddp_initial_stage_scenario_dict"
    )

    sddp._stage_cost_var = stage_cost_var
    sddp._user_equations = []

    # Python-side stage geometry - recompute from the recovered sets.
    stage_labels = stage_set.toList()
    time_labels = time_set.toList()
    n_stages = len(stage_labels)
    n_times = len(time_labels)
    time_steps_per_stage = n_times // n_stages
    stage_end_time: dict[str, str] = {}
    previous_stage_end_time: dict[str, str] = {}
    for stage_index, stage_label in enumerate(stage_labels):
        stage_end_time[stage_label] = time_labels[
            (stage_index + 1) * time_steps_per_stage - 1
        ]
        previous_stage_end_time[stage_label] = time_labels[
            ((stage_index - 1) % n_stages + 1) * time_steps_per_stage - 1
        ]

    sddp._stage_labels = stage_labels
    sddp._stage_end_time = stage_end_time
    sddp._previous_stage_end_time = previous_stage_end_time
    sddp._initial_state_time = previous_stage_end_time[stage_labels[0]]

    sddp._scenario_labels = scenario_set.toList()
    sddp._trial_labels = sddp._trial_set.toList()

    # Solve options - match the construction in build().
    sddp._solve_opts = gp.Options(
        equation_listing_limit=0,
        variable_listing_limit=0,
        report_solution=2,
        solve_link_type="memory",
        merge_strategy="clear",
    )

    # Per-state snapshot parameters for the original user bounds.
    for state_var in sddp._states:
        state_var.orig_lo_param = _lookup(f"sddp_orig_lo_{state_var.name}")
        state_var.orig_up_param = _lookup(f"sddp_orig_up_{state_var.name}")
    # Reconstruct the user-variable bound
    sddp._user_variables = []
    sddp._user_bound_snapshots = []
    for name in container.data:
        if not name.startswith("sddp_blo_"):
            continue
        var_name = name[len("sddp_blo_") :]
        up_name = f"sddp_bup_{var_name}"
        if var_name in container.data and up_name in container.data:
            v = container.data[var_name]
            sddp._user_bound_snapshots.append(
                (v, list(v.domain), container.data[name], container.data[up_name])
            )

    return sddp


# Helpers: recover ndarrays from GAMSPy parameters


def _read_scenario_data(
    stage_scenario_noise: gp.Parameter,
    stage_set: gp.Set,
    scenario_set: gp.Set,
) -> np.ndarray:
    """Recover stage/scenario noise values from sddp_stage_scenario_noise."""
    n_stages = len(stage_set.records) if stage_set.records is not None else 0
    n_scenarios = len(scenario_set.records) if scenario_set.records is not None else 0
    result = np.zeros((n_stages, n_scenarios), dtype=float)
    rec = stage_scenario_noise.records
    if rec is None or len(rec) == 0:
        return result

    stage_index_by_label = {
        stage_label: index for index, stage_label in enumerate(stage_set.toList())
    }
    scenario_index_by_label = {
        scenario_label: index
        for index, scenario_label in enumerate(scenario_set.toList())
    }
    for row in rec.itertuples(index=False, name=None):
        stage_label = str(row[0])
        scenario_label = str(row[1])
        value = float(row[2])
        if (
            stage_label in stage_index_by_label
            and scenario_label in scenario_index_by_label
        ):
            result[
                stage_index_by_label[stage_label],
                scenario_index_by_label[scenario_label],
            ] = value
    return result


def _read_probabilities(scenario_prob: gp.Parameter) -> np.ndarray:
    """Recover the 1-D probabilities ndarray from sddp_prob."""
    rec = scenario_prob.records
    if rec is None or len(rec) == 0:
        return np.array([], dtype=float)
    return rec["value"].to_numpy(dtype=float)
