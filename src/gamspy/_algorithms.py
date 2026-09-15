from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from gamspy._categoricals import assemble_categorical

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gamspy._symbols import Equation, Parameter, Set, Variable


def get_keys_and_values(
    symobj: Set | Parameter | Variable | Equation,
) -> tuple[np.ndarray, np.ndarray]:
    """Gets the keys and values of a specific symbol."""
    from gamspy._symbols import Parameter, Set

    if symobj.records is None:
        return np.array([]), np.array([])

    nrecs = symobj.number_records
    dim = symobj.dimension
    records = symobj.records

    dtype_keys = int

    if dim == 0:
        arrkeys = np.array([[]], dtype=dtype_keys)
    else:
        arrkeys = np.empty((nrecs, dim), dtype=dtype_keys, order="F")
        for i in range(dim):
            column = cast("pd.Categorical", records.iloc[:, i].array)
            arrkeys[:, i] = column.codes

    if dim == 0:
        arrvals = records.to_numpy(dtype=np.float64)
    elif isinstance(symobj, (Set, Parameter)):
        arrvals = np.asarray(records.iloc[:, -1].array).reshape((-1, 1))
    else:
        num_attr = len(symobj._attributes)
        arrvals = np.empty((nrecs, num_attr), dtype=np.float64, order="F")
        for i in range(num_attr):
            arrvals[:, i] = records.iloc[:, i + dim].array

    return arrkeys, arrvals


def convert_to_categoricals_cat(
    arrkeys: np.ndarray, arrvals: np.ndarray, unique_uels: Sequence
) -> pd.DataFrame | None:
    """Converts arrays into a pandas DataFrame with code-based categoricals."""
    has_domains = arrkeys.size > 0
    has_values = arrvals.size > 0

    if not has_domains and not has_values:
        return None

    data = {}
    col_idx = 0

    # Build categorical columns directly from raw numpy slices
    if has_domains:
        for i in range(arrkeys.shape[1]):
            # `unique_uels` come straight from GMD and are unique by
            # construction, so skip pandas' redundant uniqueness/bounds checks
            # (`is_unique` on large UEL lists dominates the read-back otherwise).
            data[col_idx] = assemble_categorical(
                arrkeys[:, i], unique_uels[i], ordered=True, fastpath=True
            )
            col_idx += 1

    # Insert value columns
    if has_values:
        for j in range(arrvals.shape[1]):
            data[col_idx] = arrvals[:, j]
            col_idx += 1

    return pd.DataFrame(data, copy=False)


def generate_unique_labels(
    labels: list | str, reserved: Sequence[str] | None = None
) -> list[str]:
    """Generate unique labels from a list of labels. `reserved` names (e.g.
    a symbol's attribute columns) are treated as already taken, so a label
    colliding with one of them gets suffixed too."""
    if not isinstance(labels, list):
        labels = [labels]

    labels = [label if label != "*" else "uni" for label in labels]

    # Append suffixes if the list is not entirely unique, or collides with a reserved name.
    reserved_set = set(reserved) if reserved else set()
    if len(labels) != len(set(labels)) or not reserved_set.isdisjoint(labels):
        labels = [f"{label}_{n}" for n, label in enumerate(labels)]

    return labels


def cartesian_product(*arrays: np.ndarray) -> list[np.ndarray]:
    """
    Calculate the Cartesian product of multiple input arrays, returned as one
    column per input array (rather than a single combined 2D array).
    """
    if not arrays:
        return []

    shape = tuple(len(a) for a in arrays)

    columns = []
    for i, a in enumerate(arrays):
        inner = math.prod(shape[i + 1 :])
        outer = math.prod(shape[:i])
        columns.append(np.tile(np.repeat(np.asarray(a), inner), outer))

    return columns


def choice_no_replace(
    choose_from: int,
    n_choose: int,
    seed: int | None = None,
) -> np.ndarray:
    if not isinstance(seed, (int, type(None))):
        raise TypeError("Argument 'seed' must be type int or NoneType")

    choose_from, n_choose = int(choose_from), int(n_choose)
    density = n_choose / choose_from

    if not (0 <= density <= 1):
        raise ValueError(
            "Argument 'density' is out of bounds, must be on the interval [0, 1]."
        )

    if density == 1:
        return np.arange(choose_from, dtype=int)

    _DENSITY_THRESHOLD = 0.3

    rng = np.random.default_rng(seed)

    if density <= _DENSITY_THRESHOLD:
        # directly draw the set of indices to keep, then sort
        idx = rng.choice(choose_from, replace=False, size=n_choose)
        idx.sort()
        return idx

    # draw the excluded complement instead, then invert
    # np.flatnonzero is ascending for free -- no sort needed
    excluded = rng.choice(choose_from, replace=False, size=choose_from - n_choose)
    mask = np.ones(choose_from, dtype=bool)
    mask[excluded] = False
    return np.flatnonzero(mask)
