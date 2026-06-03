from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

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
        arrkeys = np.empty(dim * nrecs, dtype=dtype_keys)
        for i in range(dim):
            col_data = records.iloc[:, i]
            col_data = col_data.cat.codes

            idx_start, idx_end = i * nrecs, (i + 1) * nrecs
            arrkeys[idx_start:idx_end] = col_data

        arrkeys = arrkeys.reshape((nrecs, dim), order="F")

    if dim == 0:
        arrvals = records.to_numpy()
    elif isinstance(symobj, (Set, Parameter)):
        arrvals = records.iloc[:, -1].to_numpy().reshape((-1, 1))
    else:
        num_attr = len(symobj._attributes)
        arrvals = np.empty(num_attr * nrecs, dtype=np.float64)
        for i in range(num_attr):
            idx_start, idx_end = i * nrecs, (i + 1) * nrecs
            arrvals[idx_start:idx_end] = records.iloc[:, i + dim].to_numpy()

        arrvals = arrvals.reshape((nrecs, num_attr), order="F")

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
            dtype = pd.CategoricalDtype(categories=unique_uels[i], ordered=True)
            data[col_idx] = pd.Categorical.from_codes(codes=arrkeys[:, i], dtype=dtype)
            col_idx += 1

    # Insert value columns
    if has_values:
        for j in range(arrvals.shape[1]):
            data[col_idx] = arrvals[:, j]
            col_idx += 1

    return pd.DataFrame(data, copy=False)


def generate_unique_labels(labels: list | str) -> list[str]:
    """Generate unique labels from a list of labels."""
    if not isinstance(labels, list):
        labels = [labels]

    labels = [label if label != "*" else "uni" for label in labels]

    # Append suffixes if the list is not entirely unique
    if len(labels) != len(set(labels)):
        labels = [f"{label}_{n}" for n, label in enumerate(labels)]

    return labels


def cartesian_product(*arrays: np.ndarray) -> np.ndarray:
    """Calculate the Cartesian product of multiple input arrays."""
    if not arrays:
        return np.empty((0, 0))

    la = len(arrays)
    dtype = np.result_type(*arrays)

    # Pre-allocate array: (num_arrays, len(arr1), len(arr2), ...)
    arr = np.empty((la, *map(len, arrays)), dtype=dtype)

    for i, a in enumerate(arrays):
        # Explicitly build a shape to align 'a' along the i-th dimension.
        # e.g., for 3 arrays: i=0 -> (-1, 1, 1), i=1 -> (1, -1, 1), i=2 -> (1, 1, -1)
        broadcast_shape = [1] * la
        broadcast_shape[i] = -1

        # Reshape 'a' so broadcasting explicitly matches the target array
        arr[i, ...] = np.asarray(a).reshape(broadcast_shape)

    return arr.reshape(la, -1).T


def choice_no_replace(
    choose_from: int,
    n_choose: int,
    seed: int | None = None,
) -> np.ndarray:
    """Randomly select unique items from a pool without replacement."""
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

    # numpy is faster as density grows
    if density > 0.08:
        rng = np.random.default_rng(seed)
        idx = rng.choice(choose_from, replace=False, size=n_choose)
    # random.shuffle is much faster at low density
    else:
        random.seed(seed)
        idx = np.array(random.sample(range(choose_from), n_choose), dtype=int)

    return np.sort(idx)
