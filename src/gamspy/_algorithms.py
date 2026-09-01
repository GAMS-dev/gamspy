from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

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
            # `unique_uels` come straight from GMD and are unique by
            # construction, so skip pandas' redundant uniqueness/bounds checks
            # (`is_unique` on large UEL lists dominates the read-back otherwise).
            dtype = pd.CategoricalDtype._from_fastpath(
                categories=unique_uels[i], ordered=True
            )
            data[col_idx] = pd.Categorical.from_codes(
                codes=arrkeys[:, i], dtype=dtype, validate=False
            )
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


def used_category_positions(
    codes: np.ndarray, num_categories: int
) -> tuple[np.ndarray, bool]:
    if codes.size == 0:
        return np.empty(0, dtype=np.intp), False

    has_na = bool(codes.min() < 0)
    present = np.zeros(num_categories, dtype=bool)
    if has_na:
        present[codes[codes >= 0]] = True
    else:
        present[codes] = True

    return np.flatnonzero(present), has_na


def drop_unused_categories(column: pd.Series) -> pd.Series:
    # `.array.codes` reads the Categorical backing array directly
    codes = cast("pd.Categorical", column.array).codes
    categories = column.cat.categories

    used, has_na = used_category_positions(codes, categories.size)

    if used.size == categories.size:
        return column

    remap = np.full(categories.size, -1, dtype=codes.dtype)
    remap[used] = np.arange(used.size, dtype=codes.dtype)
    new_codes = remap[codes]
    if has_na:
        new_codes[codes < 0] = -1

    dtype = pd.CategoricalDtype._from_fastpath(categories[used], ordered=True)

    return pd.Series(
        pd.Categorical.from_codes(new_codes, dtype=dtype, validate=False),
        index=column.index,
        name=column.name,
    )


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
