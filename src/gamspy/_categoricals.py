from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

CategoriesT = TypeVar("CategoriesT", np.ndarray, pd.Index)

# Code of a record whose domain entry is missing; it points at no category at all.
MISSING = -1


def _used_mask(codes: np.ndarray, n_categories: int) -> np.ndarray:
    used = np.zeros(n_categories, dtype=bool)
    used[codes[codes >= 0]] = True
    return used


def _min_int_dtype(n: int) -> type[np.signedinteger]:
    if n <= np.iinfo(np.int8).max:
        return np.int8
    elif n <= np.iinfo(np.int16).max:
        return np.int16
    elif n <= np.iinfo(np.int32).max:
        return np.int32
    return np.int64


def apply_remap(codes: np.ndarray, remap: np.ndarray) -> np.ndarray:
    new_codes = remap[codes]
    if codes.size > 0 and codes.min() < 0:
        new_codes[codes < 0] = MISSING

    return new_codes


def codes_and_categories(cat: pd.Categorical) -> tuple[np.ndarray, np.ndarray]:
    return cat.codes, np.asarray(cat.categories.array)


def used_categories(codes: np.ndarray, categories: CategoriesT) -> CategoriesT:
    """The categories that `codes` actually references, in category order."""
    used = _used_mask(codes, len(categories))
    if used.all():
        return categories

    return categories[used]


def union_categories(
    cats: Sequence[tuple[np.ndarray, np.ndarray]],
    ignore_unused: bool = False,
) -> np.ndarray:
    cats_arr = []
    for codes_i, categories_i in cats:
        if ignore_unused:
            categories_i = np.asarray(used_categories(codes_i, categories_i))

        cats_arr.append(categories_i)

    if len(cats_arr) == 1:
        return cats_arr[0]

    combined_cats = np.concatenate(cats_arr)
    return np.asarray(pd.unique(combined_cats))


def remove_unused_categories(
    codes: np.ndarray, categories: np.ndarray | pd.Index
) -> tuple[np.ndarray, np.ndarray | pd.Index]:
    used_idx = np.flatnonzero(_used_mask(codes, len(categories)))
    if used_idx.size == len(categories):
        return codes, categories

    out_dtype = _min_int_dtype(used_idx.size)
    remap = np.full(len(categories), MISSING, dtype=out_dtype)
    remap[used_idx] = np.arange(used_idx.size, dtype=out_dtype)

    return apply_remap(codes, remap), categories[used_idx]


def remove_categories(
    codes: np.ndarray, categories: np.ndarray, removal_set: set
) -> tuple[np.ndarray, np.ndarray]:
    if not removal_set:
        return codes, categories

    keep_mask = np.fromiter(
        (category not in removal_set for category in categories),
        dtype=bool,
        count=len(categories),
    )

    if keep_mask.all():
        return codes, categories

    kept_categories = categories[keep_mask]

    remap = np.full(len(categories), MISSING, dtype=np.intp)
    remap[keep_mask] = np.arange(len(kept_categories), dtype=np.intp)

    return apply_remap(codes, remap), kept_categories


def set_categories(
    codes: np.ndarray,
    categories: np.ndarray,
    new_categories: np.ndarray,
    validate: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-encode codes against new_categories"""
    new_categories = np.asarray(new_categories, dtype=object)

    if validate:
        if pd.isna(new_categories).any():
            raise ValueError("Categorical categories cannot be null")

        if len(set(new_categories)) != len(new_categories):
            raise ValueError("Categorical categories must be unique")

    new_index = {cat: i for i, cat in enumerate(new_categories)}
    remap = np.fromiter(
        (new_index.get(cat, MISSING) for cat in categories),
        dtype=np.intp,
        count=len(categories),
    )

    return apply_remap(codes, remap), new_categories


def rstrip_categories(categories: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
    if categories.dtype == object:
        try:
            if all(x == x.rstrip() for x in categories):
                return None, categories
        except AttributeError:
            pass

    stripped = np.array([str(x).rstrip() for x in categories], dtype=object)
    remap, new_categories = pd.factorize(stripped, sort=False)
    return remap, np.asarray(new_categories)


def assemble_categorical(
    codes: np.ndarray,
    categories: Sequence | np.ndarray | pd.Index,
    ordered: bool = True,
    fastpath: bool = False,
) -> pd.Categorical:
    """
    Build a `pd.Categorical` directly from already-computed `codes` and
    `categories`, bypassing `pd.Categorical`'s normal constructor validation.

    `fastpath=True` skips even the (lighter-weight) bookkeeping
    `from_codes(..., validate=False)` still does, by going straight through
    `CategoricalDtype._from_fastpath()`/`Categorical._simple_new()` --
    only safe when `categories` is already known unique/non-null and `codes`
    is already known in-range, true for this module's own callers.
    """
    if not isinstance(categories, pd.Index):
        categories = np.asarray(categories, dtype=object)

    if fastpath:
        return pd.Categorical._simple_new(
            codes, pd.CategoricalDtype._from_fastpath(categories, ordered=ordered)
        )
    return pd.Categorical.from_codes(
        codes, categories=categories, ordered=ordered, validate=False
    )
