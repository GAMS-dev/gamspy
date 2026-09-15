from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gamspy._categoricals import (
    MISSING,
    _min_int_dtype,
    _used_mask,
    apply_remap,
    assemble_categorical,
    codes_and_categories,
    remove_categories,
    remove_unused_categories,
    rstrip_categories,
    set_categories,
    union_categories,
    used_categories,
)

pytestmark = pytest.mark.unit


def obj_array(values):
    return np.array(values, dtype=object)


def test_used_mask_basic():
    codes = np.array([0, 2, 2])
    assert _used_mask(codes, 3).tolist() == [True, False, True]


def test_used_mask_missing_code_does_not_wrap_to_last_category():
    # -1 (missing) must never mark the last category as used
    codes = np.array([0, -1])
    assert _used_mask(codes, 3).tolist() == [True, False, False]


def test_used_mask_empty_codes():
    assert _used_mask(np.array([], dtype=int), 3).tolist() == [False, False, False]


def test_codes_and_categories_extracts_raw_arrays():
    cat = pd.Categorical(["b", "a", "b"], categories=["a", "b"])
    codes, categories = codes_and_categories(cat)

    assert codes.tolist() == [1, 0, 1]
    assert categories.tolist() == ["a", "b"]
    assert isinstance(codes, np.ndarray)
    assert isinstance(categories, np.ndarray)


def test_used_categories_excludes_unreferenced():
    codes = np.array([2, 0])
    categories = obj_array(["a", "b", "c"])

    assert used_categories(codes, categories).tolist() == ["a", "c"]


def test_used_categories_ignores_missing_code():
    codes = np.array([0, -1])
    categories = obj_array(["a", "b"])

    assert used_categories(codes, categories).tolist() == ["a"]


def test_union_categories_single_input_passthrough():
    categories = obj_array(["a", "b"])
    result = union_categories([(np.array([0, 1]), categories)])

    assert result is categories


def test_union_categories_preserves_first_appearance_order_across_inputs():
    a = (np.array([0, 1]), obj_array(["b", "a"]))
    b = (np.array([0, 1]), obj_array(["a", "c"]))

    assert union_categories([a, b]).tolist() == ["b", "a", "c"]


def test_union_categories_ignore_unused_restricts_each_input():
    a = (np.array([1]), obj_array(["a", "b"]))
    b = (np.array([1]), obj_array(["c", "d"]))

    assert union_categories([a, b], ignore_unused=True).tolist() == ["b", "d"]
    assert union_categories([a, b], ignore_unused=False).tolist() == [
        "a",
        "b",
        "c",
        "d",
    ]


def test_union_categories_ignore_unused_with_missing_code_in_one_input():
    # a missing (-1) code must not be mistaken for a reference to the last category
    a = (np.array([0, -1]), obj_array(["a", "b"]))
    b = (np.array([0]), obj_array(["c"]))

    assert union_categories([a, b], ignore_unused=True).tolist() == ["a", "c"]


def test_min_int_dtype_picks_smallest_fitting_dtype():
    assert _min_int_dtype(1) is np.int8
    assert _min_int_dtype(np.iinfo(np.int8).max) is np.int8
    assert _min_int_dtype(np.iinfo(np.int8).max + 1) is np.int16
    assert _min_int_dtype(np.iinfo(np.int16).max + 1) is np.int32
    assert _min_int_dtype(np.iinfo(np.int32).max + 1) is np.int64


def test_remove_unused_categories_shrinks_and_remaps():
    codes = np.array([0, 2])
    categories = obj_array(["a", "b", "c"])

    new_codes, new_categories = remove_unused_categories(codes, categories)

    assert new_categories.tolist() == ["a", "c"]
    assert new_codes.tolist() == [0, 1]


def test_remove_unused_categories_preserves_missing_code():
    codes = np.array([0, 1, -1, 2])
    categories = obj_array(["a", "b", "d"])

    new_codes, new_categories = remove_unused_categories(codes, categories)

    assert new_categories.tolist() == ["a", "b", "d"]
    assert new_codes.tolist() == [0, 1, -1, 2]

    # missing code is at the last position
    codes = np.array([0, 1, -1])
    categories = obj_array(["a", "b", "c"])

    new_codes, new_categories = remove_unused_categories(codes, categories)

    assert new_categories.tolist() == ["a", "b"]
    assert new_codes.tolist() == [0, 1, -1]


def test_remove_unused_categories_no_unused_is_identity_remap():
    codes = np.array([1, 0, 1])
    categories = obj_array(["a", "b"])

    new_codes, new_categories = remove_unused_categories(codes, categories)

    assert new_categories.tolist() == ["a", "b"]
    assert new_codes.tolist() == codes.tolist()
    # nothing to drop
    assert new_codes is codes
    assert new_categories is categories


def test_remove_unused_categories_downcasts_dtype():
    codes = np.array([0])
    categories = obj_array(["a"] + [f"unused{i}" for i in range(200)])

    new_codes, _ = remove_unused_categories(codes, categories)

    assert new_codes.dtype == np.int8


def test_remove_categories_empty_removals_is_identity():
    codes = np.array([0, 1])
    categories = obj_array(["a", "b"])

    new_codes, new_categories = remove_categories(codes, categories, set())

    assert new_codes is codes
    assert new_categories is categories


def test_remove_categories_removes_and_shifts_positions():
    codes = np.array([0, 1, 2, 3])
    categories = obj_array(["a", "b", "c", "d"])

    new_codes, new_categories = remove_categories(codes, categories, {"b"})

    assert new_categories.tolist() == ["a", "c", "d"]
    assert new_codes.tolist() == [0, -1, 1, 2]


def test_remove_categories_ignores_removals_not_present():
    codes = np.array([0, 1])
    categories = obj_array(["a", "b"])

    new_codes, new_categories = remove_categories(codes, categories, {"not_a_category"})

    assert new_categories.tolist() == ["a", "b"]
    assert new_codes.tolist() == [0, 1]
    # non-intersecting removal_set is a no-op -- returned as-is, not remapped
    assert new_codes is codes
    assert new_categories is categories


def test_remove_categories_preserves_already_missing_code():
    codes = np.array([0, 1, -1])
    categories = obj_array(["a", "b", "c"])

    new_codes, new_categories = remove_categories(codes, categories, {"a"})

    assert new_categories.tolist() == ["b", "c"]
    assert new_codes.tolist() == [-1, 0, -1]


def test_remove_categories_preserves_already_missing_code_at_last_position():
    codes = np.array([0, 1, -1])
    categories = obj_array(["a", "b", "c"])

    new_codes, new_categories = remove_categories(codes, categories, {"b"})

    assert new_categories.tolist() == ["a", "c"]
    assert new_codes.tolist() == [0, -1, -1]


def test_remove_categories_removing_everything():
    codes = np.array([0, 1])
    categories = obj_array(["a", "b"])

    new_codes, new_categories = remove_categories(codes, categories, {"a", "b"})

    assert new_categories.tolist() == []
    assert new_codes.tolist() == [-1, -1]


def test_set_categories_value_based_remap():
    codes = np.array([0, 1, 2])
    categories = obj_array(["a", "b", "c"])

    new_codes, new_categories = set_categories(codes, categories, obj_array(["c", "a"]))

    # "b" isn't in new_categories -> missing; "a" and "c" land at their new
    # positions
    assert new_codes.tolist() == [1, -1, 0]
    assert new_categories.tolist() == ["c", "a"]


def test_set_categories_value_based_disjoint_becomes_all_missing():
    codes = np.array([0, 1])
    categories = obj_array(["a", "b"])

    new_codes, _ = set_categories(codes, categories, obj_array(["x", "y"]))

    assert new_codes.tolist() == [-1, -1]


def test_set_categories_preserves_already_missing_code():
    codes = np.array([0, -1])
    categories = obj_array(["a", "b"])

    new_codes, _ = set_categories(codes, categories, obj_array(["a"]))

    assert new_codes.tolist() == [0, -1]


def test_set_categories_rejects_duplicate_new_categories():
    codes = np.array([0])
    categories = obj_array(["a"])

    with pytest.raises(ValueError, match="unique"):
        set_categories(codes, categories, obj_array(["x", "x"]), validate=True)


def test_set_categories_does_not_validate_by_default():
    # the sole caller feeds categories taken off an existing Categorical, so the
    # checks are two wasted passes unless asked for
    codes = np.array([0])
    categories = obj_array(["a"])

    _, new_categories = set_categories(codes, categories, obj_array(["x", "x"]))

    assert new_categories.tolist() == ["x", "x"]


def test_set_categories_rejects_null_new_categories():
    codes = np.array([0])
    categories = obj_array(["a"])

    with pytest.raises(ValueError, match="null"):
        set_categories(codes, categories, obj_array(["x", None]), validate=True)


def test_rstrip_categories_already_clean_is_noop():
    categories = obj_array(["a", "b"])

    remap, new_categories = rstrip_categories(categories)

    assert remap is None
    assert new_categories is categories


def test_rstrip_categories_strips_without_collision():
    categories = obj_array(["a ", "b"])

    remap, new_categories = rstrip_categories(categories)

    assert remap is not None
    assert new_categories.tolist() == ["a", "b"]
    assert remap.tolist() == [0, 1]


def test_rstrip_categories_strips_with_collision():
    categories = obj_array(["a ", "a", "b"])

    remap, new_categories = rstrip_categories(categories)

    assert new_categories.tolist() == ["a", "b"]
    # "a " and "a" collapse onto the same new position
    assert remap.tolist() == [0, 0, 1]


def test_rstrip_categories_coerces_non_str_categories():
    categories = np.array([1, 2], dtype=object)

    remap, new_categories = rstrip_categories(categories)

    assert new_categories.tolist() == ["1", "2"]
    assert remap is not None


def test_rstrip_categories_empty_is_noop():
    categories = obj_array([])

    remap, new_categories = rstrip_categories(categories)

    assert remap is None
    assert new_categories is categories


def test_assemble_categorical_fastpath_matches_non_fastpath():
    codes = np.array([1, 0])
    categories = obj_array(["a", "b"])

    fast = assemble_categorical(codes, categories, ordered=True, fastpath=True)
    slow = assemble_categorical(codes, categories, ordered=True, fastpath=False)

    assert fast.tolist() == slow.tolist() == ["b", "a"]
    assert fast.ordered == slow.ordered is True
    assert fast.categories.tolist() == slow.categories.tolist() == ["a", "b"]


def test_assemble_categorical_accepts_plain_list_categories():
    result = assemble_categorical(np.array([0, 1]), ["a", "b"], fastpath=True)

    assert result.tolist() == ["a", "b"]


def test_assemble_categorical_normalizes_empty_categories_dtype():
    result = assemble_categorical(np.array([], dtype=np.int8), [], fastpath=True)

    assert result.categories.dtype == object


def test_apply_remap_relabels_codes():
    codes = np.array([0, 1, 2])
    remap = np.array([2, 0, 1])

    assert apply_remap(codes, remap).tolist() == [2, 0, 1]


def test_apply_remap_keeps_missing_missing():
    # remap[-1] would wrap around to the last entry
    codes = np.array([0, MISSING, 1])
    remap = np.array([5, 6, 7])

    assert apply_remap(codes, remap).tolist() == [5, MISSING, 6]


def test_apply_remap_empty_codes():
    result = apply_remap(np.array([], dtype=np.intp), np.array([1, 2]))

    assert result.tolist() == []


def test_used_categories_all_used_returns_input_untouched():
    categories = obj_array(["a", "b"])

    assert used_categories(np.array([0, 1]), categories) is categories


def test_used_categories_accepts_an_index():
    categories = pd.Index(["a", "b", "c"])

    assert used_categories(np.array([0, 2]), categories).tolist() == ["a", "c"]
