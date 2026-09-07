from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from gamspy import (
    Alias,
    Equation,
    Parameter,
    Set,
    SpecialValues,
    UniverseAlias,
    Variable,
)
from gamspy._internals import GAMS_MAX_INDEX_DIM, DomainStatus
from gamspy.exceptions import ValidationError

pytestmark = pytest.mark.unit

VARIABLE_ATTRIBUTES = ["level", "marginal", "lower", "upper", "scale"]


def categorical(rows, columns, n_domain):
    """DataFrame with the first ``n_domain`` columns as categoricals."""
    frame = pd.DataFrame(rows, columns=columns)
    for column in columns[:n_domain]:
        frame[column] = frame[column].astype("category")

    return frame


def variable_records(rows, domain_columns):
    return categorical(rows, domain_columns + VARIABLE_ATTRIBUTES, len(domain_columns))


def test_domain_labels_setter_scalar_label(container):
    i = Set(container, "i", records=["a", "b"])
    p = Parameter(container, "p", domain=[i], records=[("a", 1.0)])

    p.domain_labels = "label"
    assert p.domain_labels == ["label"]
    assert p.records.columns.tolist() == ["label", "value"]


def test_domain_labels_setter_length_mismatch(container):
    i = Set(container, "i", records=["a", "b"])
    j = Set(container, "j", records=["x", "y"])
    p = Parameter(container, "p", domain=[i, j], records=[("a", "x", 1.0)])

    with pytest.raises(ValidationError):
        p.domain_labels = "only_one_label"

    with pytest.raises(ValidationError):
        p.domain_labels = ["a", "b", "c"]


def test_domain_labels_setter_makes_labels_unique(container):
    i = Set(container, "i", records=["a", "b"])
    p = Parameter(container, "p", domain=[i, i], records=[("a", "b", 1.0)])

    p.domain_labels = ["dup", "dup"]
    assert p.domain_labels == ["dup_0", "dup_1"]


def test_validate_domain_invalid_element_type(container):
    with pytest.raises(TypeError):
        Parameter(container, "p", domain=[1])

    with pytest.raises(TypeError):
        Variable(container, "v", domain=[None])


def test_validate_domain_element_must_be_one_dimensional(container):
    i = Set(container, "i", records=["a"])
    j = Set(container, "j", records=["x"])
    two_dimensional = Set(container, "k", domain=[i, j])

    with pytest.raises(ValueError):
        Parameter(container, "p", domain=[two_dimensional])


def test_validate_domain_too_many_dimensions(container):
    with pytest.raises(ValueError):
        Parameter(container, "p", domain=["*"] * (GAMS_MAX_INDEX_DIM + 1))


def test_domain_violations_without_records(container):
    i = Set(container, "i", records=["a", "b"])
    p = Parameter(container, "p", domain=[i])

    assert p._getDomainViolations() is None
    assert p._findDomainViolations() is None
    assert p._dropDomainViolations() is None


def test_domain_violations_when_domain_has_no_records(container):
    i = Set(container, "i")
    p = Parameter(container, "p", domain=[i])
    p.records = categorical([["zz", 1.0]], ["i", "value"], 1)

    (violation,) = p._getDomainViolations()
    assert violation.dimension == 0
    assert violation.domain is i
    assert violation.violations == ["zz"]


def test_domain_violations_in_multiple_dimensions(container):
    i = Set(container, "i", records=["a", "b"])
    j = Set(container, "j", records=["x", "y"])
    p = Parameter(container, "p", domain=[i, j])
    p.records = categorical(
        [["a", "x", 1.0], ["bad_i", "x", 2.0], ["a", "bad_j", 3.0]],
        ["i", "j", "value"],
        2,
    )

    violations = p._getDomainViolations()
    assert [(v.dimension, v.violations) for v in violations] == [
        (0, ["bad_i"]),
        (1, ["bad_j"]),
    ]

    found = p._findDomainViolations()
    assert found.index.tolist() == [1, 2]

    p._dropDomainViolations()
    assert p.records["value"].tolist() == [1.0]
    assert p._getDomainViolations() == []


def test_find_domain_violations_returns_empty_frame(container):
    i = Set(container, "i", records=["a", "b"])
    p = Parameter(container, "p", domain=[i], records=[("a", 1.0)])

    found = p._findDomainViolations()
    assert found.empty
    assert found.columns.tolist() == p.records.columns.tolist()


def test_assert_valid_records_without_records(container):
    i = Set(container, "i", records=["a", "b"])
    p = Parameter(container, "p", domain=[i])

    assert p._assert_valid_records() is None


def test_assert_valid_records_with_missing_category(container):
    i = Set(container, "i", records=["a", "b"])
    p = Parameter(container, "p", domain=[i], records=[("a", 1.0), ("b", 2.0)])
    p._removeUELs(uels=["a"], dimensions=0)

    with pytest.raises(ValidationError, match="Categories are missing from the data"):
        p._assert_valid_records()


@pytest.fixture
def special_parameter(container):
    i = Set(container, "i", records=[f"i{n}" for n in range(6)])
    yield Parameter(
        container,
        "p",
        domain=[i],
        records=[
            ("i0", SpecialValues.EPS),
            ("i1", SpecialValues.NA),
            ("i2", SpecialValues.UNDEF),
            ("i3", SpecialValues.POSINF),
            ("i4", SpecialValues.NEGINF),
            ("i5", 3.0),
        ],
    )


def test_find_special_values(special_parameter):
    p = special_parameter

    assert p.findEps()["i"].tolist() == ["i0"]
    assert p.findNA()["i"].tolist() == ["i1"]
    assert p.findUndef()["i"].tolist() == ["i2"]
    assert p.findPosInf()["i"].tolist() == ["i3"]
    assert p.findNegInf()["i"].tolist() == ["i4"]


def test_find_special_values_multiple(special_parameter):
    p = special_parameter

    found = p.findSpecialValues(
        [
            SpecialValues.EPS,
            SpecialValues.NA,
            SpecialValues.UNDEF,
            SpecialValues.POSINF,
            SpecialValues.NEGINF,
        ]
    )
    assert found["i"].tolist() == ["i0", "i1", "i2", "i3", "i4"]

    found = p.findSpecialValues([SpecialValues.NA, SpecialValues.EPS])
    assert found["i"].tolist() == ["i0", "i1"]


def test_find_special_values_without_records(container):
    i = Set(container, "i", records=["a"])
    p = Parameter(container, "p", domain=[i])

    assert p.findEps() is None


def test_find_special_values_validations(special_parameter):
    p = special_parameter

    with pytest.raises(TypeError):
        p.findSpecialValues("eps")

    with pytest.raises(TypeError):
        p.findSpecialValues(SpecialValues.EPS, column=5)

    with pytest.raises(TypeError):
        p.findSpecialValues(SpecialValues.EPS, column="level")

    with pytest.raises(ValidationError):
        p.findSpecialValues(1.0)

    with pytest.raises(ValidationError):
        p.findSpecialValues([SpecialValues.EPS, 1.0])


def test_find_special_values_on_variable(container):
    i = Set(container, "i", records=["a", "b"])
    v = Variable(container, "v", domain=[i])
    v.setRecords(
        pd.DataFrame([["a", SpecialValues.NA], ["b", 1.0]], columns=["i", "level"])
    )

    assert v.findNA()["i"].tolist() == ["a"]
    assert v.findSpecialValues(SpecialValues.NA, column="marginal").empty


def test_find_special_values_on_equation(container):
    i = Set(container, "i", records=["a", "b"])
    e = Equation(container, "e", domain=[i])
    e.records = variable_records(
        [
            ["a", SpecialValues.POSINF, 0.0, 0.0, 0.0, 1.0],
            ["b", 1.0, 0.0, 0.0, 0.0, 1.0],
        ],
        ["i"],
    )

    assert e.findPosInf()["i"].tolist() == ["a"]


def test_count_special_values(special_parameter):
    p = special_parameter

    assert p.countEps() == 1
    assert p.countNA() == 1
    assert p.countUndef() == 1
    assert p.countPosInf() == 1
    assert p.countNegInf() == 1


def test_count_special_values_without_records(container):
    i = Set(container, "i", records=["a"])
    p = Parameter(container, "p", domain=[i])

    assert p.countNA() == 0
    assert type(p.countNA()) is int
    assert p.countEps() == 0
    assert p.countUndef() == 0
    assert p.countPosInf() == 0
    assert p.countNegInf() == 0


def test_count_special_values_validations(special_parameter):
    p = special_parameter

    with pytest.raises(TypeError):
        p.countNA(columns=5)

    with pytest.raises(TypeError):
        p.countNA(columns=["value", 5])

    with pytest.raises(TypeError):
        p.countNA(columns=["level"])

    with pytest.raises(ValidationError):
        p._countSpecialValues(1.0, columns=None)


def test_count_special_values_on_variable(container):
    i = Set(container, "i", records=["a", "b"])
    v = Variable(container, "v", domain=[i])
    v.setRecords(
        pd.DataFrame([["a", SpecialValues.EPS], ["b", 1.0]], columns=["i", "level"])
    )

    assert v.countEps() == 1
    assert v.countEps(columns=["level", "marginal"]) == 1


def test_where_metrics_on_parameter(container):
    i = Set(container, "i", records=["a", "b", "c"])
    p = Parameter(
        container, "p", domain=[i], records=[("a", 5.0), ("b", 1.0), ("c", -3.0)]
    )

    assert p.whereMax() == ["a"]
    assert p.whereMin() == ["c"]
    assert p.whereMaxAbs() == ["a"]


def test_where_max_abs_of_negative_value(container):
    i = Set(container, "i", records=["a", "b"])
    p = Parameter(container, "p", domain=[i], records=[("a", -7.0), ("b", 1.0)])

    # the largest magnitude belongs to a negative record
    assert p.whereMaxAbs() == ["a"]
    assert p.getMaxAbsValue() == 7.0


def test_where_max_abs_ties_report_the_first_record(container):
    i = Set(container, "i", records=["a", "b"])
    p = Parameter(container, "p", domain=[i], records=[("a", -4.0), ("b", 4.0)])

    assert p.whereMaxAbs() == ["a"]


def test_where_metrics_on_all_na_column(container):
    i = Set(container, "i", records=["a", "b"])
    p = Parameter(
        container,
        "p",
        domain=[i],
        records=[("a", SpecialValues.NA), ("b", SpecialValues.UNDEF)],
    )

    assert p.whereMax() is None
    assert p.whereMin() is None
    assert p.whereMaxAbs() is None


def test_where_metrics_on_multi_dimensional_symbol(container):
    i = Set(container, "i", records=["a", "b"])
    j = Set(container, "j", records=["x", "y"])
    p = Parameter(
        container,
        "p",
        domain=[i, j],
        records=[("a", "x", 1.0), ("b", "y", -9.0)],
    )

    assert p.whereMax() == ["a", "x"]
    assert p.whereMin() == ["b", "y"]
    assert p.whereMaxAbs() == ["b", "y"]


def test_where_metrics_on_scalar(container):
    p = Parameter(container, "p", records=7.0)

    assert p.whereMax() is None
    assert p.whereMin() is None
    assert p.whereMaxAbs() is None


def test_where_metrics_on_empty_records(container):
    i = Set(container, "i", records=["a", "b"])
    p = Parameter(container, "p", domain=[i], records=[("a", 1.0)])
    p.records = p.records.iloc[0:0]

    assert p.whereMax() is None
    assert p.whereMin() is None
    assert p.whereMaxAbs() is None


def test_where_metrics_without_records(container):
    i = Set(container, "i", records=["a", "b"])
    p = Parameter(container, "p", domain=[i])

    assert p.whereMax() is None


def test_where_metrics_validations(container):
    i = Set(container, "i", records=["a", "b"])
    p = Parameter(container, "p", domain=[i], records=[("a", 1.0)])

    with pytest.raises(TypeError):
        p.whereMax(column=5)

    with pytest.raises(TypeError):
        p.whereMax(column="level")


def test_where_metrics_on_variable(container):
    i = Set(container, "i", records=["a", "b", "c"])
    v = Variable(container, "v", domain=[i])
    v.setRecords(
        pd.DataFrame([["a", 3.0], ["b", 1.0], ["c", 2.0]], columns=["i", "level"])
    )

    assert v.whereMax() == ["a"]
    assert v.whereMin() == ["b"]
    assert v.whereMaxAbs() == ["a"]


def test_get_metrics_on_parameter(container):
    i = Set(container, "i", records=["a", "b", "c"])
    p = Parameter(
        container, "p", domain=[i], records=[("a", -5.0), ("b", 1.0), ("c", 3.0)]
    )

    assert p.getMaxValue() == 3.0
    assert p.getMinValue() == -5.0
    assert p.getMeanValue() == pytest.approx(-1.0 / 3.0)
    assert p.getMaxAbsValue() == 5.0


def test_get_mean_value_with_both_infinities(container):
    i = Set(container, "i", records=["a", "b", "c"])
    p = Parameter(
        container,
        "p",
        domain=[i],
        records=[
            ("a", SpecialValues.NEGINF),
            ("b", 1.0),
            ("c", SpecialValues.POSINF),
        ],
    )

    assert np.isnan(p.getMeanValue())


def test_get_metrics_without_records(container):
    i = Set(container, "i", records=["a"])
    p = Parameter(container, "p", domain=[i])

    assert p.getMaxValue() is None


def test_get_metrics_validations(container):
    i = Set(container, "i", records=["a", "b"])
    p = Parameter(container, "p", domain=[i], records=[("a", 1.0)])

    with pytest.raises(TypeError):
        p.getMaxValue(columns=5)

    with pytest.raises(TypeError):
        p.getMaxValue(columns=["value", 5])

    with pytest.raises(TypeError):
        p.getMaxValue(columns=["level"])


def test_get_metrics_on_variable(container):
    i = Set(container, "i", records=["a", "b"])
    v = Variable(container, "v", domain=[i])
    v.setRecords(pd.DataFrame([["a", -4.0], ["b", 2.0]], columns=["i", "level"]))

    assert v.getMaxValue() == 2.0
    assert v.getMinValue() == -4.0
    assert v.getMeanValue() == -1.0
    assert v.getMaxAbsValue() == 4.0


def test_find_special_values_with_an_empty_sequence(special_parameter):
    assert special_parameter.findSpecialValues([]).empty


def test_find_special_values_accepts_a_tuple(special_parameter):
    found = special_parameter.findSpecialValues((SpecialValues.NA, SpecialValues.EPS))

    assert found["i"].tolist() == ["i0", "i1"]


def test_find_special_values_rejects_a_bool(special_parameter):
    with pytest.raises(TypeError):
        special_parameter.findSpecialValues(True)


def test_find_special_values_reports_an_unknown_value_without_records(container):
    i = Set(container, "i", records=["a"])
    p = Parameter(container, "p", domain=[i])

    with pytest.raises(ValidationError):
        p.findSpecialValues(1.0)


def test_count_special_values_returns_a_builtin_int(special_parameter):
    count = special_parameter.countNA()

    assert type(count) is int


def test_count_special_values_across_multiple_columns(container):
    i = Set(container, "i", records=["a", "b"])
    v = Variable(container, "v", domain=[i])
    v.records = variable_records(
        [
            ["a", SpecialValues.EPS, SpecialValues.EPS, 0.0, 0.0, 1.0],
            ["b", 1.0, SpecialValues.EPS, 0.0, 0.0, 1.0],
        ],
        ["i"],
    )

    assert v.countEps() == 1
    assert v.countEps(columns=["level", "marginal"]) == 3


def test_count_special_values_rejects_empty_columns(special_parameter):
    with pytest.raises(ValidationError):
        special_parameter.countNA(columns=[])


def test_get_metrics_return_builtin_floats(container):
    i = Set(container, "i", records=["a", "b"])
    p = Parameter(container, "p", domain=[i], records=[("a", -5.0), ("b", 3.0)])

    for value in (
        p.getMaxValue(),
        p.getMinValue(),
        p.getMeanValue(),
        p.getMaxAbsValue(),
    ):
        assert type(value) is float


def test_get_metrics_reject_empty_columns(container):
    i = Set(container, "i", records=["a", "b"])
    p = Parameter(container, "p", domain=[i], records=[("a", 1.0)])

    with pytest.raises(ValidationError):
        p.getMaxValue(columns=[])


def test_get_metrics_on_an_all_na_column(container):
    i = Set(container, "i", records=["a"])
    p = Parameter(container, "p", domain=[i], records=[("a", SpecialValues.NA)])

    assert np.isnan(p.getMaxValue())
    assert np.isnan(p.getMinValue())
    assert np.isnan(p.getMeanValue())
    assert np.isnan(p.getMaxAbsValue())


def test_get_mean_value_does_not_warn_on_infinities(container):
    i = Set(container, "i", records=["a", "b"])
    p = Parameter(
        container,
        "p",
        domain=[i],
        records=[("a", SpecialValues.NEGINF), ("b", SpecialValues.POSINF)],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(p.getMeanValue())


def test_get_metrics_across_multiple_columns(container):
    i = Set(container, "i", records=["a", "b"])
    v = Variable(container, "v", domain=[i])
    v.records = variable_records(
        [
            ["a", -8.0, 2.0, 0.0, 0.0, 1.0],
            ["b", 1.0, 4.0, 0.0, 0.0, 1.0],
        ],
        ["i"],
    )

    assert v.getMaxValue(columns=["level", "marginal"]) == 4.0
    assert v.getMinValue(columns=["level", "marginal"]) == -8.0
    assert v.getMaxAbsValue(columns=["level", "marginal"]) == 8.0


def test_get_uels_distinguishes_used_from_declared_categories(container):
    i = Set(container, "i", records=["a", "b", "c"])
    p = Parameter(container, "p", domain=[i])
    p.records = pd.DataFrame(
        {
            "i": pd.Categorical(["c", "a"], categories=["a", "b", "c"], ordered=True),
            "value": [1.0, 2.0],
        }
    )

    # both are reported in category order, not in record order
    assert p._getUELs(0) == ["a", "b", "c"]
    assert p._getUELs(0, ignore_unused=True) == ["a", "c"]


def test_get_uels_ignores_missing_domain_entries(container):
    i = Set(container, "i", records=["a", "b"])
    p = Parameter(container, "p", domain=[i])
    p.records = pd.DataFrame(
        {
            "i": pd.Categorical(["a", None], categories=["a", "b"], ordered=True),
            "value": [1.0, 2.0],
        }
    )

    # the missing entry has code -1, which refers to no category at all
    assert p.records["i"].cat.codes.tolist() == [0, -1]
    assert p._getUELs(0, ignore_unused=True) == ["a"]


def test_get_uels_across_dimensions_keeps_first_seen_order(container):
    i = Set(container, "i", records=["b", "a"])
    j = Set(container, "j", records=["a", "c"])
    p = Parameter(container, "p", domain=[i, j])
    p.records = pd.DataFrame(
        {
            "i": pd.Categorical(["b"], categories=["b", "a"], ordered=True),
            "j": pd.Categorical(["c"], categories=["a", "c"], ordered=True),
            "value": [1.0],
        }
    )

    # "a" is declared in both dimensions and must be reported once, where it
    # was first seen
    assert p._getUELs() == ["b", "a", "c"]
    assert p._getUELs(ignore_unused=True) == ["b", "c"]
    assert p._getUELs([0, 1]) == p._getUELs()


def test_get_uels_validations(container):
    i = Set(container, "i", records=["a", "b"])
    p = Parameter(container, "p", domain=[i], records=[("a", 1.0)])

    with pytest.raises(TypeError):
        p._getUELs("0")

    with pytest.raises(TypeError):
        p._getUELs(["0"])

    with pytest.raises(ValueError):
        p._getUELs(1)


def test_get_uel_codes(container):
    i = Set(container, "i", records=["b", "a", "c"])
    p = Parameter(container, "p", domain=[i], records=[("a", 1.0)])

    # the codes follow the declaration order of the set, not the data order
    assert i._getUELCodes(0) == {"b": 0, "a": 1, "c": 2}
    assert p._getUELCodes(0) == {"a": 0}

    ii = Alias(container, "ii", alias_with=i)
    assert ii._getUELCodes(0) == i._getUELCodes(0)


def test_shape_counts_used_uels(container):
    i = Set(container, "i", records=["a", "b", "c"])
    j = Set(container, "j", records=["x", "y"])
    p = Parameter(container, "p", domain=[i, j], records=[("a", "x", 1.0)])

    assert p.shape == (3, 2)

    relaxed = Parameter(
        container, "relaxed", domain=["*"], records=[("a", 1.0), ("b", 2.0)]
    )
    assert relaxed.shape == (2,)


def test_to_sparse_coo_without_records(container):
    i = Set(container, "i", records=["a", "b"])
    v = Variable(container, "v", domain=[i])

    assert v.toSparseCoo() is None


def test_to_sparse_coo_validations(container):
    i = Set(container, "i", records=["a", "b"])
    v = Variable(container, "v", domain=[i])

    with pytest.raises(TypeError):
        v.toSparseCoo(column=5)

    with pytest.raises(TypeError):
        v.toSparseCoo(column="value")


def test_to_sparse_coo_relaxed_domain(container):
    v = Variable(container, "v", domain=["*"])
    v.setRecords(pd.DataFrame([["a", 1.0], ["b", 2.0]], columns=["uni", "level"]))

    assert v._domain_status is not DomainStatus.regular
    assert np.allclose(v.toSparseCoo().toarray(), [[1.0, 2.0]])


def test_to_sparse_coo_relaxed_domain_two_dimensional(container):
    v = Variable(container, "v", domain=["*", "*"])
    v.setRecords(
        pd.DataFrame(
            [["a", "x", 1.0], ["b", "y", 2.0]],
            columns=["uni_0", "uni_1", "level"],
        )
    )

    assert v._domain_status is not DomainStatus.regular
    assert np.allclose(v.toSparseCoo().toarray(), [[1.0, 0.0], [0.0, 2.0]])


def test_to_sparse_coo_parameter(container):
    i = Set(container, "i", records=["a", "b"])
    j = Set(container, "j", records=["x", "y"])

    zero_dimensional = Parameter(container, "s", records=42.5)
    assert np.allclose(zero_dimensional.toSparseCoo().toarray(), [[42.5]])

    one_dimensional = Parameter(container, "p", domain=[i], records=[("b", 2.0)])
    assert np.allclose(one_dimensional.toSparseCoo().toarray(), [[0.0, 2.0]])

    two_dimensional = Parameter(
        container, "q", domain=[i, j], records=[("a", "x", 1.0), ("b", "y", 2.0)]
    )
    assert np.allclose(
        two_dimensional.toSparseCoo().toarray(), [[1.0, 0.0], [0.0, 2.0]]
    )

    assert Parameter(container, "empty", domain=[i]).toSparseCoo() is None


def test_to_sparse_coo_parameter_relaxed_domain(container):
    p = Parameter(container, "p", domain=["*", "*"])
    p.setRecords(
        pd.DataFrame(
            [["a", "x", 1.0], ["b", "y", 2.0]],
            columns=["uni_0", "uni_1", "value"],
        )
    )

    assert p._domain_status is not DomainStatus.regular
    assert np.allclose(p.toSparseCoo().toarray(), [[1.0, 0.0], [0.0, 2.0]])


def test_to_sparse_coo_rejects_more_than_two_dimensions(container):
    i = Set(container, "i", records=["a"])
    j = Set(container, "j", records=["x"])
    k = Set(container, "k", records=["1"])
    p = Parameter(container, "p", domain=[i, j, k], records=[("a", "x", "1", 1.0)])

    with pytest.raises(ValidationError, match="dimension <= 2"):
        p.toSparseCoo()


def test_equals_compares_symbol_attributes(container):
    i = Set(container, "i", records=["a", "b"])
    j = Set(container, "j", records=["x", "y"])

    p = Parameter(container, "p", domain=[i], records=[("a", 1.0), ("b", 2.0)])

    # dimension mismatch
    assert not p.equals(
        Parameter(container, "two_dimensional", domain=[i, j]), check_meta_data=False
    )

    # domain_type mismatch
    assert not p.equals(
        Parameter(container, "relaxed", domain=["*"], records=[("a", 1.0), ("b", 2.0)]),
        check_meta_data=False,
    )

    # number of records mismatch
    assert not p.equals(
        Parameter(container, "shorter", domain=[i], records=[("a", 1.0)]),
        check_meta_data=False,
    )

    # records type mismatch, one symbol has no records at all
    assert not p.equals(
        Parameter(container, "without_records", domain=[i]), check_meta_data=False
    )

    # name and description are only compared when asked for
    same = Parameter(
        container,
        "same",
        domain=[i],
        records=[("a", 1.0), ("b", 2.0)],
        description="a description",
    )
    assert same.equals(p, check_meta_data=False)
    assert not same.equals(p)


def test_to_dense_relaxed_domain(container):
    v = Variable(container, "v", domain=["*", "*"])
    v.setRecords(
        pd.DataFrame(
            [["a", "x", 1.0], ["b", "y", 2.0]],
            columns=["uni_0", "uni_1", "level"],
        )
    )

    assert np.allclose(v.toDense(), [[1.0, 0.0], [0.0, 2.0]])


def test_to_dense_regular_domain_with_unordered_uels(container):
    i = Set(container, "i", records=["a", "b"])
    v = Variable(container, "v", domain=[i])
    v.setRecords(pd.DataFrame([["a", 1.0], ["b", 2.0]], columns=["i", "level"]))

    # data order of the domain set no longer matches its category order
    i.records = i.records.iloc[::-1].reset_index(drop=True)

    with pytest.raises(ValidationError):
        v.toDense()


def test_to_dense_relaxed_domain_with_invalid_category(container):
    v = Variable(container, "v", domain=["*"])
    v.setRecords(pd.DataFrame([["a", 1.0], ["b", 2.0]], columns=["uni", "level"]))
    v._removeUELs(uels=["a"], dimensions=0)

    with pytest.raises(ValidationError):
        v.toDense()


def test_to_dense_relaxed_domain_with_unordered_uels(container):
    v = Variable(container, "v", domain=["*"])
    v.records = variable_records(
        [["b", 1.0, 0.0, 0.0, 0.0, 1.0], ["a", 2.0, 0.0, 0.0, 0.0, 1.0]],
        ["uni"],
    )

    with pytest.raises(ValidationError):
        v.toDense()


def test_redeclaration_is_idempotent_for_every_symbol_type(container):
    # SymbolConstructor must hand a repeated declaration to _redefine instead of
    # re-running __init__, so no symbol type can queue a second declaration.
    i = Set(container, "i", records=["i1"])
    factories = {
        "s": lambda: Set(container, "s"),
        "p": lambda: Parameter(container, "p"),
        "v": lambda: Variable(container, "v"),
        "e": lambda: Equation(container, "e"),
        "a": lambda: Alias(container, "a", alias_with=i),
        "u": lambda: UniverseAlias(container, "u"),
    }

    for name, make in factories.items():
        first = make()
        assert make() is first, name
        assert [s for s in container._unsaved_statements if s is first] == [first], name


def test_redeclaration_with_a_different_symbol_type_is_rejected(container):
    _ = Set(container, "i")

    with pytest.raises(TypeError):
        _ = Parameter(container, "i")

    with pytest.raises(TypeError):
        _ = Variable(container, "i")

    with pytest.raises(TypeError):
        _ = Alias(container, "i", alias_with=Set(container, "j"))


def test_redeclaration_updates_description(container):
    factories = {
        "s": lambda description: Set(container, "s", description=description),
        "p": lambda description: Parameter(container, "p", description=description),
        "v": lambda description: Variable(container, "v", description=description),
        "e": lambda description: Equation(container, "e", description=description),
    }

    for name, make in factories.items():
        make("original")
        assert make("updated").description == "updated", name


def test_redeclaration_without_description(container):
    factories = {
        "s": lambda **kwargs: Set(container, "s", **kwargs),
        "p": lambda **kwargs: Parameter(container, "p", **kwargs),
        "v": lambda **kwargs: Variable(container, "v", **kwargs),
        "e": lambda **kwargs: Equation(container, "e", **kwargs),
    }

    for name, make in factories.items():
        make(description="keep me")
        assert make().description == "keep me", name


def test_failed_declaration_restores_miro_protect(container):
    i = Set(container, "i", records=["i1"])
    classes = {"s": Set, "p": Parameter, "v": Variable, "e": Equation}

    for name, cls in classes.items():
        # first declaration, which goes through __init__
        with pytest.raises(TypeError):
            cls(container, f"{name}_new", domain=i, records=object())

        assert container._options.miro_protect is True, name

        # redeclaration of an existing symbol, which goes through _redefine
        existing = f"{name}_existing"
        cls(container, existing, domain=i)
        with pytest.raises(TypeError):
            cls(container, existing, domain=i, records=object())

        assert container._options.miro_protect is True, name


def test_failed_declaration_restores_a_disabled_miro_protect(container):
    # The previous value is restored, not hardcoded back to True.
    i = Set(container, "i", records=["i1"])
    container._options.miro_protect = False

    with pytest.raises(TypeError):
        Parameter(container, "p", domain=i, records=object())

    assert container._options.miro_protect is False
