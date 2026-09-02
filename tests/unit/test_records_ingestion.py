from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gamspy import (
    Equation,
    Parameter,
    Set,
    SpecialValues,
    Variable,
)
from gamspy.exceptions import GdxException

pytestmark = pytest.mark.unit


def test_object_domain_labels_are_rstripped(container):
    p = Parameter(container, "p", domain=["*"])
    p.setRecords(pd.DataFrame({"d": ["a ", "b"], "v": [1.0, 2.0]}))

    assert p.records["d"].tolist() == ["a", "b"]


def test_object_domain_labels_collapse_after_rstrip(container):
    # "a " and "a" are the same GAMS UEL; the pair stays distinct only because
    # the second dimension differs
    p = Parameter(container, "p", domain=["*", "*"])
    p.setRecords(pd.DataFrame({"d0": ["a ", "a"], "d1": ["x", "y"], "v": [1.0, 2.0]}))

    assert p.records["d0"].tolist() == ["a", "a"]
    assert p.toList() == [("a", "x", 1.0), ("a", "y", 2.0)]


def test_categorical_domain_categories_are_rstripped(container):
    p = Parameter(container, "p", domain=["*"])
    p.setRecords(pd.DataFrame({"d": pd.Categorical(["a ", "b"]), "v": [1.0, 2.0]}))

    assert p.records["d"].cat.categories.tolist() == ["a", "b"]
    assert p.records["d"].tolist() == ["a", "b"]


def test_categorical_domain_categories_collapse_after_rstrip(container):
    # the unused "a " category collapses onto "a", so the category list shrinks
    # and the column has to be re-encoded
    p = Parameter(container, "p", domain=["*"])
    p.setRecords(
        pd.DataFrame(
            {
                "d": pd.Categorical(["a", "b"], categories=["a", "a ", "b"]),
                "v": [1.0, 2.0],
            }
        )
    )

    assert p.records["d"].cat.categories.tolist() == ["a", "b"]
    assert p.toList() == [("a", 1.0), ("b", 2.0)]


def test_rstrip_collapse_that_creates_duplicates_is_rejected(container):
    # Labels differing only by trailing whitespace collapse to one UEL. Here that
    # makes two identical keys, which GAMS rejects when the records are written.
    p = Parameter(container, "p", domain=["*"])
    with pytest.raises(GdxException, match="data errors"):
        p.setRecords(pd.DataFrame({"d": ["a ", "a"], "v": [1.0, 2.0]}))


def test_set_domain_labels_are_rstripped(container):
    s = Set(container, "s", domain=["*", "*"])
    s.setRecords(pd.DataFrame({"d0": ["a ", "a"], "d1": ["x", "y"]}))

    assert s.toList() == [("a", "x"), ("a", "y")]


def test_string_special_values_are_remapped(container):
    p = Parameter(container, "p", domain=["*"])
    p.setRecords(
        pd.DataFrame({"d": ["e", "n", "u", "w"], "v": ["eps", "NA", "Undef", "undf"]})
    )

    values = p.records["v"] if "v" in p.records else p.records.iloc[:, -1]
    assert SpecialValues.isEps(values[0])
    assert SpecialValues.isNA(values[1])
    assert SpecialValues.isUndef(values[2])
    assert SpecialValues.isUndef(values[3])


def test_zero_records_are_filtered_but_eps_is_kept(container):
    i = Set(container, "i", records=["a", "b", "c"])
    p = Parameter(container, "p", domain=[i])
    p.setRecords(np.array([0.0, 3.0, SpecialValues.EPS]))

    labels = [row[0] for row in p.toList()]
    assert labels == ["b", "c"]


def test_parameter_scalar_value_on_non_scalar(container):
    i = Set(container, "i", records=["a"])
    p = Parameter(container, "p", domain=[i])

    with pytest.raises(ValueError, match="cannot be set with a scalar value"):
        container.setRecords({p: 5})


def test_parameter_scalar_value_through_container(container):
    p = Parameter(container, "p")
    container.setRecords({p: 5})

    assert p.toValue() == 5.0


def test_parameter_ndarray_row_and_column_vectors(container):
    i = Set(container, "i", records=["a", "b"])

    row = Parameter(container, "row", domain=[i])
    row.setRecords(np.array([[1.0, 2.0]]))
    assert row.toList() == [("a", 1.0), ("b", 2.0)]

    col = Parameter(container, "col", domain=[i])
    col.setRecords(np.array([[1.0], [2.0]]))
    assert col.toList() == [("a", 1.0), ("b", 2.0)]


def test_parameter_scalar_ndarray(container):
    p = Parameter(container, "p")
    p.setRecords(np.array(5.0))

    assert p.toValue() == 5.0


def test_parameter_ndarray_validations(container):
    i = Set(container, "i", records=["a", "b"])
    j = Set(container, "j", records=["x", "y"])

    with pytest.raises(TypeError, match="Attempted conversion to numpy array"):
        Parameter(container, "bad", domain=[i]).setRecords(np.array([object()]))

    with pytest.raises(ValueError, match="Dimension mismatch"):
        Parameter(container, "ndim", domain=[i, j]).setRecords(np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="requires a 'regular' domain type"):
        Parameter(container, "relaxed", domain=["*"]).setRecords(np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="Shape mismatch"):
        Parameter(container, "shape", domain=[i]).setRecords(np.array([1.0, 2.0, 3.0]))


def test_parameter_flat_dataframe_validations(container):
    i = Set(container, "i", records=["a", "b"])

    with pytest.raises(ValueError, match="Dimensionality of records"):
        Parameter(container, "dim", domain=[i]).setRecords(
            pd.DataFrame({"d": ["a"], "x": [1.0], "y": [2.0]})
        )

    with pytest.raises(ValueError, match="multiple records for a scalar"):
        Parameter(container, "sc").setRecords(pd.DataFrame({"v": [1.0, 2.0]}))


def test_parameter_from_table_dataframe(container):
    i = Set(container, "i", records=["a", "b"])
    j = Set(container, "j", records=["x", "y"])
    p = Parameter(container, "p", domain=[i, j])

    p.setRecords(
        pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=["a", "b"], columns=["x", "y"]),
        uels_on_axes=True,
    )

    assert p.toList() == [
        ("a", "x", 1.0),
        ("a", "y", 2.0),
        ("b", "x", 3.0),
        ("b", "y", 4.0),
    ]


def test_parameter_table_dataframe_dimension_mismatch(container):
    i = Set(container, "i", records=["a"])

    # a DataFrame table implies two dimensions, the parameter only has one
    with pytest.raises(ValueError, match="Dimensionality of table"):
        Parameter(container, "p", domain=[i]).setRecords(
            pd.DataFrame([[1.0]], index=["a"], columns=["x"]),
            uels_on_axes=True,
        )


def test_parameter_from_series(container):
    scalar = Parameter(container, "scalar_p")
    scalar.setRecords(pd.Series([4.0]))
    assert scalar.toValue() == 4.0

    i = Set(container, "i", records=["a", "b"])
    p = Parameter(container, "p", domain=[i])
    p.setRecords(pd.Series([1.0, 2.0], index=["a", "b"]), uels_on_axes=True)
    assert p.toList() == [("a", 1.0), ("b", 2.0)]


def test_parameter_series_validations(container):
    i = Set(container, "i", records=["a", "b"])
    j = Set(container, "j", records=["x"])

    with pytest.raises(ValueError, match="size exactly = 1"):
        Parameter(container, "sc").setRecords(pd.Series([1.0, 2.0]))

    with pytest.raises(ValueError, match="Dimensionality of data"):
        Parameter(container, "p", domain=[i, j]).setRecords(
            pd.Series([1.0], index=["a"])
        )


def test_parameter_from_dict(container):
    p = Parameter(container, "p", domain=["*"])
    p.setRecords({"d": ["a"], "v": [1.0]})

    assert p.toList() == [("a", 1.0)]


def test_parameter_from_empty_iterable(container):
    p = Parameter(container, "p", domain=["*"])
    p.setRecords([])

    assert p.toList() == []


def test_parameter_sequence_dimension_mismatch(container):
    i = Set(container, "i", records=["a"])

    with pytest.raises(ValueError, match="Dimensionality of records"):
        Parameter(container, "p", domain=[i]).setRecords([("a", 1.0, 2.0)])


def test_parameter_unconvertible_records(container):
    p = Parameter(container, "p", domain=["*"])

    with pytest.raises(TypeError, match="Could not convert to pandas DataFrame"):
        p.setRecords(object())


def test_set_cannot_be_initialized_with_a_number(container):
    with pytest.raises(TypeError, match="Sets cannot be initialized"):
        Set(container, "s").setRecords(5)


def test_set_from_ndarray(container):
    s = Set(container, "s")
    s.setRecords(np.array(["a", "b"]))

    assert s.toList() == ["a", "b"]


def test_set_from_bool_table(container):
    i = Set(container, "i", records=["a", "b"])
    j = Set(container, "j", records=["x", "y"])
    s = Set(container, "s", domain=[i, j])

    s.setRecords(
        pd.DataFrame(
            [[True, False], [False, True]], index=["a", "b"], columns=["x", "y"]
        ),
        uels_on_axes=True,
    )
    assert s.toList() == [("a", "x"), ("b", "y")]


def test_set_from_object_table_is_converted(container):
    i = Set(container, "i", records=["a", "b"])
    j = Set(container, "j", records=["x", "y"])
    s = Set(container, "s", domain=[i, j])

    # object dtype that convert_dtypes() can turn into bool
    s.setRecords(
        pd.DataFrame(
            [[True, False], [False, True]], index=["a", "b"], columns=["x", "y"]
        ).astype(object),
        uels_on_axes=True,
    )
    assert s.toList() == [("a", "x"), ("b", "y")]


def test_set_table_validations(container):
    i = Set(container, "i", records=["a", "b"])
    j = Set(container, "j", records=["x", "y"])
    bool_df = pd.DataFrame(
        [[True, False], [False, True]], index=["a", "b"], columns=["x", "y"]
    )

    with pytest.raises(TypeError, match="must be type bool"):
        Set(container, "s1", domain=[i, j]).setRecords(
            pd.DataFrame([[1, 0], [0, 1]], index=["a", "b"], columns=["x", "y"]),
            uels_on_axes=True,
        )

    with pytest.raises(ValueError, match="Dimensionality of table"):
        Set(container, "s2", domain=[i]).setRecords(bool_df, uels_on_axes=True)


def test_set_from_series(container):
    plain = Set(container, "plain")
    plain.setRecords(pd.Series(["a", "b"]))
    assert plain.toList() == ["a", "b"]

    i = Set(container, "i", records=["a", "b"])
    on_axes = Set(container, "on_axes", domain=[i])
    on_axes.setRecords(pd.Series([True, False], index=["a", "b"]), uels_on_axes=True)

    assert on_axes.toList() == ["a", "b"]
    assert on_axes.records["element_text"].tolist() == ["True", "False"]


def test_set_series_dimension_mismatch(container):
    i = Set(container, "i", records=["a"])
    j = Set(container, "j", records=["x"])

    with pytest.raises(ValueError, match="Dimensionality of data"):
        Set(container, "s", domain=[i, j]).setRecords(pd.Series(["a"]))

    # same mismatch, taken through the uels_on_axes branch
    with pytest.raises(ValueError, match="Dimensionality of data"):
        Set(container, "s2", domain=[i, j]).setRecords(
            pd.Series([True], index=["a"]), uels_on_axes=True
        )


def test_set_from_dict(container):
    s = Set(container, "s")
    s.setRecords({0: ["a", "b"]})

    assert s.toList() == ["a", "b"]


def test_set_from_string(container):
    s = Set(container, "s")
    s.setRecords("only")

    assert s.toList() == ["only"]


def test_set_from_empty_iterable(container):
    s = Set(container, "s")
    s.setRecords([])

    assert s.toList() == []


def test_set_dimensionality_mismatch(container):
    i = Set(container, "i", records=["a"])

    with pytest.raises(ValueError, match="Dimensionality of records"):
        Set(container, "s", domain=[i]).setRecords([("a", "text", "extra")])


def test_set_unconvertible_records(container):
    with pytest.raises(TypeError, match="Could not convert to pandas DataFrame"):
        Set(container, "s").setRecords(object())


def test_varequ_scalar_value_on_non_scalar(container):
    i = Set(container, "i", records=["a"])

    with pytest.raises(ValueError, match="symbol is not scalar"):
        Variable(container, "v", domain=[i]).setRecords(3.0)


def test_varequ_scalar_value(container):
    v = Variable(container, "v")
    v.setRecords(2.5)

    assert v.toValue() == 2.5


def test_varequ_from_ndarray(container):
    i = Set(container, "i", records=["a", "b"])
    v = Variable(container, "v", domain=[i])
    v.setRecords(np.array([1.0, 2.0]))

    assert v.records["level"].tolist() == [1.0, 2.0]
    # unspecified attributes fall back to the variable defaults
    assert v.records["lower"].tolist() == [float("-inf")] * 2


def test_varequ_dict_of_arrays(container):
    i = Set(container, "i", records=["a", "b"])
    v = Variable(container, "v", domain=[i])
    v.setRecords({"level": np.array([1.0, 2.0]), "marginal": np.array([0.1, 0.2])})

    assert v.records["level"].tolist() == [1.0, 2.0]
    assert v.records["marginal"].tolist() == [0.1, 0.2]


def test_varequ_dict_of_arrays_validations(container):
    i = Set(container, "i", records=["a", "b"])
    j = Set(container, "j", records=["x"])

    with pytest.raises(ValueError, match="Dimensionality mismatch"):
        Variable(container, "ndim", domain=[i, j]).setRecords(
            {"level": np.array([1.0, 2.0])}
        )

    with pytest.raises(ValueError, match="do not have the same shape"):
        Variable(container, "shapes", domain=[i]).setRecords(
            {"level": np.array([1.0, 2.0]), "marginal": np.array([1.0])}
        )

    with pytest.raises(ValueError, match="requires a 'regular' domain type"):
        Variable(container, "relaxed", domain=["*"]).setRecords(
            {"level": np.array([1.0, 2.0])}
        )

    with pytest.raises(ValueError, match="Shape mismatch"):
        Variable(container, "shape", domain=[i]).setRecords(
            {"level": np.array([1.0, 2.0, 3.0])}
        )


def test_varequ_dict_of_arrays_row_and_column_vectors(container):
    i = Set(container, "i", records=["a", "b"])

    row = Variable(container, "row", domain=[i])
    row.setRecords({"level": np.array([[1.0, 2.0]])})
    assert row.records["level"].tolist() == [1.0, 2.0]

    col = Variable(container, "col", domain=[i])
    col.setRecords({"level": np.array([[1.0], [2.0]])})
    assert col.records["level"].tolist() == [1.0, 2.0]


def test_varequ_unconvertible_records(container):
    i = Set(container, "i", records=["a"])

    with pytest.raises(TypeError, match="Could not convert to pandas DataFrame"):
        Variable(container, "v", domain=[i]).setRecords(object())


def test_varequ_scalar_dict_of_arrays(container):
    v = Variable(container, "v")
    v.setRecords({"level": 3.0, "marginal": 0.5})

    assert v.records["level"].tolist() == [3.0]
    assert v.records["marginal"].tolist() == [0.5]


def test_varequ_flat_dataframe_validations(container):
    i = Set(container, "i", records=["a", "b"])

    with pytest.raises(ValueError, match="Dimensionality of records"):
        Variable(container, "dim", domain=[i]).setRecords(
            pd.DataFrame({"i": ["a"], "level": [1.0], "extra": [2.0]})
        )

    with pytest.raises(ValueError, match="multiple records for a scalar"):
        Variable(container, "sc").setRecords(pd.DataFrame({"level": [1.0, 2.0]}))


def test_varequ_table_dataframe_without_attributes(container):
    i = Set(container, "i", records=["a", "b"])
    j = Set(container, "j", records=["x", "y"])
    v = Variable(container, "v", domain=[i, j])

    v.setRecords(
        pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=["a", "b"], columns=["x", "y"]),
        uels_on_axes=True,
    )

    assert v.records["level"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert v.records["scale"].tolist() == [1.0] * 4


def test_varequ_table_dimension_mismatch(container):
    i = Set(container, "i", records=["a", "b"])

    with pytest.raises(ValueError, match="Dimensionality of table"):
        Variable(container, "v", domain=[i]).setRecords(
            pd.DataFrame([[1.0, 2.0]], index=["a"], columns=["x", "y"]),
            uels_on_axes=True,
        )


def test_varequ_series_on_scalar_with_attribute_index(container):
    v = Variable(container, "v")
    v.setRecords(pd.Series({"level": 3.0, "marginal": 1.0}))

    assert v.records["level"].tolist() == [3.0]
    assert v.records["marginal"].tolist() == [1.0]
    assert v.records["scale"].tolist() == [1.0]


def test_varequ_series_on_scalar(container):
    v = Variable(container, "v")
    v.setRecords(pd.Series([9.0]))

    assert v.toValue() == 9.0


def test_varequ_series_validations(container):
    with pytest.raises(ValueError, match="size exactly = 1"):
        Variable(container, "sc").setRecords(pd.Series([1.0, 2.0]))

    multi_attr = pd.MultiIndex.from_tuples([("level", "marginal")])
    with pytest.raises(ValueError, match="more than one level"):
        Variable(container, "mi").setRecords(pd.Series([1.0], index=multi_attr))

    i = Set(container, "i", records=["a"])
    j = Set(container, "j", records=["x"])
    with pytest.raises(ValueError, match="Dimensionality of table"):
        Variable(container, "dim", domain=[i, j]).setRecords(
            pd.Series([1.0], index=["a"]), uels_on_axes=True
        )


def test_varequ_series_without_attributes(container):
    i = Set(container, "i", records=["a", "b"])
    v = Variable(container, "v", domain=[i])
    v.setRecords(pd.Series([1.0, 2.0], index=["a", "b"]), uels_on_axes=True)

    assert v.records["level"].tolist() == [1.0, 2.0]


def test_varequ_attributes_in_more_than_one_index(container):
    v = Variable(container, "v", domain=["*"])
    multi = pd.MultiIndex.from_tuples([("level", "marginal"), ("lower", "upper")])

    with pytest.raises(ValueError, match="more than one index"):
        v.setRecords(
            pd.DataFrame(index=multi, columns=["level", "marginal"]),
            uels_on_axes=True,
        )


def test_varequ_from_dict_of_columns(container):
    i = Set(container, "i", records=["a", "b"])
    v = Variable(container, "v", domain=[i])

    # the domain column has to come first -- see the xfail below
    v.setRecords({"i": ["a", "b"], "level": [1.0, 2.0]})

    assert v.records["level"].tolist() == [1.0, 2.0]


def test_equation_ingestion(container):
    i = Set(container, "i", records=["a", "b"])
    e = Equation(container, "e", domain=[i])
    e.setRecords(pd.Series([1.0, 2.0], index=["a", "b"]), uels_on_axes=True)

    assert e.records["level"].tolist() == [1.0, 2.0]
    # equation defaults differ from variable defaults
    assert e.records["lower"].tolist() == [0.0, 0.0]


def test_varequ_table_dataframe_with_attributes_on_an_axis(container):
    i = Set(container, "i", records=["a", "b"])
    v = Variable(container, "v", domain=[i])

    v.setRecords(
        pd.DataFrame(
            [[1.0, 0.5], [2.0, 0.6]], index=["a", "b"], columns=["level", "marginal"]
        ),
        uels_on_axes=True,
    )

    assert v.records["level"].tolist() == [1.0, 2.0]
    assert v.records["marginal"].tolist() == [0.5, 0.6]


def test_varequ_series_with_attribute_level(container):
    i = Set(container, "i", records=["a", "b"])
    v = Variable(container, "v", domain=[i])
    index = pd.MultiIndex.from_tuples(
        [("a", "level"), ("a", "marginal"), ("b", "level"), ("b", "marginal")]
    )

    v.setRecords(pd.Series([1.0, 0.1, 2.0, 0.2], index=index), uels_on_axes=True)

    assert v.records["level"].tolist() == [1.0, 2.0]
    assert v.records["marginal"].tolist() == [0.1, 0.2]


def test_varequ_from_sequence_of_mappings(container):
    i = Set(container, "i", records=["a", "b"])
    v = Variable(container, "v", domain=[i])

    # a Sequence works as long as it yields named attribute columns
    v.setRecords([{"i": "a", "level": 1.0}, {"i": "b", "level": 2.0}])

    assert v.records["i"].tolist() == ["a", "b"]
    assert v.records["level"].tolist() == [1.0, 2.0]


def test_varequ_from_positional_sequence_is_rejected(container):
    i = Set(container, "i", records=["a", "b"])
    v = Variable(container, "v", domain=[i])

    with pytest.raises(ValueError, match="Attribute columns must be named"):
        v.setRecords([("a", 1.0), ("b", 2.0)])


def test_varequ_from_dict_of_columns_attribute_first(container):
    i = Set(container, "i", records=["a", "b"])
    v = Variable(container, "v", domain=[i])

    # same data as the test above, only the key order differs
    v.setRecords({"level": [1.0, 2.0], "i": ["a", "b"]})

    assert v.records["i"].tolist() == ["a", "b"]
    assert v.records["level"].tolist() == [1.0, 2.0]
