from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gamspy import (
    Container,
    Equation,
    Parameter,
    Set,
    SpecialValues,
    Variable,
)
from gamspy.exceptions import GdxException

pytestmark = pytest.mark.unit


@pytest.fixture
def m():
    container = Container()
    yield container
    container.close()


def test_object_domain_labels_are_rstripped(m):
    p = Parameter(m, "p", domain=["*"])
    p.setRecords(pd.DataFrame({"d": ["a ", "b"], "v": [1.0, 2.0]}))

    assert p.records["d"].tolist() == ["a", "b"]


def test_object_domain_labels_collapse_after_rstrip(m):
    # "a " and "a" are the same GAMS UEL; the pair stays distinct only because
    # the second dimension differs
    p = Parameter(m, "p", domain=["*", "*"])
    p.setRecords(pd.DataFrame({"d0": ["a ", "a"], "d1": ["x", "y"], "v": [1.0, 2.0]}))

    assert p.records["d0"].tolist() == ["a", "a"]
    assert p.toList() == [("a", "x", 1.0), ("a", "y", 2.0)]


def test_categorical_domain_categories_are_rstripped(m):
    p = Parameter(m, "p", domain=["*"])
    p.setRecords(pd.DataFrame({"d": pd.Categorical(["a ", "b"]), "v": [1.0, 2.0]}))

    assert p.records["d"].cat.categories.tolist() == ["a", "b"]
    assert p.records["d"].tolist() == ["a", "b"]


def test_categorical_domain_categories_collapse_after_rstrip(m):
    # the unused "a " category collapses onto "a", so the category list shrinks
    # and the column has to be re-encoded
    p = Parameter(m, "p", domain=["*"])
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


def test_rstrip_collapse_that_creates_duplicates_is_rejected(m):
    # Labels differing only by trailing whitespace collapse to one UEL. Here that
    # makes two identical keys, which GAMS rejects when the records are written.
    p = Parameter(m, "p", domain=["*"])
    with pytest.raises(GdxException, match="data errors"):
        p.setRecords(pd.DataFrame({"d": ["a ", "a"], "v": [1.0, 2.0]}))


def test_set_domain_labels_are_rstripped(m):
    s = Set(m, "s", domain=["*", "*"])
    s.setRecords(pd.DataFrame({"d0": ["a ", "a"], "d1": ["x", "y"]}))

    assert s.toList() == [("a", "x"), ("a", "y")]


def test_string_special_values_are_remapped(m):
    p = Parameter(m, "p", domain=["*"])
    p.setRecords(
        pd.DataFrame({"d": ["e", "n", "u", "w"], "v": ["eps", "NA", "Undef", "undf"]})
    )

    values = p.records["v"] if "v" in p.records else p.records.iloc[:, -1]
    assert SpecialValues.isEps(values[0])
    assert SpecialValues.isNA(values[1])
    assert SpecialValues.isUndef(values[2])
    assert SpecialValues.isUndef(values[3])


def test_zero_records_are_filtered_but_eps_is_kept(m):
    i = Set(m, "i", records=["a", "b", "c"])
    p = Parameter(m, "p", domain=[i])
    p.setRecords(np.array([0.0, 3.0, SpecialValues.EPS]))

    labels = [row[0] for row in p.toList()]
    assert labels == ["b", "c"]


def test_parameter_scalar_value_on_non_scalar(m):
    i = Set(m, "i", records=["a"])
    p = Parameter(m, "p", domain=[i])

    with pytest.raises(ValueError, match="cannot be set with a scalar value"):
        m.setRecords({p: 5})


def test_parameter_scalar_value_through_container(m):
    p = Parameter(m, "p")
    m.setRecords({p: 5})

    assert p.toValue() == 5.0


def test_parameter_ndarray_row_and_column_vectors(m):
    i = Set(m, "i", records=["a", "b"])

    row = Parameter(m, "row", domain=[i])
    row.setRecords(np.array([[1.0, 2.0]]))
    assert row.toList() == [("a", 1.0), ("b", 2.0)]

    col = Parameter(m, "col", domain=[i])
    col.setRecords(np.array([[1.0], [2.0]]))
    assert col.toList() == [("a", 1.0), ("b", 2.0)]


def test_parameter_scalar_ndarray(m):
    p = Parameter(m, "p")
    p.setRecords(np.array(5.0))

    assert p.toValue() == 5.0


def test_parameter_ndarray_validations(m):
    i = Set(m, "i", records=["a", "b"])
    j = Set(m, "j", records=["x", "y"])

    with pytest.raises(TypeError, match="Attempted conversion to numpy array"):
        Parameter(m, "bad", domain=[i]).setRecords(np.array([object()]))

    with pytest.raises(ValueError, match="Dimension mismatch"):
        Parameter(m, "ndim", domain=[i, j]).setRecords(np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="requires a 'regular' domain type"):
        Parameter(m, "relaxed", domain=["*"]).setRecords(np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="Shape mismatch"):
        Parameter(m, "shape", domain=[i]).setRecords(np.array([1.0, 2.0, 3.0]))


def test_parameter_flat_dataframe_validations(m):
    i = Set(m, "i", records=["a", "b"])

    with pytest.raises(ValueError, match="Dimensionality of records"):
        Parameter(m, "dim", domain=[i]).setRecords(
            pd.DataFrame({"d": ["a"], "x": [1.0], "y": [2.0]})
        )

    with pytest.raises(ValueError, match="multiple records for a scalar"):
        Parameter(m, "sc").setRecords(pd.DataFrame({"v": [1.0, 2.0]}))


def test_parameter_from_table_dataframe(m):
    i = Set(m, "i", records=["a", "b"])
    j = Set(m, "j", records=["x", "y"])
    p = Parameter(m, "p", domain=[i, j])

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


def test_parameter_table_dataframe_dimension_mismatch(m):
    i = Set(m, "i", records=["a"])

    # a DataFrame table implies two dimensions, the parameter only has one
    with pytest.raises(ValueError, match="Dimensionality of table"):
        Parameter(m, "p", domain=[i]).setRecords(
            pd.DataFrame([[1.0]], index=["a"], columns=["x"]),
            uels_on_axes=True,
        )


def test_parameter_from_series(m):
    scalar = Parameter(m, "scalar_p")
    scalar.setRecords(pd.Series([4.0]))
    assert scalar.toValue() == 4.0

    i = Set(m, "i", records=["a", "b"])
    p = Parameter(m, "p", domain=[i])
    p.setRecords(pd.Series([1.0, 2.0], index=["a", "b"]), uels_on_axes=True)
    assert p.toList() == [("a", 1.0), ("b", 2.0)]


def test_parameter_series_validations(m):
    i = Set(m, "i", records=["a", "b"])
    j = Set(m, "j", records=["x"])

    with pytest.raises(ValueError, match="size exactly = 1"):
        Parameter(m, "sc").setRecords(pd.Series([1.0, 2.0]))

    with pytest.raises(ValueError, match="Dimensionality of data"):
        Parameter(m, "p", domain=[i, j]).setRecords(pd.Series([1.0], index=["a"]))


def test_parameter_from_dict(m):
    p = Parameter(m, "p", domain=["*"])
    p.setRecords({"d": ["a"], "v": [1.0]})

    assert p.toList() == [("a", 1.0)]


def test_parameter_from_empty_iterable(m):
    p = Parameter(m, "p", domain=["*"])
    p.setRecords([])

    assert p.toList() == []


def test_parameter_sequence_dimension_mismatch(m):
    i = Set(m, "i", records=["a"])

    with pytest.raises(ValueError, match="Dimensionality of records"):
        Parameter(m, "p", domain=[i]).setRecords([("a", 1.0, 2.0)])


def test_parameter_unconvertible_records(m):
    p = Parameter(m, "p", domain=["*"])

    with pytest.raises(TypeError, match="Could not convert to pandas DataFrame"):
        p.setRecords(object())


def test_set_cannot_be_initialized_with_a_number(m):
    with pytest.raises(TypeError, match="Sets cannot be initialized"):
        Set(m, "s").setRecords(5)


def test_set_from_ndarray(m):
    s = Set(m, "s")
    s.setRecords(np.array(["a", "b"]))

    assert s.toList() == ["a", "b"]


def test_set_from_bool_table(m):
    i = Set(m, "i", records=["a", "b"])
    j = Set(m, "j", records=["x", "y"])
    s = Set(m, "s", domain=[i, j])

    s.setRecords(
        pd.DataFrame(
            [[True, False], [False, True]], index=["a", "b"], columns=["x", "y"]
        ),
        uels_on_axes=True,
    )
    assert s.toList() == [("a", "x"), ("b", "y")]


def test_set_from_object_table_is_converted(m):
    i = Set(m, "i", records=["a", "b"])
    j = Set(m, "j", records=["x", "y"])
    s = Set(m, "s", domain=[i, j])

    # object dtype that convert_dtypes() can turn into bool
    s.setRecords(
        pd.DataFrame(
            [[True, False], [False, True]], index=["a", "b"], columns=["x", "y"]
        ).astype(object),
        uels_on_axes=True,
    )
    assert s.toList() == [("a", "x"), ("b", "y")]


def test_set_table_validations(m):
    i = Set(m, "i", records=["a", "b"])
    j = Set(m, "j", records=["x", "y"])
    bool_df = pd.DataFrame(
        [[True, False], [False, True]], index=["a", "b"], columns=["x", "y"]
    )

    with pytest.raises(TypeError, match="must be type bool"):
        Set(m, "s1", domain=[i, j]).setRecords(
            pd.DataFrame([[1, 0], [0, 1]], index=["a", "b"], columns=["x", "y"]),
            uels_on_axes=True,
        )

    with pytest.raises(ValueError, match="Dimensionality of table"):
        Set(m, "s2", domain=[i]).setRecords(bool_df, uels_on_axes=True)


def test_set_from_series(m):
    plain = Set(m, "plain")
    plain.setRecords(pd.Series(["a", "b"]))
    assert plain.toList() == ["a", "b"]

    i = Set(m, "i", records=["a", "b"])
    on_axes = Set(m, "on_axes", domain=[i])
    on_axes.setRecords(pd.Series([True, False], index=["a", "b"]), uels_on_axes=True)

    assert on_axes.toList() == ["a", "b"]
    assert on_axes.records["element_text"].tolist() == ["True", "False"]


def test_set_series_dimension_mismatch(m):
    i = Set(m, "i", records=["a"])
    j = Set(m, "j", records=["x"])

    with pytest.raises(ValueError, match="Dimensionality of data"):
        Set(m, "s", domain=[i, j]).setRecords(pd.Series(["a"]))

    # same mismatch, taken through the uels_on_axes branch
    with pytest.raises(ValueError, match="Dimensionality of data"):
        Set(m, "s2", domain=[i, j]).setRecords(
            pd.Series([True], index=["a"]), uels_on_axes=True
        )


def test_set_from_dict(m):
    s = Set(m, "s")
    s.setRecords({0: ["a", "b"]})

    assert s.toList() == ["a", "b"]


def test_set_from_string(m):
    s = Set(m, "s")
    s.setRecords("only")

    assert s.toList() == ["only"]


def test_set_from_empty_iterable(m):
    s = Set(m, "s")
    s.setRecords([])

    assert s.toList() == []


def test_set_dimensionality_mismatch(m):
    i = Set(m, "i", records=["a"])

    with pytest.raises(ValueError, match="Dimensionality of records"):
        Set(m, "s", domain=[i]).setRecords([("a", "text", "extra")])


def test_set_unconvertible_records(m):
    with pytest.raises(TypeError, match="Could not convert to pandas DataFrame"):
        Set(m, "s").setRecords(object())


def test_varequ_scalar_value_on_non_scalar(m):
    i = Set(m, "i", records=["a"])

    with pytest.raises(ValueError, match="symbol is not scalar"):
        Variable(m, "v", domain=[i]).setRecords(3.0)


def test_varequ_scalar_value(m):
    v = Variable(m, "v")
    v.setRecords(2.5)

    assert v.toValue() == 2.5


def test_varequ_from_ndarray(m):
    i = Set(m, "i", records=["a", "b"])
    v = Variable(m, "v", domain=[i])
    v.setRecords(np.array([1.0, 2.0]))

    assert v.records["level"].tolist() == [1.0, 2.0]
    # unspecified attributes fall back to the variable defaults
    assert v.records["lower"].tolist() == [float("-inf")] * 2


def test_varequ_dict_of_arrays(m):
    i = Set(m, "i", records=["a", "b"])
    v = Variable(m, "v", domain=[i])
    v.setRecords({"level": np.array([1.0, 2.0]), "marginal": np.array([0.1, 0.2])})

    assert v.records["level"].tolist() == [1.0, 2.0]
    assert v.records["marginal"].tolist() == [0.1, 0.2]


def test_varequ_dict_of_arrays_validations(m):
    i = Set(m, "i", records=["a", "b"])
    j = Set(m, "j", records=["x"])

    with pytest.raises(ValueError, match="Dimensionality mismatch"):
        Variable(m, "ndim", domain=[i, j]).setRecords({"level": np.array([1.0, 2.0])})

    with pytest.raises(ValueError, match="do not have the same shape"):
        Variable(m, "shapes", domain=[i]).setRecords(
            {"level": np.array([1.0, 2.0]), "marginal": np.array([1.0])}
        )

    with pytest.raises(ValueError, match="requires a 'regular' domain type"):
        Variable(m, "relaxed", domain=["*"]).setRecords({"level": np.array([1.0, 2.0])})

    with pytest.raises(ValueError, match="Shape mismatch"):
        Variable(m, "shape", domain=[i]).setRecords(
            {"level": np.array([1.0, 2.0, 3.0])}
        )


def test_varequ_dict_of_arrays_row_and_column_vectors(m):
    i = Set(m, "i", records=["a", "b"])

    row = Variable(m, "row", domain=[i])
    row.setRecords({"level": np.array([[1.0, 2.0]])})
    assert row.records["level"].tolist() == [1.0, 2.0]

    col = Variable(m, "col", domain=[i])
    col.setRecords({"level": np.array([[1.0], [2.0]])})
    assert col.records["level"].tolist() == [1.0, 2.0]


def test_varequ_unconvertible_records(m):
    i = Set(m, "i", records=["a"])

    with pytest.raises(TypeError, match="Could not convert to pandas DataFrame"):
        Variable(m, "v", domain=[i]).setRecords(object())


def test_varequ_scalar_dict_of_arrays(m):
    v = Variable(m, "v")
    v.setRecords({"level": 3.0, "marginal": 0.5})

    assert v.records["level"].tolist() == [3.0]
    assert v.records["marginal"].tolist() == [0.5]


def test_varequ_flat_dataframe_validations(m):
    i = Set(m, "i", records=["a", "b"])

    with pytest.raises(ValueError, match="Dimensionality of records"):
        Variable(m, "dim", domain=[i]).setRecords(
            pd.DataFrame({"i": ["a"], "level": [1.0], "extra": [2.0]})
        )

    with pytest.raises(ValueError, match="multiple records for a scalar"):
        Variable(m, "sc").setRecords(pd.DataFrame({"level": [1.0, 2.0]}))


def test_varequ_table_dataframe_without_attributes(m):
    i = Set(m, "i", records=["a", "b"])
    j = Set(m, "j", records=["x", "y"])
    v = Variable(m, "v", domain=[i, j])

    v.setRecords(
        pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=["a", "b"], columns=["x", "y"]),
        uels_on_axes=True,
    )

    assert v.records["level"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert v.records["scale"].tolist() == [1.0] * 4


def test_varequ_table_dimension_mismatch(m):
    i = Set(m, "i", records=["a", "b"])

    with pytest.raises(ValueError, match="Dimensionality of table"):
        Variable(m, "v", domain=[i]).setRecords(
            pd.DataFrame([[1.0, 2.0]], index=["a"], columns=["x", "y"]),
            uels_on_axes=True,
        )


def test_varequ_series_on_scalar_with_attribute_index(m):
    v = Variable(m, "v")
    v.setRecords(pd.Series({"level": 3.0, "marginal": 1.0}))

    assert v.records["level"].tolist() == [3.0]
    assert v.records["marginal"].tolist() == [1.0]
    assert v.records["scale"].tolist() == [1.0]


def test_varequ_series_on_scalar(m):
    v = Variable(m, "v")
    v.setRecords(pd.Series([9.0]))

    assert v.toValue() == 9.0


def test_varequ_series_validations(m):
    with pytest.raises(ValueError, match="size exactly = 1"):
        Variable(m, "sc").setRecords(pd.Series([1.0, 2.0]))

    multi_attr = pd.MultiIndex.from_tuples([("level", "marginal")])
    with pytest.raises(ValueError, match="more than one level"):
        Variable(m, "mi").setRecords(pd.Series([1.0], index=multi_attr))

    i = Set(m, "i", records=["a"])
    j = Set(m, "j", records=["x"])
    with pytest.raises(ValueError, match="Dimensionality of table"):
        Variable(m, "dim", domain=[i, j]).setRecords(
            pd.Series([1.0], index=["a"]), uels_on_axes=True
        )


def test_varequ_series_without_attributes(m):
    i = Set(m, "i", records=["a", "b"])
    v = Variable(m, "v", domain=[i])
    v.setRecords(pd.Series([1.0, 2.0], index=["a", "b"]), uels_on_axes=True)

    assert v.records["level"].tolist() == [1.0, 2.0]


def test_varequ_attributes_in_more_than_one_index(m):
    v = Variable(m, "v", domain=["*"])
    multi = pd.MultiIndex.from_tuples([("level", "marginal"), ("lower", "upper")])

    with pytest.raises(ValueError, match="more than one index"):
        v.setRecords(
            pd.DataFrame(index=multi, columns=["level", "marginal"]),
            uels_on_axes=True,
        )


def test_varequ_from_dict_of_columns(m):
    i = Set(m, "i", records=["a", "b"])
    v = Variable(m, "v", domain=[i])

    # the domain column has to come first -- see the xfail below
    v.setRecords({"i": ["a", "b"], "level": [1.0, 2.0]})

    assert v.records["level"].tolist() == [1.0, 2.0]


def test_equation_ingestion(m):
    i = Set(m, "i", records=["a", "b"])
    e = Equation(m, "e", domain=[i])
    e.setRecords(pd.Series([1.0, 2.0], index=["a", "b"]), uels_on_axes=True)

    assert e.records["level"].tolist() == [1.0, 2.0]
    # equation defaults differ from variable defaults
    assert e.records["lower"].tolist() == [0.0, 0.0]


def test_varequ_table_dataframe_with_attributes_on_an_axis(m):
    i = Set(m, "i", records=["a", "b"])
    v = Variable(m, "v", domain=[i])

    v.setRecords(
        pd.DataFrame(
            [[1.0, 0.5], [2.0, 0.6]], index=["a", "b"], columns=["level", "marginal"]
        ),
        uels_on_axes=True,
    )

    assert v.records["level"].tolist() == [1.0, 2.0]
    assert v.records["marginal"].tolist() == [0.5, 0.6]


def test_varequ_series_with_attribute_level(m):
    i = Set(m, "i", records=["a", "b"])
    v = Variable(m, "v", domain=[i])
    index = pd.MultiIndex.from_tuples(
        [("a", "level"), ("a", "marginal"), ("b", "level"), ("b", "marginal")]
    )

    v.setRecords(pd.Series([1.0, 0.1, 2.0, 0.2], index=index), uels_on_axes=True)

    assert v.records["level"].tolist() == [1.0, 2.0]
    assert v.records["marginal"].tolist() == [0.1, 0.2]


def test_varequ_from_sequence_of_mappings(m):
    i = Set(m, "i", records=["a", "b"])
    v = Variable(m, "v", domain=[i])

    # a Sequence works as long as it yields named attribute columns
    v.setRecords([{"i": "a", "level": 1.0}, {"i": "b", "level": 2.0}])

    assert v.records["i"].tolist() == ["a", "b"]
    assert v.records["level"].tolist() == [1.0, 2.0]


def test_varequ_from_positional_sequence_is_rejected(m):
    i = Set(m, "i", records=["a", "b"])
    v = Variable(m, "v", domain=[i])

    with pytest.raises(ValueError, match="Attribute columns must be named"):
        v.setRecords([("a", 1.0), ("b", 2.0)])


def test_varequ_from_dict_of_columns_attribute_first(m):
    i = Set(m, "i", records=["a", "b"])
    v = Variable(m, "v", domain=[i])

    # same data as the test above, only the key order differs
    v.setRecords({"level": [1.0, 2.0], "i": ["a", "b"]})

    assert v.records["i"].tolist() == ["a", "b"]
    assert v.records["level"].tolist() == [1.0, 2.0]
