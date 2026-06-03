from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from gamspy._special_values import SpecialValues
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from gamspy._symbols import Equation, Parameter, Set, Variable


def _validate_axes(
    symbol: Any, index: str | list | None, columns: str | list | None
) -> tuple[list, list]:
    """Helper function to validate and set default index and columns for pivoting."""
    if symbol.dimension < 2:
        raise ValidationError(
            "Pivoting operations only possible on symbols with dimension > 1, "
            f"symbol dimension is {symbol.dimension}"
        )

    if not isinstance(index, (str, list, type(None))):
        raise TypeError("Argument 'index' must be type str, list or NoneType")

    if index is None:
        index = symbol.records.columns[: symbol.dimension - 1].tolist()
    elif isinstance(index, str):
        index = [index]

    if not isinstance(columns, (str, list, type(None))):
        raise TypeError("Argument 'columns' must be type str, list, or NoneType")

    if columns is None:
        columns = symbol.records.columns[
            symbol.dimension - 1 : symbol.dimension
        ].tolist()
    elif isinstance(columns, str):
        columns = [columns]

    if set(index + columns) != set(symbol.domain_labels):
        raise ValidationError(
            "Must specify all domain_labels to pivot in either 'index' or 'columns' "
            f"arguments, user did not specify: {set(symbol.domain_labels) - set(index + columns)}"
        )

    return index, columns


def _fill_missing_values(df: pd.DataFrame, fill_value: Any) -> None:
    """Helper function to fill NA values safely across different Pandas versions."""
    major, minor, *_ = pd.__version__.split(".")
    if (int(major), int(minor)) >= (2, 2) and (int(major), int(minor)) < (3, 0):
        with pd.option_context("future.no_silent_downcasting", True):
            df.fillna(fill_value, inplace=True)
    else:
        df.fillna(fill_value, inplace=True)


def _restore_special_values(
    symbol: Parameter | Variable | Equation,
    df: pd.DataFrame,
    index: list,
    columns: list,
    value_col: str,
    stored_columns: Any,
) -> pd.DataFrame:
    """Helper function to find and restore special values (NA, UNDEF) to the pivoted dataframe."""
    specnans = symbol.findSpecialValues(
        [SpecialValues.UNDEF, SpecialValues.NA], column=value_col
    )

    if specnans is None or specnans.empty:
        return df

    idx = list(zip(*[specnans[i] for i in index]))
    col = list(zip(*[specnans[i] for i in columns]))

    # Use numpy array to prevent pandas from silently dropping the NaN payload
    arr = df.to_numpy(copy=True)

    for n, (i, c) in zip(specnans.index, zip(idx, col)):
        i_loc = i[0] if len(i) == 1 else i
        c_loc = c[0] if len(c) == 1 else c

        row_idx = df.index.get_loc(i_loc)
        col_idx = df.columns.get_loc(c_loc)

        arr[row_idx, col_idx] = specnans.loc[n, value_col]

    df = pd.DataFrame(arr, index=df.index, columns=stored_columns)
    return df


def pivot_set(
    symbol: Set,
    index: str | list | None = None,
    columns: str | list | None = None,
    fill_value: int | float | str | None = None,
) -> pd.DataFrame | None:
    if symbol.records is None:
        return None

    index, columns = _validate_axes(symbol, index, columns)
    value = "value"
    fill_value = False if fill_value is None else fill_value

    df = symbol.records.copy()
    if "element_text" in df.columns:
        df.drop(columns=["element_text"], inplace=True)

    df.insert(symbol.dimension, value, True)
    df = df.pivot(index=index, columns=columns, values=value)

    _fill_missing_values(df, fill_value)

    df = df.astype(bool) if isinstance(fill_value, bool) else df.infer_objects()
    df.index.names = [None] * len(index)
    df.columns.names = [None] * len(columns)

    return df


def pivot_parameter(
    symbol: Parameter,
    index: str | list | None = None,
    columns: str | list | None = None,
    fill_value: int | float | str | None = None,
) -> pd.DataFrame | None:
    if symbol.records is None:
        return None

    index, columns = _validate_axes(symbol, index, columns)
    value = "value"
    fill_value = 0.0 if fill_value is None else fill_value

    df = symbol.records[symbol.domain_labels + [value]]
    has_nans = df[value].isna().any()

    df = df.pivot(index=index, columns=columns, values=value)
    stored_columns = df.columns

    _fill_missing_values(df, fill_value)

    df = df.astype(float) if isinstance(fill_value, float) else df.infer_objects()

    if has_nans:
        df = _restore_special_values(symbol, df, index, columns, value, stored_columns)

    df.index.names = [None] * len(index)
    df.columns.names = [None] * len(columns)

    return df


def pivot_variable(
    symbol: Variable | Equation,
    index: str | list | None = None,
    columns: str | list | None = None,
    value: str | None = None,
    fill_value: int | float | str | None = None,
) -> pd.DataFrame | None:
    if symbol.records is None:
        return None

    index, columns = _validate_axes(symbol, index, columns)

    if not isinstance(value, (str, type(None))):
        raise TypeError("Argument 'value' must be type str or NoneType")

    value = "level" if value is None else value

    if value not in symbol._attributes:
        raise TypeError(
            f"Argument 'value' must be one of the following symbol attributes: {symbol._attributes}"
        )

    fill_value = 0.0 if fill_value is None else fill_value

    df = symbol.records[symbol.domain_labels + [value]]
    has_nans = df[value].isna().any()

    df = df.pivot(index=index, columns=columns, values=value)
    stored_columns = df.columns

    _fill_missing_values(df, fill_value)

    df = df.astype(float) if isinstance(fill_value, float) else df.infer_objects()

    if has_nans:
        df = _restore_special_values(symbol, df, index, columns, value, stored_columns)

    df.index.names = [None] * len(index)
    df.columns.names = [None] * len(columns)

    return df


pivot_equation = pivot_variable
