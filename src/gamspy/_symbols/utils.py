from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from gamspy._symbols import Equation, Parameter, Set, Variable


class DictFormat(Enum):
    UNKNOWN = -1
    COLUMNS = 0
    NATURAL = 1

    @classmethod
    def _missing_(cls, value):
        return DictFormat.UNKNOWN


DICT_FORMAT = {
    "natural": DictFormat.NATURAL,
    "columns": DictFormat.COLUMNS,
}
AXES = ["index", "columns"]


def _get_implied_dimension_from_axes(records):
    return sum([axis.nlevels for axis in records.axes])


def _assert_axes_no_nans(records):
    for axis_name, axis in zip(AXES, records.axes, strict=False):
        if isinstance(axis, pd.MultiIndex):
            axis_as_frame = pd.DataFrame(np.array(axis.tolist(), dtype=object))
            for n in range(axis.nlevels):
                if axis_as_frame.iloc[:, n].hasnans:
                    raise ValidationError(
                        "Tabular 'records' cannot have missing index information "
                        f"(i.e., NaNs detected in 'records.{axis_name} level_index={n}')"
                    )
        else:
            if axis.hasnans:
                raise ValidationError(
                    "Tabular 'records' cannot have missing index information "
                    f"(i.e., NaNs detected in 'records.{axis_name}')"
                )


def _flatten_and_convert(records):
    AXES = ["index", "columns"]
    drop_needed = False
    for axis_name, axis in zip(AXES, records.axes, strict=False):
        idx = pd.DataFrame(columns=list(range(axis.nlevels)))

        # go through axis
        axis_as_frame = pd.DataFrame(np.array(axis.tolist(), dtype=object))
        for n in range(axis.nlevels):
            level = axis.levels[n] if hasattr(axis, "levels") else axis

            # factorize
            # preserve order of appearance, not lexicographical order + str/rstrip
            codes, cats = pd.Series(
                map(str.rstrip, map(str, axis_as_frame.iloc[:, n]))
            ).factorize()

            # create categorical
            categorical = pd.Categorical.from_codes(
                codes, categories=cats, ordered=True
            )

            # preserve user order if CategoricalDtype
            if isinstance(level.dtype, pd.CategoricalDtype):
                categorical = categorical.reorder_categories(
                    dict.fromkeys(map(str.rstrip, level.categories)),
                    ordered=True,
                )

            # set
            idx.isetitem(n, categorical)

        # TODO: need a workaround here to avoid an error with pandas stack
        if (
            axis_name == "columns"
            and idx.set_index(list(idx.columns)).index.has_duplicates
        ):
            drop_needed = True
            idx["__ai"] = range(len(idx))

        # create categorical index and set on records
        setattr(records, axis_name, idx.set_index(list(idx.columns)).index)

        # remove names
        getattr(records, axis_name).names = [None] * getattr(records, axis_name).nlevels

    if isinstance(records, pd.DataFrame):
        major, minor, *_ = pd.__version__.split(".")
        major, minor = (int(major), int(minor))

        # TODO: remove in future... allows support for pandas < 2.1.0
        if (major, minor) >= (2, 2):
            to_drop = (records.index.nlevels - 1) + (records.columns.nlevels)
            records = records.stack(
                list(range(records.columns.nlevels)),
                future_stack=True,
            ).reset_index(drop=False)
        else:
            to_drop = (records.index.nlevels - 1) + (records.columns.nlevels)
            records = records.stack(
                list(range(records.columns.nlevels)), dropna=False
            ).reset_index(drop=False)

        if drop_needed:
            records.drop(columns=records.columns[to_drop], inplace=True)
    else:
        records = records.reset_index(drop=False)

    return records


def toValueParameter(symbol: Parameter) -> float | None:
    if not symbol.is_scalar:
        raise TypeError(
            "Cannot extract value data for non-scalar symbols "
            f"(symbol dimension is {symbol.dimension})"
        )

    if symbol.records is None:
        return None

    return symbol.records["value"][0]


def toValueVariableEquation(
    symbol: Variable | Equation, column: str | None
) -> float | None:
    if not symbol.is_scalar:
        raise TypeError(
            "Cannot extract value data for non-scalar symbols "
            f"(symbol dimension is {symbol.dimension})"
        )

    if column is None:
        column = "level"

    if not isinstance(column, str):
        raise TypeError(
            f"Argument 'column' must be type str. User passed {type(column)}."
        )

    if column not in symbol._attributes:
        raise TypeError(
            f"Argument 'column' must be one of the following: {symbol._attributes}"
        )

    if symbol.records is None:
        return None

    return symbol.records[column][0]


def toListSet(symbol: Set, *, include_element_text: bool = False) -> list:
    if symbol.records is None:
        return []

    if include_element_text:
        return symbol.records.set_index(
            symbol.records.columns.to_list()
        ).index.to_list()

    return symbol.records.set_index(
        symbol.records.columns.to_list()[: symbol.dimension]
    ).index.to_list()


def toListParameter(symbol: Parameter) -> list:
    if symbol.records is None:
        return []

    if symbol.is_scalar:
        return list(symbol.records.iloc[:, 0])

    return list(
        zip(
            *[symbol.records.iloc[:, x] for x in range(len(symbol.records.columns))],
            strict=True,
        )
    )


def toListVariableEquation(
    symbol: Variable | Equation, columns: str | list[str] | None
) -> list:
    if columns is None:
        columns = "level"

    if not isinstance(columns, (str, list)):
        raise TypeError(
            f"Argument 'columns' must be type str or list. User passed {type(columns)}."
        )

    if isinstance(columns, str):
        columns = [columns]

    if any(not isinstance(i, str) for i in columns):
        raise TypeError("Argument 'columns' must contain only type str.")

    if any(i not in symbol._attributes for i in columns):
        raise TypeError(
            f"Argument 'columns' must be a subset of the following: {symbol._attributes}"
        )

    if symbol.records is None:
        return []

    return symbol.records.set_index(
        list(symbol.domain_labels + columns)
    ).index.to_list()


def _toDict_orient_chk(orient):
    if orient is None:
        orient = "natural"

    if orient.casefold() not in DICT_FORMAT:
        raise ValueError(
            f"Argument 'orient' expects one of the following (mixed-case OK): {list(DICT_FORMAT.keys())}"
        )

    return DICT_FORMAT[orient.casefold()]


def toDictParameter(symbol: Parameter, orient=None):
    # check and/or set orient
    orient = _toDict_orient_chk(orient)

    if symbol.is_scalar:
        raise TypeError(
            f"Symbol `{symbol.name}` is a scalar and cannot be converted into a dict."
        )

    if symbol.records is not None:
        if orient is DictFormat.NATURAL:
            if symbol.dimension == 1:
                return dict(
                    zip(
                        symbol.records.iloc[:, 0],
                        symbol.records.iloc[:, -1],
                        strict=True,
                    )
                )

            else:
                doms = zip(
                    *[symbol.records.iloc[:, i] for i in range(symbol.dimension)],
                    strict=True,
                )
                vals = symbol.records[symbol.records.columns[-1]]

                return dict(zip(doms, vals, strict=True))

        if orient is DictFormat.COLUMNS:
            return symbol.records.to_dict()


def toDictVariableEquation(
    symbol: Variable | Equation, columns: str | list[str] | None, orient=None
):
    # check and/or set orient
    orient = _toDict_orient_chk(orient)

    if columns is None:
        columns = "level"

    if not isinstance(columns, (str, list)):
        raise TypeError(
            f"Argument 'columns' must be type str or list. User passed {type(columns)}."
        )

    if isinstance(columns, str):
        columns = [columns]

    if any(not isinstance(i, str) for i in columns):
        raise TypeError("Argument 'columns' must contain only type str.")

    if any(i not in symbol._attributes for i in columns):
        raise TypeError(
            f"Argument 'columns' must be a subset of the following: {symbol._attributes}"
        )

    if symbol.is_scalar:
        raise TypeError(
            f"Symbol `{symbol.name}` is a scalar and cannot be converted into a dict."
        )

    if symbol.records is not None:
        if orient is DictFormat.NATURAL:
            if symbol.dimension == 1:
                doms = symbol.records.iloc[:, 0]
            else:
                doms = zip(
                    *[symbol.records.iloc[:, i] for i in range(symbol.dimension)],
                    strict=True,
                )

            if len(columns) == 1:
                return dict(zip(doms, symbol.records[columns[0]], strict=True))
            else:
                vals = symbol.records[columns].to_dict("records")

                return dict(zip(doms, vals, strict=True))

        if orient is DictFormat.COLUMNS:
            return symbol.records[symbol.domain_labels + columns].to_dict()
