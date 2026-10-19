from __future__ import annotations

import math
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype

import gamspy._algebra.expression as expression
import gamspy._algebra.sparse as sparse
import gamspy.utils as utils
from gamspy._categoricals import assemble_categorical, rstrip_categories
from gamspy._internals import DataSource
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from gamspy._algebra.sparse import SparseAssignment
    from gamspy._symbols import Alias, Equation, Parameter, Set, Variable
    from gamspy._symbols.base import VarEquSymbol
    from gamspy._symbols.implicits import ImplicitParameter, ImplicitSet
    from gamspy._types import OperableType, ValidatedDomainType


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


def _level_codes_and_labels(axis: pd.Index, n: int) -> tuple[np.ndarray, pd.Index]:
    """Codes of every entry in level n of the axis and the labels they point at."""
    if isinstance(axis, pd.MultiIndex):
        return np.asarray(axis.codes[n]), axis.levels[n]

    return axis.factorize()


def _assert_axes_no_nans(records: pd.DataFrame | pd.Series) -> None:
    for axis_name, axis in zip(AXES, records.axes, strict=False):
        if isinstance(axis, pd.MultiIndex):
            for n in range(axis.nlevels):
                codes, labels = _level_codes_and_labels(axis, n)
                if (codes == -1).any() or (
                    labels.hasnans and labels.isna()[codes].any()
                ):
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


def _axis_to_codes(axis: pd.Index) -> list[tuple[np.ndarray, pd.CategoricalDtype]]:
    """
    The codes and the ordered categorical dtype of every level of the axis. Labels
    are converted to str and right-stripped. Categories follow the order of
    appearance unless the level is already categorical, in which case the user
    order is preserved.
    """
    levels = []
    for n in range(axis.nlevels):
        level = axis.levels[n] if isinstance(axis, pd.MultiIndex) else axis
        codes, labels = _level_codes_and_labels(axis, n)

        # Work on the distinct labels only: order them by first appearance (the
        # codes of a flat index already are) and str/rstrip them, then merge labels
        # that became equal after stripping.
        if isinstance(axis, pd.MultiIndex):
            codes, appearance = pd.factorize(codes)
            labels = labels[appearance]

        categories: np.ndarray | pd.Index
        if (
            isinstance(labels.dtype, pd.StringDtype)
            or labels.dtype.kind in "iu"
            or labels.dtype == np.float64
            or (labels.dtype == object and infer_dtype(labels) == "string")
        ):
            # Strings or numbers only: no need to convert each label by hand.
            remap, categories = rstrip_categories(labels)
            if remap is not None:
                codes = remap[codes]
        else:
            stripped = [str(x).rstrip() for x in labels]
            if len(set(stripped)) == len(stripped):
                categories = np.array(stripped, dtype=object)
            else:
                remap, categories = pd.factorize(np.array(stripped, dtype=object))
                codes = remap[codes]

        if isinstance(level.dtype, pd.CategoricalDtype):
            # preserve user order
            categorical = assemble_categorical(
                codes, categories, ordered=True
            ).reorder_categories(
                dict.fromkeys(map(str.rstrip, level.dtype.categories)),
                ordered=True,
            )
            levels.append((categorical.codes, categorical.dtype))
        else:
            levels.append((codes, pd.CategoricalDtype(categories, ordered=True)))

    return levels


def _axis_to_categoricals(axis: pd.Index) -> list[pd.Categorical]:
    return [
        pd.Categorical.from_codes(codes, dtype=dtype, validate=False)
        for codes, dtype in _axis_to_codes(axis)
    ]


def _flatten_columns(records: pd.DataFrame | pd.Series) -> list[Any]:
    """
    One row per entry of a Series, or per cell of a DataFrame in row-major order: a
    categorical column per level of the axes followed by the values.
    """
    if isinstance(records, pd.Series):
        # What is made of the values must not share data with the records of the
        # user: set_axis copies them, or lets Copy-on-Write track them.
        values = records.set_axis(pd.RangeIndex(len(records)))
        return [*_axis_to_categoricals(records.index), values]

    n_rows, n_columns = records.shape
    rows = np.repeat(np.arange(n_rows), n_columns)
    columns = np.tile(np.arange(n_columns), n_rows)

    flattened: list[Any] = [
        pd.Categorical.from_codes(codes[rows], dtype=dtype, validate=False)
        for codes, dtype in _axis_to_codes(records.index)
    ]
    flattened += [
        pd.Categorical.from_codes(codes[columns], dtype=dtype, validate=False)
        for codes, dtype in _axis_to_codes(records.columns)
    ]

    dtypes = set(records.dtypes)
    if len(dtypes) == 1 and isinstance(next(iter(dtypes)), np.dtype):
        values = records.to_numpy().reshape(-1)
        # A view of the records of the user, e.g. when they have a single column.
        flattened.append(values if values.flags.owndata else values.copy())
    else:
        # Let pandas find the common dtype of the columns.
        flattened.append(
            records.set_axis(range(n_rows), axis=0)
            .set_axis(range(n_columns), axis=1)
            .stack(future_stack=True)
            .array
        )

    return flattened


def _flatten_and_convert(records: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """The columns of `_flatten_columns` as a frame."""
    return pd.DataFrame(dict(enumerate(_flatten_columns(records))))


# Value dtypes that the attribute reshaping below handles itself.
_SPREADABLE_KINDS = "biufO"
_DUPLICATES_MESSAGE = "Index contains duplicate entries, cannot reshape"


def _rank_keys(
    labels: list[pd.Categorical],
) -> tuple[np.ndarray, np.ndarray, list[int]] | None:
    """
    Combines the codes of the label columns into one key per row and ranks the keys.
    Returns the distinct keys in ascending order None if the combined key would overflow.
    """
    sizes = [len(label.categories) for label in labels]
    n_keys = math.prod(sizes)
    if n_keys > np.iinfo(np.intp).max:
        return None

    key = np.ravel_multi_index([label.codes for label in labels], sizes)
    keys, rows = np.unique(key, return_inverse=True)
    return keys, rows, sizes


def _label_columns(
    keys: np.ndarray, sizes: list[int], labels: list[pd.Categorical]
) -> dict[int, Any]:
    """The label columns of the rows identified by the keys of `_rank_keys`."""
    return {
        n: pd.Categorical.from_codes(codes, dtype=label.dtype, validate=False)
        for n, (codes, label) in enumerate(
            zip(np.unravel_index(keys, sizes), labels, strict=True)
        )
    }


def _unstack_attributes(records: pd.DataFrame, position: int) -> pd.DataFrame:
    """
    Spreads the values of flattened records over one column per attribute, the attribute
    of each record being in column `position`.
    """
    labels = [records.iloc[:, n].array for n in range(records.shape[1] - 1)]
    attributes = labels.pop(position)
    values = records.iloc[:, -1]

    ranked = (
        _rank_keys(labels)
        if isinstance(values.dtype, np.dtype) and values.dtype.kind in _SPREADABLE_KINDS
        else None
    )
    if ranked is None:
        return (
            records.set_index(records.columns.tolist()[:-1])
            .unstack(position)
            .reset_index(drop=False)
        )

    keys, rows, sizes = ranked

    # Cells in column-major order, so that each attribute gets a contiguous column.
    n_attributes = len(attributes.categories)
    cells = attributes.codes.astype(np.intp) * keys.size + rows
    seen = np.zeros(keys.size * n_attributes, dtype=bool)
    seen[cells] = True
    if np.count_nonzero(seen) != cells.size:
        raise ValueError(_DUPLICATES_MESSAGE)

    spread = np.full(
        keys.size * n_attributes,
        np.nan,
        dtype=object if values.dtype.kind == "O" else float,
    )
    spread[cells] = values.to_numpy()

    data = _label_columns(keys, sizes, labels)
    for n in range(n_attributes):
        data[len(data)] = spread[n * keys.size : (n + 1) * keys.size]

    return pd.DataFrame(data)


def toValueParameter(symbol: Parameter) -> float:
    if not symbol.is_scalar:
        raise TypeError(
            "Cannot extract value data for non-scalar symbols "
            f"(symbol dimension is {symbol.dimension})"
        )

    if symbol.records is None:
        return 0.0

    return symbol.records["value"][0]


def toValueVariableEquation(symbol: VarEquSymbol, column: str | None) -> float:
    if not symbol.is_scalar:
        raise TypeError(
            "Cannot extract value data for non-scalar symbols "
            f"(symbol dimension is {symbol.dimension})"
        )

    column = symbol._validate_column(column)

    if symbol.records is None:
        raise ValidationError(
            f"Cannot call toValue on `{symbol.name}` because it has no records."
        )

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
    symbol: VarEquSymbol, columns: str | list[str] | None
) -> list:
    columns = symbol._validate_columns(columns)

    if symbol.records is None:
        return []

    return symbol.records.set_index(
        list(symbol.domain_labels + columns)
    ).index.to_list()


def _toDict_orient_chk(orient: str | None) -> DictFormat:
    if orient is None:
        orient = "natural"

    if orient.casefold() not in DICT_FORMAT:
        raise ValueError(
            f"Argument 'orient' expects one of the following (mixed-case OK): {list(DICT_FORMAT.keys())}"
        )

    return DICT_FORMAT[orient.casefold()]


def toDictParameter(symbol: Parameter, orient: str | None = None) -> dict:
    dict_format = _toDict_orient_chk(orient)

    if symbol.is_scalar:
        raise TypeError(
            f"Symbol `{symbol.name}` is a scalar and cannot be converted into a dict."
        )

    if symbol.records is None:
        return {}

    if dict_format is DictFormat.NATURAL:
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

    return symbol.records.to_dict()


def toDictVariableEquation(
    symbol: VarEquSymbol,
    columns: str | list[str] | None,
    orient: str | None = None,
) -> dict:
    dict_format = _toDict_orient_chk(orient)

    columns = symbol._validate_columns(columns)

    if symbol.is_scalar:
        raise TypeError(
            f"Symbol `{symbol.name}` is a scalar and cannot be converted into a dict."
        )

    if symbol.records is None:
        return {}

    if dict_format is DictFormat.NATURAL:
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

    return symbol.records[symbol.domain_labels + columns].to_dict()


def assign(
    owner: Set | Alias | Parameter | Variable | Equation,
    lhs: ImplicitSet | ImplicitParameter,
    rhs: OperableType | str | SparseAssignment,
    domain: ValidatedDomainType,
) -> None:
    owner._check_miro_protection()

    rhs, operator = sparse._unwrap(rhs)
    if isinstance(rhs, bool):
        rhs = "yes" if rhs else "no"

    statement = expression.Expression(lhs, operator, rhs)

    # Cannot validate definition if we are in a gp.Loop since the control indices can be provided by the gp.Loop
    if not owner.container._in_loop:
        statement._validate_definition(utils._unpack(domain))

    owner.container._add_statement(statement)
    owner._assignment = statement

    owner.container._synch_with_gams()
    owner._should_load_from = DataSource.GAMS
