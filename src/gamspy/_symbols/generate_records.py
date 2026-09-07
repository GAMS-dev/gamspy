from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from gamspy._algorithms import (
    cartesian_product,
    choice_no_replace,
    used_category_positions,
)
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from collections.abc import Callable

    from gamspy._symbols import Alias, Equation, Parameter, Set, UniverseAlias, Variable


def _validate_density_and_domain(
    symbol: Set | Parameter | Variable | Equation, density: int | float | list | None
) -> tuple[int | float | list, bool]:
    """Helper function to validate density, check the domain, and determine if records should be empty."""
    if not isinstance(density, (int, float, list, type(None))):
        raise TypeError("Argument 'density' must be type int, float, list or NoneType")

    if density is None:
        density = 1.0

    if isinstance(density, list):
        if len(density) != symbol.dimension:
            raise ValueError(
                f"Argument 'density' must be of length <symbol>.dimension ({len(density)} != {symbol.dimension})"
            )

        for dense in density:
            if not isinstance(dense, (int, float)):
                raise TypeError(
                    "Argument 'density' must contain only type int or float"
                )

            if not (0 <= dense <= 1):
                raise ValueError(
                    "Argument 'density' must contain values on the interval [0,1]."
                )
    elif not (0 <= density <= 1):
        raise ValueError("Argument 'density' must be a value on the interval [0,1].")

    if symbol.domain_type != "regular":
        raise ValidationError(
            "Cannot generate records unless the symbol has domain "
            "objects for all dimensions (i.e., <symbol>.domain_type == 'regular')"
        )

    domain = cast("list[Set | Alias | UniverseAlias]", symbol.domain)
    for symobj in domain:
        if symobj.records is None:
            raise ValidationError(
                f"Symbol `{symobj.name}` was used as a domain, but it does not have records "
                "-- cannot generate records unless all domain objects have records."
            )

    is_empty = False
    if (isinstance(density, (int, float)) and density == 0) or (
        isinstance(density, list) and any(d == 0 for d in density)
    ):
        is_empty = True

    return density, is_empty


def _set_empty_records(symbol: Any) -> None:
    """Helper function to initialize empty records for a symbol."""
    attributes = getattr(symbol, "_attributes", [])
    symbol.records = pd.DataFrame(
        columns=[list(range(symbol.dimension + len(attributes)))]
    )

    symbol.domain_labels = symbol.domain_names

    for x in range(len(symbol.domain)):
        symbol.records.isetitem(
            x, symbol.records.iloc[:, x].astype(CategoricalDtype([], ordered=True))
        )


def _get_categories(symobj: Set | Alias | UniverseAlias) -> pd.Index | list[str]:
    """
    Helper function to get the labels a domain symbol contributes, as a pandas Index when possible.
    """
    from gamspy._symbols import UniverseAlias

    records = symobj.records
    if isinstance(symobj, UniverseAlias) or symobj.dimension != 1 or records is None:
        return symobj._getUELs(ignore_unused=True)

    column = records.iloc[:, 0]
    if not isinstance(column.dtype, CategoricalDtype):
        return symobj._getUELs(ignore_unused=True)

    categories = column.cat.categories
    # `.array.codes` reads the Categorical's own backing array directly --
    # `.cat.codes.to_numpy()` wraps the codes in a brand new `pd.Series`
    # (with its own defensive copy) just to immediately unwrap it again.
    used, _ = used_category_positions(column.array.codes, categories.size)
    if used.size == categories.size:
        return categories

    return categories[used]


def _get_categorical_dtype(categories: pd.Index | list[str]) -> CategoricalDtype:
    if isinstance(categories, pd.Index):
        # Domain labels are unique by construction, so skip pandas' uniqueness
        # check -- it dominates the runtime for large domain sets.
        return CategoricalDtype._from_fastpath(categories=categories, ordered=True)

    return CategoricalDtype(categories, ordered=True)


def _validate_cardinality(num_rows: int) -> None:
    """Helper function to reject record counts that cannot even be enumerated."""
    if num_rows > np.iinfo(np.int64).max:
        raise ValidationError(
            f"Generating records would require enumerating {num_rows} rows, which is "
            "too large to index. Either use smaller domain sets or pass a `density` "
            "list to reduce the domains before their cartesian product is taken."
        )


def _sample_cartesian_rows(
    shape: list[int], num_rows: int, seed: int | None
) -> list[np.ndarray]:
    """Helper function to draw `num_rows` rows out of the cartesian product of `shape`."""
    cardinality = math.prod(shape)

    if num_rows == cardinality:
        # Every row is kept; enumerate them directly
        return list(np.indices(shape).reshape(len(shape), -1))

    idx = choice_no_replace(cardinality, num_rows, seed=seed)

    return list(np.unravel_index(idx, shape))


def _generate_base_dataframe(
    domain: list[Set | Alias | UniverseAlias],
    density: int | float | list,
    seed: int | None = None,
) -> pd.DataFrame:
    """Helper function to perform cartesian products, density sampling, and categorical conversions."""

    categories = [_get_categories(symobj) for symobj in domain]

    if isinstance(density, (int, float)):
        # A scalar density samples rows out of the cartesian product of the complete domains.
        shape = [len(cats) for cats in categories]
        cardinality = math.prod(shape)
        _validate_cardinality(cardinality)
        num_rows = min(int(density * cardinality), cardinality)
        codes = _sample_cartesian_rows(shape, num_rows, seed)
    elif isinstance(density, list):
        # A density per dimension samples the labels of each domain first and
        # then takes the cartesian product of whatever survived.
        selected = [
            choice_no_replace(len(cats), dense * len(cats), seed=seed)
            for cats, dense in zip(categories, density, strict=True)
        ]
        _validate_cardinality(math.prod(len(sel) for sel in selected))
        codes = cartesian_product(*tuple(selected))
    else:
        raise TypeError(f"Encountered unsupported 'density' type: {type(density)}")

    # Codes are generated from the positions of the domain labels, hence they are
    # in-bounds by construction: skip the validation pandas would otherwise run.
    data = {
        x: pd.Categorical.from_codes(
            codes=codes[x], dtype=_get_categorical_dtype(categories[x]), validate=False
        )
        for x in range(len(domain))
    }

    return pd.DataFrame(data, copy=False)


def generate_records_set(
    symbol: Set,
    density: int | float | list | None = None,
    seed: int | None = None,
) -> None:
    density, is_empty = _validate_density_and_domain(symbol, density)

    if is_empty:
        _set_empty_records(symbol)
        return

    domain = cast("list[Set | Alias | UniverseAlias]", symbol.domain)
    records = _generate_base_dataframe(domain, density, seed)
    records.insert(len(records.columns), "element_text", "")
    symbol.records = records
    symbol.domain_labels = symbol.domain_names
    symbol._removeUELs()


def generate_records_parameter(
    symbol: Parameter,
    density: int | float | list | None = None,
    func: Callable | None = None,
    seed: int | None = None,
) -> None:
    density, is_empty = _validate_density_and_domain(symbol, density)

    if not (callable(func) or func is None):
        raise TypeError("Argument 'func' must be a callable or None")

    if is_empty:
        _set_empty_records(symbol)
        return

    domain = cast("list[Set | Alias | UniverseAlias]", symbol.domain)
    records = _generate_base_dataframe(domain, density, seed)

    if func is None:
        rng = np.random.default_rng(seed)
        records["value"] = rng.uniform(low=0.0, high=1.0, size=(len(records),))
    else:
        records["value"] = func(seed=seed, size=(len(records),))
        cols = list(records.columns)
        records.isetitem(cols.index("value"), records["value"].astype(float))

    symbol.records = records
    symbol.domain_labels = symbol.domain_names
    symbol._removeUELs()


def generate_records_variable(
    symbol: Variable | Equation,
    density: int | float | list | None = None,
    func: dict[str, Callable] | None = None,
    seed: int | None = None,
) -> None:
    density, is_empty = _validate_density_and_domain(symbol, density)

    if func is not None and not isinstance(func, dict):
        raise TypeError("Argument 'func' must be a dict or NoneType")

    if isinstance(func, dict):
        if any(i not in symbol._attributes for i in func):
            raise ValueError(
                f"Unrecognized equation attribute detected in `func`. "
                f"Attributes must be {symbol._attributes}, user passed "
                f"dict keys: {list(func.keys())}."
            )
        for i in func:
            if not callable(func[i]):
                raise TypeError(
                    f"Object supplied to `func` argument (`{i}`) must be callable -- received {type(func[i])}"
                )

    if is_empty:
        _set_empty_records(symbol)
        return

    domain = cast("list[Set | Alias | UniverseAlias]", symbol.domain)
    records = _generate_base_dataframe(domain, density, seed)

    if func is None:
        rng = np.random.default_rng(seed)
        records["level"] = rng.uniform(low=0.0, high=1.0, size=(len(records),))

        for i in symbol._attributes:
            if i != "level":
                records[i] = symbol._default_records[i]
    else:
        for i in symbol._attributes:
            if i in func:
                records[i] = func[i](seed=seed, size=(len(records),))
                cols = list(records.columns)
                records.isetitem(cols.index(i), records[i].astype(float))
            else:
                records[i] = symbol._default_records[i]

    symbol.records = records
    symbol.domain_labels = symbol.domain_names
    symbol._removeUELs()


generate_records_equation = generate_records_variable
