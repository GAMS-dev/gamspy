from __future__ import annotations

import typing
from typing import cast, no_type_check

import numpy as np
import pandas as pd

from gamspy._special_values import SpecialValues
from gamspy.exceptions import ValidationError

if typing.TYPE_CHECKING:
    from gamspy._symbols import Alias, Equation, Parameter, Set, Variable
    from gamspy._types import SymbolType


def _validate_symbols(
    symbol: SymbolType,
    other: SymbolType,
) -> SymbolType:
    from gamspy._symbols import Alias
    from gamspy._symbols.base import BaseSymbol

    if not isinstance(other, BaseSymbol):
        raise TypeError("Argument 'other' must be a GAMS Symbol object")

    if isinstance(other, Alias):
        other = other.alias_with

    if not isinstance(symbol, type(other)):
        raise TypeError(
            f"Symbol are not of the same type (`{type(symbol)}` != `{type(other)}`)"
        )

    # test for equal variable and equation types
    if getattr(symbol, "type", None) != getattr(other, "type", None):
        raise TypeError(
            f"Symbol types do not match (`{getattr(symbol, 'type', None)}` != `{getattr(other, 'type', None)}`)"
        )

    return other


# TODO: Legacy function from GTP. Pay the technical debt.
@no_type_check
def _assert_symbol_attributes(
    symbol: SymbolType,
    other: SymbolType,
    check_meta_data: bool,
) -> None:
    # Mandatory checks
    if symbol.dimension != other.dimension:
        raise ValidationError(
            f"Symbol dimensions do not match (`{symbol.dimension}` != `{other.dimension}`)"
        )

    if symbol.domain_type != other.domain_type:
        raise ValidationError(
            f"Symbol domain_types do not match (`{symbol.domain_type}` != `{other.domain_type}`)"
        )

    if (
        symbol.records is not None
        and other.records is not None
        and symbol.number_records != other.number_records
    ):
        raise ValidationError(
            "Symbols do not have the same number of records "
            f"(`{symbol.number_records}` != `{other.number_records}`)"
        )

    if not isinstance(symbol.records, type(other.records)):
        raise ValidationError(
            f"Symbol records type do not match (`{type(symbol.records)}` != `{type(other.records)}`)"
        )

    # Check metadata (optional)
    if check_meta_data:
        if symbol.name != other.name:
            raise ValidationError(
                f"Symbol names do not match (`{symbol.name}` != `{other.name}`)"
            )

        if symbol.description != other.description:
            raise ValidationError(
                f"Symbol descriptions do not match (`{symbol.description}` != `{other.description}`)"
            )


def _merge_records(symbol: SymbolType, other: SymbolType) -> pd.DataFrame:
    merged = pd.DataFrame()
    if (
        symbol.records is not None
        and other.records is not None
        and not symbol.records.empty
        and not other.records.empty
    ):
        merged = symbol.records.merge(
            other.records,
            how="outer",
            left_on=symbol.domain_labels,
            right_on=other.domain_labels,
            indicator=True,
        )
    return merged


def _assert_scalar_values(
    symbol: SymbolType,
    other: SymbolType,
    columns: list[str],
    rtol: float,
    atol: float,
) -> None:
    if symbol.records is None or other.records is None:
        return None

    for attr in columns:
        for svlabel, SV in zip(
            ["EPS", "NA", "UNDEF"],
            [SpecialValues.isEps, SpecialValues.isNA, SpecialValues.isUndef],
            strict=True,
        ):
            self_is_special = SV(symbol.records[attr])
            other_is_special = SV(other.records[attr])

            if self_is_special != other_is_special:
                raise ValidationError(
                    f"Symbol records with `{svlabel}` special values "
                    f"do not match in the `{attr}` column."
                )

        if (
            not self_is_special
            and not other_is_special
            and not np.isclose(
                symbol.records[attr],
                other.records[attr],
                rtol=rtol,
                atol=atol,
            )
        ):
            raise ValidationError(
                f"Symbol records contain numeric difference in the `{attr}` attribute "
                f"that are outside the specified tolerances (rtol={rtol}, atol={atol})"
            )


def _assert_symbol_domains(
    symbol: SymbolType, other: SymbolType, merged: pd.DataFrame
) -> None:
    if set(merged["_merge"]) != {"both"}:
        self_only_recs = merged[merged["_merge"].isin({"left_only"})].head()
        other_only_recs = merged[merged["_merge"].isin({"right_only"})].head()

        if self_only_recs.empty:
            self_only_recs = "All matched OK"
        else:
            self_only_recs = list(
                self_only_recs[self_only_recs.columns[: symbol.dimension]].itertuples(
                    index=False, name=None
                )
            )

        if other_only_recs.empty:
            other_only_recs = "All matched OK"
        else:
            other_only_recs = list(
                other_only_recs[other_only_recs.columns[: other.dimension]].itertuples(
                    index=False, name=None
                )
            )

        raise ValidationError(
            "Symbol records do not match. First five unmatched domains: \n\n"
            f"left_only :  {self_only_recs} \n"
            f"right_only:  {other_only_recs} \n"
        )


def _assert_symbol_values(
    symbol: SymbolType,
    merged: pd.DataFrame,
    columns: list[str],
    rtol: float,
    atol: float,
) -> None:
    if not merged.empty:
        for attr in columns:
            small_merged = merged[
                list(merged.columns[: symbol.dimension]) + [f"{attr}_x", f"{attr}_y"]
            ].copy()

            for svlabel, SV in zip(
                ["EPS", "NA", "UNDEF"],
                [SpecialValues.isEps, SpecialValues.isNA, SpecialValues.isUndef],
                strict=True,
            ):
                self_idx = SV(small_merged[f"{attr}_x"])
                other_idx = SV(small_merged[f"{attr}_y"])

                if any(self_idx != other_idx):
                    raise ValidationError(
                        f"Symbol records with `{svlabel}` special values "
                        f"do not match in the `{attr}` column."
                    )

                # drop special values if all indices match
                small_merged.drop(index=small_merged[self_idx].index, inplace=True)

            # check attr values (subject to tolerances)
            isclose = np.isclose(
                small_merged[f"{attr}_x"],
                small_merged[f"{attr}_y"],
                rtol=rtol,
                atol=atol,
            )

            if any(~isclose):
                raise ValidationError(
                    f"Symbol records contain numeric difference in the `{attr}` attribute "
                    f"that are outside the specified tolerances (rtol={rtol}, atol={atol})"
                )


def equals_set(
    symbol: Set | Alias,
    other: Set | Alias,
    check_element_text: bool,
    check_meta_data: bool,
) -> bool:
    try:
        other = cast("Set | Alias", _validate_symbols(symbol, other))

        if symbol.is_singleton != other.is_singleton:
            raise ValidationError("Symbols do not have matching 'is_singleton' state")

        if not isinstance(check_element_text, bool):
            raise TypeError("Argument 'check_element_text' must be type bool")

        _assert_symbol_attributes(symbol, other, check_meta_data)
        merged = _merge_records(symbol, other)

        if not merged.empty:
            _assert_symbol_domains(symbol, other, merged)

            if check_element_text:
                merged["_element_text"] = (
                    merged["element_text_x"] != merged["element_text_y"]
                )
                recs = merged[merged["_element_text"]].head()

                if not recs.empty:
                    self_only_recs = list(
                        recs[
                            list(recs.columns[: symbol.dimension]) + ["element_text_x"]
                        ].itertuples(index=False, name=None)
                    )
                    other_only_recs = list(
                        recs[
                            list(recs.columns[: symbol.dimension]) + ["element_text_y"]
                        ].itertuples(index=False, name=None)
                    )

                    raise ValidationError(
                        "Symbol element_text does not match. First five unmatched domains: \n\n"
                        f"left_only :  {self_only_recs} \n"
                        f"right_only:  {other_only_recs} \n"
                    )

        return True
    except Exception:
        return False


def equals_parameter(
    symbol: Parameter,
    other: Parameter,
    check_meta_data: bool = True,
    rtol: int | float | None = None,
    atol: int | float | None = None,
) -> bool:
    if not isinstance(rtol, (type(None), int, float)):
        raise ValueError(
            "Argument 'rtol' (relative tolerance) must be "
            f"numeric (int, float) or None.  User passed: {type(rtol)}."
        )

    if not isinstance(atol, (type(None), int, float)):
        raise ValueError(
            "Argument 'atol' (relative tolerance) must be "
            f"numeric (int, float) or None.  User passed: {type(atol)}."
        )

    rtol = 0.0 if rtol is None else rtol
    atol = 0.0 if atol is None else atol
    columns = symbol._attributes

    try:
        other = cast("Parameter", _validate_symbols(symbol, other))
        _assert_symbol_attributes(symbol, other, check_meta_data)

        if symbol.dimension == 0:
            _assert_scalar_values(symbol, other, columns, rtol, atol)
        else:
            merged = _merge_records(symbol, other)
            _assert_symbol_domains(symbol, other, merged)
            _assert_symbol_values(symbol, merged, columns, rtol, atol)

        return True
    except Exception:
        return False


def equals_variable(
    symbol: Variable | Equation,
    other: Variable | Equation,
    columns: str | list[str] | None = None,
    check_meta_data: bool = True,
    rtol: int | float | None = None,
    atol: int | float | None = None,
) -> bool:
    if not isinstance(columns, (str, list, type(None))):
        raise TypeError("Argument 'columns' must be type str, list or NoneType")

    if symbol.records is None and other.records is None:
        return True

    if isinstance(columns, str):
        columns = [columns]

    if columns is None:
        columns = symbol._attributes

    if any(i not in symbol._attributes for i in columns):
        raise ValueError(
            f"Argument 'columns' can only contain the following symbol attributes: {symbol._attributes}"
        )

    if not isinstance(rtol, (type(None), int, float)):
        raise ValueError(
            "Argument 'rtol' (relative tolerance) must be "
            f"numeric (int, float) or None.  User passed: {type(rtol)}."
        )

    if not isinstance(atol, (type(None), int, float)):
        raise ValueError(
            "Argument 'atol' (relative tolerance) must be "
            f"numeric (int, float) or None.  User passed: {type(atol)}."
        )

    rtol = 0.0 if rtol is None else rtol
    atol = 0.0 if atol is None else atol

    try:
        other = cast("Variable | Equation", _validate_symbols(symbol, other))
        _assert_symbol_attributes(symbol, other, check_meta_data)

        if symbol.dimension == 0:
            _assert_scalar_values(symbol, other, columns, rtol, atol)
        else:
            merged = _merge_records(symbol, other)
            _assert_symbol_domains(symbol, other, merged)
            _assert_symbol_values(symbol, merged, columns, rtol, atol)

        return True
    except Exception:
        return False


equals_equation = equals_variable
