from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from gamspy._algorithms import cartesian_product, generate_unique_labels
from gamspy._internals import EPS, NA, UNDEF, DomainStatus
from gamspy._special_values import SpecialValues
from gamspy._symbols.utils import (
    _assert_axes_no_nans,
    _flatten_and_convert,
    _get_implied_dimension_from_axes,
)

if TYPE_CHECKING:
    from gamspy._symbols import Equation, Parameter, Variable
    from gamspy._types import (
        ParameterRecordsType,
        SetRecordsType,
        SymbolWithRecordsType,
        VarEquRecordsType,
    )


class BaseIngestor:
    """Base strategy for ingesting records into GAMSPy symbols."""

    def __init__(self, symbol: SymbolWithRecordsType):
        self.symbol = symbol

    def ingest(
        self,
        records: SetRecordsType | ParameterRecordsType | VarEquRecordsType,
        *,
        uels_on_axes: bool = False,
    ) -> None:
        if isinstance(records, (int, float)):
            self._from_int_float(records)
        elif isinstance(records, np.ndarray):
            self._from_ndarray(records)
        elif isinstance(records, pd.DataFrame):
            self._from_dataframe(records, uels_on_axes)
        elif isinstance(records, pd.Series):
            self._from_series(records, uels_on_axes)
        elif isinstance(records, dict):
            self._from_dict(records)
        else:
            self._from_else(records)

    def _format_categorical_domains(self, records: pd.DataFrame) -> pd.DataFrame:
        """Converts domain columns into pd.Categorical objects, standardizing strings."""
        for i in range(self.symbol.dimension):
            col = records.iloc[:, i]

            if isinstance(col.dtype, pd.CategoricalDtype):
                # Already categorical (e.g. coming from GAMS Transfer): operate on
                # the category list.
                old_cats = col.cat.categories.tolist()
                new_cats = list(dict.fromkeys(str(x).rstrip() for x in old_cats))

                if len(old_cats) != len(new_cats):
                    # Whitespace stripping collapsed categories: re-encode the column.
                    records.isetitem(
                        i,
                        col.astype(str)
                        .str.rstrip()
                        .astype(
                            pd.CategoricalDtype(
                                categories=new_cats, ordered=col.cat.ordered
                            )
                        ),
                    )
                elif new_cats != old_cats:
                    records.isetitem(i, col.cat.rename_categories(new_cats))
                # else: categories unchanged, nothing to do.
            else:
                # Single factorize pass yields codes + appearance-ordered uniques,
                # matching the previous `col.unique()` ordering semantics.
                codes, uniques = pd.factorize(col, sort=False)
                stripped = [str(x).rstrip() for x in uniques]
                new_cats = list(dict.fromkeys(stripped))

                if len(new_cats) == len(stripped):
                    # No collisions after stripping: the codes are still valid
                    # skip pandas validations.
                    dtype = pd.CategoricalDtype._from_fastpath(
                        categories=pd.Index(stripped), ordered=True
                    )
                    records.isetitem(
                        i,
                        pd.Categorical.from_codes(codes, dtype=dtype, validate=False),
                    )
                else:
                    records.isetitem(
                        i,
                        col.astype(str)
                        .str.rstrip()
                        .astype(pd.CategoricalDtype(categories=new_cats, ordered=True)),
                    )

        return records

    def _remap_str_special_values(self, records: pd.DataFrame) -> pd.DataFrame:
        """Converts string representations of EPS, NA, and UNDEF to float equivalents."""
        from pandas.api.types import infer_dtype

        start_idx = self.symbol.dimension
        for i in records.columns[start_idx:]:
            if infer_dtype(records[i]) not in [
                "integer",
                "floating",
                "mixed-integer-float",
            ]:
                records[i] = records[i].astype(object)

                # Replace EPS
                idx = records.loc[:, i].isin(EPS)
                if idx.any():
                    records.loc[idx, i] = SpecialValues.EPS

                # Replace UNDEF
                idx = records.loc[:, i].isin(UNDEF)
                if idx.any():
                    records.loc[idx, i] = SpecialValues.UNDEF

                # Replace NA
                idx = records.loc[:, i].isin(NA)
                if idx.any():
                    records.loc[idx, i] = SpecialValues.NA

        return records

    def _filter_zero_records(self, records: pd.DataFrame) -> pd.DataFrame:
        val_cols = records.columns[self.symbol.dimension :]
        vals_array = records[val_cols].to_numpy(dtype=np.float64)

        # Evaluate zeros (this captures both 0.0 and -0.0 aka EPS)
        is_zero = vals_array == 0
        is_all_zero = is_zero.all(axis=1)
        has_eps = (is_zero & np.signbit(vals_array)).any(axis=1)

        # Keep rows that are not all zeros, or rows that contain at least one EPS
        keep_mask = ~is_all_zero | has_eps

        return records[keep_mask].reset_index(drop=True)

    def _position_of_attributes(self, records: pd.DataFrame | pd.Series) -> list[bool]:
        idx = []
        attr_set = set(self.symbol._attributes)

        for axis in records.axes:
            for n in range(axis.nlevels):
                # Extract the correct level iterable
                levels = axis.levels[n] if isinstance(axis, pd.MultiIndex) else axis
                idx.append(all(str(i).casefold() in attr_set for i in levels))

        return idx

    def _from_int_float(self, records: int | float) -> None:  # pragma: no cover
        raise NotImplementedError

    def _from_ndarray(self, records: np.ndarray) -> None:  # pragma: no cover
        raise NotImplementedError

    def _from_dataframe(
        self, records: pd.DataFrame, uels_on_axes: bool
    ) -> None:  # pragma: no cover
        raise NotImplementedError

    def _from_series(
        self, records: pd.Series, uels_on_axes: bool
    ) -> None:  # pragma: no cover
        raise NotImplementedError

    def _from_dict(self, records: dict) -> None:  # pragma: no cover
        raise NotImplementedError

    def _from_else(self, records: Any) -> None:  # pragma: no cover
        raise NotImplementedError


class ParameterIngestor(BaseIngestor):
    symbol: Parameter

    def _finalize_records(
        self, records: pd.DataFrame, col_labels: list[str] | None = None
    ) -> None:
        records = self._remap_str_special_values(records)

        if not isinstance(records.iloc[:, -1].dtype, float):
            records.isetitem(-1, records.iloc[:, -1].astype(float))

        labels = col_labels if col_labels is not None else self.symbol.domain_names
        records.columns = generate_unique_labels(labels) + self.symbol._attributes
        self.symbol.records = records

    def _from_int_float(self, records: int | float) -> None:
        if not self.symbol.is_scalar:
            raise ValueError(
                f"The records of a {self.symbol.dimension} dimensional symbol cannot be set with a scalar value"
            )
        self.symbol.records = pd.DataFrame(
            [records], dtype=float, columns=self.symbol._attributes
        )

    def _from_ndarray(self, records: np.ndarray) -> None:
        try:
            records = np.array(records, dtype=float)
        except Exception as err:
            raise TypeError(
                f"Attempted conversion to numpy array failed. Reason: {err}"
            ) from err

        if self.symbol.dimension == 1 and records.shape in (
            (1, records.size),
            (records.size, 1),
        ):
            records = records.reshape((records.size,))

        if records.ndim != self.symbol.dimension:
            raise ValueError(
                "Dimension mismatch between numpy array and parameter domain."
            )

        if records.ndim > 0 and self.symbol.domain_type != "regular":
            raise ValueError(
                f"Data conversion for array format requires a 'regular' domain type but found {self.symbol.domain_type}."
            )

        if records.shape != self.symbol.shape:
            raise ValueError(
                "Shape mismatch between numpy array and parameter domains."
            )

        if self.symbol.is_scalar:
            df = pd.DataFrame(index=[0])
        else:
            codes = [
                np.arange(len(d._getUELs(ignore_unused=True)))
                for d in self.symbol.domain
            ]
            df = pd.DataFrame(cartesian_product(*tuple(codes)))

            for n, d in enumerate(self.symbol.domain):
                # Codes come from a cartesian product of arange(...) over each
                # domain's UELs, so they are in-bounds by construction: skip the
                # redundant validation pandas would otherwise run.
                dtype = pd.CategoricalDtype._from_fastpath(
                    categories=d.records.iloc[:, 0].cat.categories,
                    ordered=d.records.iloc[:, 0].cat.ordered,
                )
                df.isetitem(
                    n,
                    pd.Categorical.from_codes(
                        codes=df.iloc[:, n], dtype=dtype, validate=False
                    ),
                )

        df["value"] = records.reshape(-1, 1)
        df = self._filter_zero_records(df)
        df.columns = (
            generate_unique_labels(self.symbol.domain_names) + self.symbol._attributes
        )
        self.symbol.records = df

    def _from_dataframe(self, records: pd.DataFrame, uels_on_axes: bool) -> None:
        if self.symbol.is_scalar or not uels_on_axes:
            self._from_flat_dataframe(records)
        else:
            self._from_table_dataframe(records)

    def _from_flat_dataframe(self, records: pd.DataFrame) -> None:
        records = (
            records.copy()
            if isinstance(records, pd.DataFrame)
            else pd.DataFrame(records)
        )
        r, c = records.shape

        if c - 1 != self.symbol.dimension:
            raise ValueError(
                "Dimensionality of records is inconsistent with domain specification."
            )

        if self.symbol.is_scalar and r > 1:
            raise ValueError(
                "Attempting to set multiple records for a scalar parameter."
            )

        records = self._format_categorical_domains(records)
        self._finalize_records(
            records, col_labels=records.columns[: self.symbol.dimension].tolist()
        )

    def _from_table_dataframe(self, records: pd.DataFrame) -> None:
        records = (
            records.copy()
            if isinstance(records, pd.DataFrame)
            else pd.DataFrame(records)
        )
        _assert_axes_no_nans(records)

        dim = _get_implied_dimension_from_axes(records)
        if dim != self.symbol.dimension:
            raise ValueError(
                "Dimensionality of table is inconsistent with domain specification."
            )

        records = _flatten_and_convert(records)
        self._finalize_records(records)

    def _from_series(self, records: pd.Series, uels_on_axes: bool) -> None:
        records = copy.deepcopy(records)
        _assert_axes_no_nans(records)

        if self.symbol.is_scalar:
            if records.size == 1:
                self._from_dataframe(pd.DataFrame(records), False)
            else:
                raise ValueError(
                    "pandas.Series must have size exactly = 1 for a scalar parameter."
                )
        else:
            dim = _get_implied_dimension_from_axes(records)
            if dim != self.symbol.dimension:
                raise ValueError(
                    "Dimensionality of data is inconsistent with domain specification."
                )

            records = _flatten_and_convert(records)
            self._finalize_records(records)

    def _from_dict(self, records: dict) -> None:
        self._from_else(records)

    def _from_else(self, records: ParameterRecordsType) -> None:
        try:
            records = (
                records.copy()
                if isinstance(records, pd.DataFrame)
                else pd.DataFrame(records)
            )
        except Exception as err:
            raise TypeError(f"Could not convert to pandas DataFrame: {err}") from err

        c = records.shape[1]
        if c - 1 != self.symbol.dimension:
            raise ValueError(
                "Dimensionality of records is inconsistent with domain specification."
            )

        records = self._format_categorical_domains(records)
        self._finalize_records(records)


class SetIngestor(BaseIngestor):
    def _finalize_records(
        self, records: pd.DataFrame, col_labels: list[str] | None = None
    ) -> None:
        labels = col_labels if col_labels is not None else self.symbol.domain_names
        records.columns = generate_unique_labels(labels) + self.symbol._attributes

        records.isetitem(-1, records.iloc[:, -1].astype(object))
        records.iloc[records.iloc[:, -1].isna(), -1] = ""
        records.isetitem(-1, records.iloc[:, -1].astype(str))

        self.symbol.records = records

    def _from_int_float(self, records: int | float) -> None:
        raise TypeError("Sets cannot be initialized with integers/floats.")

    def _from_ndarray(self, records: np.ndarray) -> None:
        self._from_else(records)

    def _from_dataframe(self, records: pd.DataFrame, uels_on_axes: bool) -> None:
        from pandas.api.types import is_bool_dtype

        if not uels_on_axes:
            self._from_else(records)
            return

        if is_bool_dtype(records.to_numpy().dtype):
            self._from_table_dataframe(records)
        else:
            if any(
                not is_bool_dtype(dtype) for dtype in records.convert_dtypes().dtypes
            ):
                raise TypeError(
                    "All columns must be type bool when `uels_on_axes=True`."
                )
            self._from_table_dataframe(records.convert_dtypes())

    def _from_table_dataframe(self, records: pd.DataFrame) -> None:
        records = (
            records.copy()
            if isinstance(records, pd.DataFrame)
            else pd.DataFrame(records)
        )
        _assert_axes_no_nans(records)

        dim = _get_implied_dimension_from_axes(records)
        if dim != self.symbol.dimension:
            raise ValueError(
                "Dimensionality of table is inconsistent with set domain specification."
            )

        records = _flatten_and_convert(records)

        records = records[records.iloc[:, -1].astype(bool)].copy()
        records.drop(columns=records.columns[-1], inplace=True)
        records.reset_index(drop=True, inplace=True)
        records["element_text"] = ""

        self._finalize_records(records)

    def _from_series(self, records: pd.Series, uels_on_axes: bool) -> None:
        if uels_on_axes:
            records = copy.deepcopy(records)
            _assert_axes_no_nans(records)

            dim = _get_implied_dimension_from_axes(records)
            if dim != self.symbol.dimension:
                raise ValueError(
                    "Dimensionality of data is inconsistent with domain specification."
                )

            records = _flatten_and_convert(records)
            self._finalize_records(records)
        else:
            if self.symbol.dimension != 1:
                raise ValueError(
                    "Dimensionality of data (1) is inconsistent with domain specification."
                )

            records: pd.DataFrame = pd.DataFrame(records)
            if records.shape[1] == self.symbol.dimension:
                records = records.assign(element_text="")

            records.columns = (
                generate_unique_labels(self.symbol.domain_names)
                + self.symbol._attributes
            )
            self._from_dataframe(records, False)

    def _from_dict(self, records: dict) -> None:
        self._from_else(records)

    def _from_else(self, records: SetRecordsType) -> None:
        if isinstance(records, str):
            records = [records]

        try:
            from_dataframe = isinstance(records, pd.DataFrame)
            records = pd.DataFrame(
                copy.deepcopy(records) if from_dataframe else records
            )
        except Exception as err:
            raise TypeError(f"Could not convert to pandas DataFrame: {err}") from err

        if records.shape[1] == self.symbol.dimension:
            records = records.assign(element_text="")

        if records.shape[1] - 1 != self.symbol.dimension:
            raise ValueError(
                "Dimensionality of records is inconsistent with set domain specification."
            )

        records = self._format_categorical_domains(records)
        labels = (
            records.columns[: self.symbol.dimension].tolist()
            if from_dataframe
            else None
        )

        self._finalize_records(records, col_labels=labels)


class VarEquIngestor(BaseIngestor):
    symbol: Variable | Equation

    def _from_int_float(self, records: int | float) -> None:
        if not self.symbol.is_scalar:
            raise ValueError(
                "Attempting to set a scalar value, but symbol is not scalar."
            )
        self._from_flat_dataframe(pd.DataFrame([records], columns=["level"]))

    def _from_ndarray(self, records: np.ndarray) -> None:
        self._from_dict_of_arrays({"level": records})

    def _from_dict(self, records: dict) -> None:
        if all(
            i in self.symbol._attributes
            and isinstance(records[i], (np.ndarray, int, float))
            for i in records
        ):
            self._from_dict_of_arrays(records)
        else:
            self._from_else(records)

    def _from_dict_of_arrays(self, records: dict) -> None:
        if any(i not in self.symbol._attributes for i in records):
            raise ValueError(
                f"Unrecognized attribute detected. Attributes must be {self.symbol._attributes}."
            )

        for k, v in records.items():
            records[k] = np.array(v, dtype=float)

        for k, arr in records.items():
            if self.symbol.dimension == 1 and arr.shape in (
                (1, arr.size),
                (arr.size, 1),
            ):
                records[k] = arr.reshape((arr.size,))
            if arr.ndim != self.symbol.dimension:
                raise ValueError("Dimensionality mismatch between arrays and symbol.")

        shapes = [arr.shape for arr in records.values()]
        if any(i != shapes[0] for i in shapes):
            raise ValueError("Arrays passed do not have the same shape.")

        if (
            self.symbol.dimension > 0
            and self.symbol._domain_status is not DomainStatus.regular
        ):
            raise ValueError(
                "Data conversion for arrays requires a 'regular' domain type."
            )

        for arr in records.values():
            if arr.shape != self.symbol.shape:
                raise ValueError("Shape mismatch between numpy arrays and domains.")

        if self.symbol.is_scalar:
            df = pd.DataFrame(index=[0], columns=list(records.keys()))
        else:
            codes = [
                np.arange(len(d._getUELs(ignore_unused=True)))
                for d in self.symbol.domain
            ]
            df = pd.DataFrame(cartesian_product(*tuple(codes)))

            for n, d in enumerate(self.symbol.domain):
                # Codes come from a cartesian product of arange(...) over each
                # domain's UELs, so they are in-bounds by construction: skip the
                # redundant validation pandas would otherwise run.
                dtype = pd.CategoricalDtype._from_fastpath(
                    categories=d.records.iloc[:, 0].cat.categories,
                    ordered=d.records.iloc[:, 0].cat.ordered,
                )
                df.isetitem(
                    n,
                    pd.Categorical.from_codes(
                        codes=df.iloc[:, n], dtype=dtype, validate=False
                    ),
                )

        for i in records:
            df[i] = records[i].reshape(-1, 1)

        df = self._filter_zero_records(df)

        # Fill missing attributes with defaults
        if set(records.keys()) != set(self.symbol._attributes):
            for i in set(self.symbol._attributes) - set(records.keys()):
                df[i] = self.symbol._default_records[i]

        df = pd.concat(
            [df.iloc[:, : self.symbol.dimension], df[self.symbol._attributes]], axis=1
        )
        df.columns = (
            generate_unique_labels(self.symbol.domain_names) + self.symbol._attributes
        )
        self.symbol.records = df

    def _from_dataframe(self, records: pd.DataFrame, uels_on_axes: bool) -> None:
        if self.symbol.is_scalar or not uels_on_axes:
            self._from_flat_dataframe(records)
        else:
            self._from_table_dataframe(records)

    def _from_flat_dataframe(self, records: pd.DataFrame) -> None:
        records = (
            records.copy()
            if isinstance(records, pd.DataFrame)
            else pd.DataFrame(records)
        )

        # Fill defaults
        if set(records[self.symbol.dimension :].columns) != set(
            self.symbol._attributes
        ):
            for i in set(self.symbol._attributes) - set(
                records[self.symbol.dimension :].columns
            ):
                records[i] = self.symbol._default_records[i]

        r = records.shape[0]
        if len(records.columns) != self.symbol.dimension + len(self.symbol._attributes):
            raise ValueError(
                "Dimensionality of records is inconsistent with domain specification."
            )

        if self.symbol.is_scalar and r > 1:
            raise ValueError("Attempting to set multiple records for a scalar symbol.")

        records = pd.concat(
            [
                records.iloc[:, : self.symbol.dimension],
                records[self.symbol._attributes],
            ],
            axis=1,
        )
        records = self._format_categorical_domains(records)
        records = self._remap_str_special_values(records)

        # Convert attributes to float
        cols = list(records.columns)
        for i in records.columns[self.symbol.dimension :]:
            records.isetitem(cols.index(i), records[i].astype(float))

        records.columns = (
            generate_unique_labels(records.columns[: self.symbol.dimension].tolist())
            + self.symbol._attributes
        )
        self.symbol.records = records

    def _from_table_dataframe(self, records: pd.DataFrame) -> None:
        records = (
            records.copy()
            if isinstance(records, pd.DataFrame)
            else pd.DataFrame(records)
        )
        _assert_axes_no_nans(records)

        n_idx = self._position_of_attributes(records)
        if sum(n_idx) > 1:
            raise ValueError("Attributes detected in more than one index.")

        dim = _get_implied_dimension_from_axes(records) - sum(n_idx)
        if dim != self.symbol.dimension:
            raise ValueError(
                "Dimensionality of table is inconsistent with domain specification."
            )

        records = _flatten_and_convert(records)
        records = self._remap_str_special_values(records)

        if any(n_idx):
            attr = records.iloc[:, n_idx.index(True)].cat.categories.tolist()
            records = (
                records.set_index(records.columns.tolist()[:-1])
                .unstack(n_idx.index(True))
                .reset_index(drop=False)
            )
            records.columns = ["*"] * self.symbol.dimension + attr
        else:
            records.columns = ["*"] * self.symbol.dimension + ["level"]

        # Fill defaults
        if set(records.columns) != set(self.symbol._attributes):
            for i in set(self.symbol._attributes) - set(records.columns):
                records[i] = self.symbol._default_records[i]

        records = pd.concat(
            [
                records.iloc[:, : self.symbol.dimension],
                records[self.symbol._attributes],
            ],
            axis=1,
        )

        for i in range(
            self.symbol.dimension, self.symbol.dimension + len(self.symbol._attributes)
        ):
            if not isinstance(records.iloc[:, i].dtype, float):
                records.isetitem(i, records.iloc[:, i].astype(float))

        records.columns = (
            generate_unique_labels(self.symbol.domain_names) + self.symbol._attributes
        )
        self.symbol.records = records

    def _from_series(self, records: pd.Series, uels_on_axes: bool) -> None:
        records = copy.deepcopy(records)
        _assert_axes_no_nans(records)

        n_idx = self._position_of_attributes(records)
        if sum(n_idx) > 1:
            raise ValueError(
                "Attributes detected in more than one level of a MultiIndex."
            )

        if self.symbol.is_scalar:
            if sum(n_idx):
                recs = pd.DataFrame(columns=records.index.tolist())
                for i in records.index:
                    recs.loc[0, i] = records[i]
                self._from_dataframe(recs, False)
            elif records.size == 1:
                self._from_dataframe(pd.DataFrame(records, columns=["level"]), False)
            else:
                raise ValueError(
                    "pandas.Series must have size exactly = 1 for a scalar symbol."
                )
        else:
            dim = _get_implied_dimension_from_axes(records) - sum(n_idx)
            if dim != self.symbol.dimension:
                raise ValueError(
                    "Dimensionality of table is inconsistent with domain specification."
                )

            records = _flatten_and_convert(records)
            records = self._remap_str_special_values(records)

            if any(n_idx):
                attr = records.iloc[:, n_idx.index(True)].cat.categories.tolist()
                records = (
                    records.set_index(records.columns.tolist()[:-1])
                    .unstack(n_idx.index(True))
                    .reset_index(drop=False)
                )
                records.columns = ["*"] * self.symbol.dimension + attr
            else:
                records.columns = ["*"] * self.symbol.dimension + ["level"]

            self._from_dataframe(records, False)

    def _from_else(self, records: VarEquRecordsType) -> None:
        try:
            records = pd.DataFrame(records)
        except Exception as err:
            raise TypeError(f"Could not convert to pandas DataFrame: {err}") from err

        self._from_flat_dataframe(records)
