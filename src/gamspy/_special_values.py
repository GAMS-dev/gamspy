from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


class SpecialValues:
    """
    Utility class representing GAMS special values and methods to test for their presence.

    In GAMS, special values are used to represent missing data, undefined operations, infinities, or structural zeros (EPS).
    This class defines GAMSPy equivalents for these values. It also provides vectorized static methods to efficiently check
    for these special values across integers, floats, strings, pandas Series, pandas DataFrames, or array-like objects.

    Attributes
    ----------
    NA : float
        GAMS NA (Not Available), unpacked from a specific hexadecimal representation ("fffffffffffffffe").
    EPS : float
        GAMS EPS (Epsilon), represented as a negative zero (-0.0).
    UNDEF : float
        GAMS UNDEF (Undefined), represented as Not a Number (NaN).
    POSINF : float
        GAMS POSINF (Positive Infinity), represented as float("inf").
    NEGINF : float
        GAMS NEGINF (Negative Infinity), represented as float("-inf").

    Examples
    --------
    Basic usage of attributes


    >>> from gamspy._special_values import SpecialValues
    >>> SpecialValues.EPS
    -0.0
    >>> SpecialValues.POSINF
    inf


    Testing for special values in data structures


    >>> import numpy as np
    >>> import pandas as pd
    >>> data = pd.Series([1.5, -0.0, float("inf"), SpecialValues.NA])
    >>> SpecialValues.isEps(data)
    array([False,  True, False, False])
    >>> SpecialValues.isNA(data)
    array([False, False, False,  True])

    """

    NA = struct.unpack(">d", bytes.fromhex("fffffffffffffffe"))[0]
    EPS = -0.0
    UNDEF = float("nan")
    POSINF = float("inf")
    NEGINF = float("-inf")

    @staticmethod
    def _convertNPfloat64(records, sv_name) -> np.ndarray:
        if not (
            isinstance(records, np.ndarray) and np.issubdtype(records.dtype, np.float64)
        ):
            try:
                records = np.array(records, dtype=np.float64)
            except Exception as err:
                raise ValueError(
                    "Data structure passed in 'records' could not be "
                    "converted to a numpy array (dtype=np.float64) "
                    f"to test for GAMS {sv_name}, reason: {err}"
                ) from err
        return records

    @staticmethod
    def isEps(
        records: int | float | str | pd.Series | pd.DataFrame,
    ) -> np.ndarray:
        """
        Check if the input records represent a value close to zero with specific considerations for different data types.

        Parameters
        ----------
        records: int | float | str | pd.Series | pd.DataFrame | array-like
            The input records to be checked for proximity to zero.

        Returns
        -------
        np.ndarray
            True if the input records represent a value close to zero according to the specified conditions, False otherwise.

        Raises
        ------
        Exception
            If the input (string) records cannot be converted to a float.
        Exception
            If the data structure passed in 'records' could not be converted to a numpy array (dtype=float) for testing.
        """
        np_records = SpecialValues._convertNPfloat64(records, "EPS")
        return (np_records == 0) & (np.signbit(np_records))

    @staticmethod
    def isNA(
        records: int | float | str | pd.Series | pd.DataFrame,
    ) -> np.ndarray:
        """
        Check if values in records represent GAMS NA (Not Available) values.

        Parameters
        ----------
        records: int | float | str | pd.Series | pd.DataFrame | array-like
            The input records to be checked for GAMS NA values.

        Returns
        -------
        np.ndarray
            True if the values in records represent GAMS NA values; otherwise, False.

        Raises
        ------
        Exception
            If the input (string) records cannot be converted to a float.
        Exception
            If the data structure passed in 'records' could not be converted to a numpy array (dtype=float) for testing.
        """
        _NA_INT64 = np.float64(SpecialValues.NA).view(np.uint64)
        np_records = SpecialValues._convertNPfloat64(records, "NA")
        return np_records.view(np.uint64) == _NA_INT64

    @staticmethod
    def isUndef(
        records: int | float | str | pd.Series | pd.DataFrame,
    ) -> np.ndarray:
        """
        Determine if the given input(s) represent GAMS "undef" values.

        Parameters
        ----------
        records: int | float | str | pd.Series | pd.DataFrame | array-like
            The input records to be checked for GAMS "undef" values.

        Returns
        -------
        np.ndarray
            True if the values in records represent GAMS "undef" values; otherwise, False.

        Raises
        ------
        Exception
            If the input (string) records cannot be converted to a float.
        Exception
            If the data structure passed in 'records' could not be converted to a numpy array (dtype=float) for testing.
        """
        _NA_INT64 = np.float64(SpecialValues.NA).view(np.uint64)
        np_records = SpecialValues._convertNPfloat64(records, "UNDEF")
        return np.isnan(np_records) & (np_records.view(np.uint64) != _NA_INT64)

    @staticmethod
    def isPosInf(
        records: int | float | str | pd.Series | pd.DataFrame,
    ) -> np.ndarray | np.bool:
        """
        Check if the input records represent positive infinity.

        Parameters
        ----------
        records: int | float | str | pd.Series | pd.DataFrame | array-like
            The input records to be checked for positive infinity values.

        Returns
        -------
        np.ndarray | np.bool
            True if the values in records represent positive infinity values; otherwise, False.

        Raises
        ------
        Exception
            If the input (string) records cannot be converted to a float.
        Exception
            If the data structure passed in 'records' could not be converted to a numpy array (dtype=float) for testing.
        """
        np_records = SpecialValues._convertNPfloat64(records, "POSINF")
        return np.isposinf(np_records)

    @staticmethod
    def isNegInf(
        records: int | float | str | pd.Series | pd.DataFrame,
    ) -> np.ndarray | np.bool:
        """
        Check if the input records represent negative infinity.

        Parameters
        ----------
        records: int | float | str | pd.Series | pd.DataFrame | array-like
            The input records to be checked for negative infinity values.

        Returns
        -------
        np.ndarray | np.bool
            True if the values in records represent negative infinity values; otherwise, False.

        Raises
        ------
        Exception
            If the input (string) records cannot be converted to a float.
        Exception
            If the data structure passed in 'records' could not be converted to a numpy array (dtype=float) for testing.
        """
        np_records = SpecialValues._convertNPfloat64(records, "NEGINF")
        return np.isneginf(np_records)
