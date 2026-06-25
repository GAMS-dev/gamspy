from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from gams.core import gmd

from gamspy._algorithms import convert_to_categoricals_cat, generate_unique_labels
from gamspy._special_values import SpecialValues
from gamspy.exceptions import GmdException

if TYPE_CHECKING:
    from gams.core.numpy import Gams2Numpy
    from pandas import DataFrame


@dataclass
class SymbolMetadata:
    name: str
    gmd_symbol_number: int
    dimension: int
    type: int
    userinfo: int
    number_records: int
    description: str
    link_domains: list[bool]
    domains_as_ptrs: list[Any]
    domain: list[str]
    parent_set: str | None


def get_records(
    gams2np: Gams2Numpy,
    load_from,
    symbols: list[str] | None = None,
    encoding: str | None = None,
) -> dict[str, DataFrame]:
    default_special_vaules = gmd.doubleArray(gmd.GMS_SVIDX_MAX)
    gmd.gmdGetUserSpecialValues(load_from, default_special_vaules)

    # setting special values
    specVals = gmd.doubleArray(gmd.GMS_SVIDX_MAX)
    specVals[gmd.GMS_SVIDX_UNDEF] = SpecialValues.UNDEF
    specVals[gmd.GMS_SVIDX_NA] = SpecialValues.NA
    specVals[gmd.GMS_SVIDX_EPS] = SpecialValues.EPS
    specVals[gmd.GMS_SVIDX_PINF] = SpecialValues.POSINF
    specVals[gmd.GMS_SVIDX_MINF] = SpecialValues.NEGINF

    ret = gmd.gmdSetSpecialValues(load_from, specVals)
    assert ret

    rc = gmd.new_intp()
    ret = gmd.gmdInfo(load_from, gmd.GMD_NRSYMBOLSWITHALIAS)
    symCount = ret[1]

    SYMBOL_METADATA: list[SymbolMetadata] = []
    if symbols is not None:
        for sym in symbols:
            symptr = gmd.gmdFindSymbolWithAliasPy(load_from, sym, rc)
            if symptr is not None:
                ret = gmd.gmdSymbolInfo(load_from, symptr, gmd.GMD_NUMBER)
                SYMBOL_METADATA.append(
                    gmd_get_metadata_by_number(gams2np, load_from, ret[1], encoding, rc)
                )
    else:
        for i in range(1, symCount + 1):
            SYMBOL_METADATA.append(
                gmd_get_metadata_by_number(gams2np, load_from, i, encoding, rc)
            )

    GMD_UELS = gams2np.gmdGetUelList(load_from, encoding=encoding)
    GMD_UELS[0] = "*"
    records_dict = {}

    for md in SYMBOL_METADATA:
        if md.type == gmd.GMS_DT_ALIAS:
            continue

        if md.type in (gmd.GMS_DT_VAR, gmd.GMS_DT_EQU):
            attributes = ["level", "marginal", "lower", "upper", "scale"]
        elif md.type == gmd.GMS_DT_PAR:
            attributes = ["value"]
        elif md.type == gmd.GMS_DT_SET:
            attributes = ["element_text"]
        else:
            attributes = []

        if md.number_records == 0:
            records_dict[md.name] = pd.DataFrame(
                columns=generate_unique_labels(md.domain) + attributes
            )
            continue

        # fastpath for scalar symbols
        if md.dimension == 0 and md.number_records == 1:
            symptr = gmd.gmdGetSymbolByNumberPy(load_from, md.gmd_symbol_number, rc)
            recptr = gmd.gmdFindFirstRecordPy(load_from, symptr, rc)

            ret = gmd.gmdGetLevel(load_from, recptr)
            level = ret[1]
            marginal, lower, upper, scale = 0.0, 0.0, 0.0, 0.0

            if md.type in (gmd.GMS_DT_VAR, gmd.GMS_DT_EQU):
                ret = gmd.gmdGetMarginal(load_from, recptr)
                marginal = ret[1]
                ret = gmd.gmdGetLower(load_from, recptr)
                lower = ret[1]
                ret = gmd.gmdGetUpper(load_from, recptr)
                upper = ret[1]
                ret = gmd.gmdGetScale(load_from, recptr)
                scale = ret[1]

            vals = [level, marginal, lower, upper, scale]

            records_dict[md.name] = pd.DataFrame(
                [vals[: len(attributes)]],
                columns=attributes,
                dtype=float,
            )
        else:
            try:
                (
                    arrkeys,
                    arrvals,
                    unique_uels,
                ) = gams2np.gmdReadSymbolCat(
                    load_from,
                    md.name,
                    GMD_UELS,
                    encoding=encoding,
                )
            except Exception as err:
                raise GmdException(
                    f"Could not properly read symbol {md.name} from GMD object. "
                ) from err

            df = convert_to_categoricals_cat(arrkeys, arrvals, unique_uels)
            df.columns = generate_unique_labels(md.domain) + attributes
            records_dict[md.name] = df

    gmd.delete_intp(rc)

    # reset GMD special values
    gmd.gmdSetSpecialValues(load_from, default_special_vaules)

    return records_dict


def get_variable_equation_names(load_from) -> list[str]:
    """
    Return the names of all variable and equation symbols in a GMD handle.

    A frozen-model solve only updates variable/equation attributes (levels,
    marginals) and the objective; parameters and sets are inputs. The model
    instance uses this to restrict its post-solve read-back to the symbols a
    solve actually changes.
    """
    rc = gmd.new_intp()
    sym_count = gmd.gmdInfo(load_from, gmd.GMD_NRSYMBOLSWITHALIAS)[1]

    names: list[str] = []
    for i in range(1, sym_count + 1):
        symptr = gmd.gmdGetSymbolByNumberPy(load_from, i, rc)
        sym_type = gmd.gmdSymbolType(load_from, symptr)[1]
        if sym_type in (gmd.GMS_DT_VAR, gmd.GMS_DT_EQU):
            names.append(gmd.gmdSymbolInfo(load_from, symptr, gmd.GMD_NAME)[3])

    gmd.delete_intp(rc)
    return names


def gmd_get_metadata_by_number(
    gams2np: Gams2Numpy, load_from, symbol_number: int, encoding: str | None, status
) -> SymbolMetadata:
    symptr = gmd.gmdGetSymbolByNumberPy(load_from, symbol_number, status)
    ret = gmd.gmdSymbolInfo(load_from, symptr, gmd.GMD_NAME)
    syid = ret[3]
    ret = gmd.gmdSymbolInfo(load_from, symptr, gmd.GMD_USERINFO)
    userinfo = ret[1]
    ret = gmd.gmdSymbolInfo(load_from, symptr, gmd.GMD_DIM)
    dimen = ret[1]
    ret = gmd.gmdSymbolInfo(load_from, symptr, gmd.GMD_NRRECORDS)
    nrecs = ret[1]
    expltxt = gams2np._gmdGetSymbolExplTxt(load_from, symptr, encoding=encoding)
    ret = gmd.gmdGetDomain(load_from, symptr, dimen)
    domains_as_ptrs = ret[1]
    domain = ret[2]
    ret = gmd.gmdSymbolType(load_from, symptr)
    typ = ret[1]

    if typ == gmd.GMS_DT_ALIAS:
        symptr = gmd.gmdFindSymbolPy(load_from, syid, status)
        ret = gmd.gmdSymbolInfo(load_from, symptr, gmd.GMD_NAME)
        parent_set = ret[3]
    else:
        parent_set = None

    return SymbolMetadata(
        name=syid,
        gmd_symbol_number=symbol_number,
        dimension=dimen,
        type=typ,
        userinfo=userinfo,
        number_records=nrecs,
        description=expltxt,
        link_domains=[i is not None for i in domains_as_ptrs],
        domains_as_ptrs=domains_as_ptrs,
        domain=domain,
        parent_set=parent_set,
    )


def _set_special_values(write_to, *, eps_to_zero: bool):
    """
    Returns the special values that were set on the handle before this call so
    they can be restored with `_reset_special_values`.
    """
    default_special_values = gmd.doubleArray(gmd.GMS_SVIDX_MAX)
    gmd.gmdGetUserSpecialValues(write_to, default_special_values)

    spec_values = gmd.doubleArray(gmd.GMS_SVIDX_MAX)
    spec_values[gmd.GMS_SVIDX_UNDEF] = SpecialValues.UNDEF
    spec_values[gmd.GMS_SVIDX_NA] = SpecialValues.NA
    spec_values[gmd.GMS_SVIDX_EPS] = (
        SpecialValues.EPS if not eps_to_zero else gmd.GMS_SV_EPS
    )
    spec_values[gmd.GMS_SVIDX_PINF] = SpecialValues.POSINF
    spec_values[gmd.GMS_SVIDX_MINF] = SpecialValues.NEGINF

    ret = gmd.gmdSetSpecialValues(write_to, spec_values)
    assert ret

    return default_special_values


def _reset_special_values(write_to, default_special_values) -> None:
    """Restore the special values returned by a prior `_set_special_values` call."""
    gmd.gmdSetSpecialValues(write_to, default_special_values)


def _param_keys_values_major(
    records: DataFrame, dimension: int
) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
    """
    Build categorical keys, values, and the per-dimension UEL label list
    """
    nrecs = len(records)

    arrkeys = np.empty(dimension * nrecs, dtype=int)
    major: list[list[str]] = []
    for i in range(dimension):
        column = records.iloc[:, i]
        if not isinstance(column.dtype, pd.CategoricalDtype):
            column = column.astype("category")

        arrkeys[i * nrecs : (i + 1) * nrecs] = column.cat.codes
        major.append(column.cat.categories.tolist())

    arrkeys = arrkeys.reshape((nrecs, dimension), order="F")
    arrvals = records.iloc[:, -1].to_numpy().reshape((-1, 1)).astype(float)

    return arrkeys, arrvals, major


def update_parameter_records(
    gams2np: Gams2Numpy,
    gmd_handle,
    sym_ptr,
    records: DataFrame | None,
    dimension: int,
    *,
    eps_to_zero: bool = False,
) -> None:
    """
    Replace the records of an existing GMD parameter symbol from a DataFrame.
    """
    default_special_values = _set_special_values(gmd_handle, eps_to_zero=eps_to_zero)

    # Drop records from a previous solve so the symbol is fully replaced.
    gmd.gmdClearSymbol(gmd_handle, sym_ptr)

    try:
        if records is None or len(records) == 0:
            return

        if dimension == 0:
            rc = gmd.new_intp()
            recptr = gmd.gmdAddRecordPy(gmd_handle, sym_ptr, [], rc)
            value = float(records.to_numpy().reshape((-1,))[0])
            if eps_to_zero and value == 0:
                value = 0.0
            gmd.gmdSetLevel(gmd_handle, recptr, value)
            gmd.delete_intp(rc)
            return

        arrkeys, arrvals, major = _param_keys_values_major(records, dimension)

        # Categorical fill requires UELs to be registered beforehand.
        uels = list(dict.fromkeys(uel for labels in major for uel in labels))
        gams2np.gmdRegisterUels(gmd_handle, uels)

        gams2np.gmdFillSymbolCat(
            gmd_handle,
            sym_ptr,
            arrkeys,
            arrvals,
            major,
            merge=True,
            relaxedType=False,
            epsToZero=eps_to_zero,
        )
    finally:
        _reset_special_values(gmd_handle, default_special_values)
