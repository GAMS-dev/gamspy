from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast, no_type_check

import numpy as np
import pandas as pd
from gams.core import gdx

import gamspy.utils as utils
from gamspy._algorithms import (
    convert_to_categoricals_cat,
    generate_unique_labels,
    get_keys_and_values,
)
from gamspy._internals import (
    ATTR_PREFIX,
    GAMS_EQUATION_SUBTYPES,
    GAMS_VARIABLE_SUBTYPES,
    DomainStatus,
)
from gamspy._special_values import SpecialValues
from gamspy._symbols.base import BaseSymbol
from gamspy.exceptions import GdxException, ValidationError

if TYPE_CHECKING:
    import os
    from collections.abc import Iterable

    from gams.core.numpy import Gams2Numpy
    from pandas import DataFrame

    from gamspy._container import Container
    from gamspy._symbols import Equation, Parameter, Set, Variable
    from gamspy._types import SymbolWithRecordsType

cached_system_directory: str | None = None
GAMS_DOMAIN_STATUS = {1: "none", 2: "relaxed", 3: "regular"}


@dataclass(slots=True)
class GDXSymbolMetadata:
    name: str
    gdx_symbol_number: int
    dimension: int
    type: int
    userinfo: int
    number_records: int
    description: str
    domain_type: int
    domain: list[str]
    parent_set: str | None = None


@contextlib.contextmanager
def open_gdx(
    system_directory: str,
    file_path: str | os.PathLike,
    mode: str = "r",
    *,
    compress: bool = False,
):
    """Context manager to safely create, open, and close a GDX handle for reading or writing."""
    global cached_system_directory
    if cached_system_directory != system_directory:
        gdx.gdxLibraryUnload()
        cached_system_directory = system_directory

    gdx_handle = gdx.new_gdxHandle_tp()
    rc, msg = gdx.gdxCreateD(gdx_handle, system_directory, gdx.GMS_SSSIZE)
    if not rc:
        raise GdxException(f"Could not properly load GDX DLL: {msg}")

    try:
        if mode == "r":
            if not gdx.gdxOpenRead(gdx_handle, file_path):
                raise GdxException(f"Error opening GDX file `{file_path}` for reading.")
            yield gdx_handle
        elif mode == "w":
            if not compress:
                if not gdx.gdxOpenWrite(gdx_handle, file_path, "GAMS Transfer")[0]:
                    raise GdxException(f"Error opening GDX `{file_path}` for writing.")
            else:
                if not gdx.gdxOpenWriteEx(gdx_handle, file_path, "GAMS Transfer", 1)[0]:
                    raise GdxException(
                        f"Error opening GDX (w/compression) `{file_path}` for writing."
                    )

            yield gdx_handle
    except Exception as e:
        gdx.gdxClose(gdx_handle)
        gdx.gdxFree(gdx_handle)
        raise GdxException(
            f"There was a problem while opening the gdx file: {e}"
        ) from e
    else:
        gdx.gdxClose(gdx_handle)
        gdx.gdxFree(gdx_handle)


def _get_model_attr_records(
    container: Container, file_path: str, symbol_names: list[str]
) -> dict[str, float]:
    records: dict[str, float] = {}
    with open_gdx(container.system_directory, file_path) as handle:
        for name in symbol_names:
            _, arrvals = container._gams2np.gdxReadSymbolRaw(handle, name)
            records[name] = float(arrvals[0][0])

    return records


def _get_symbol_names_from_gdx(
    system_directory: str, load_from: str, symbol_type: int | None = None
) -> list[str]:
    with open_gdx(system_directory, load_from) as gdx_handle:
        _, symbol_count, _ = gdx.gdxSystemInfo(gdx_handle)

        symbol_names = []
        for i in range(1, symbol_count + 1):
            _, name, _, sym_type = gdx.gdxSymbolInfo(gdx_handle, i)
            if name.startswith(ATTR_PREFIX):
                continue

            if symbol_type is None or symbol_type == sym_type:
                symbol_names.append(name)

    return symbol_names


def _set_special_values(gdx_handle, *, eps_to_zero: bool = False) -> None:
    """Sets GAMS special values (e.g., NA, EPS, INF) for the GDX handle."""
    specVals = gdx.doubleArray(gdx.GMS_SVIDX_MAX)
    specVals[gdx.GMS_SVIDX_UNDEF] = SpecialValues.UNDEF
    specVals[gdx.GMS_SVIDX_NA] = SpecialValues.NA
    specVals[gdx.GMS_SVIDX_EPS] = gdx.GMS_SV_EPS if eps_to_zero else SpecialValues.EPS
    specVals[gdx.GMS_SVIDX_PINF] = SpecialValues.POSINF
    specVals[gdx.GMS_SVIDX_MINF] = SpecialValues.NEGINF

    if not gdx.gdxSetSpecialValues(gdx_handle, specVals):
        raise GdxException("Failed to set special values in GDX handle.")


def _fetch_metadata(
    gams2np: Gams2Numpy, gdx_handle, symbols: Iterable[str] | None, encoding: str | None
) -> list[GDXSymbolMetadata]:
    """Fetches and sorts symbol metadata from the GDX file."""
    if symbols is not None:
        metadata = gdx_get_metadata_by_names(gams2np, gdx_handle, symbols, encoding)
        # Sort symbols by GDX number to read them in the native GDX order
        return sorted(metadata, key=lambda x: x.gdx_symbol_number)

    _, symCount, _ = gdx.gdxSystemInfo(gdx_handle)
    return [
        gdx_get_metadata_by_number(gams2np, gdx_handle, i, encoding)
        for i in range(1, symCount + 1)
    ]


def load_missing_symbols(
    container: Container,
    load_from: str | os.PathLike,
    symbol_names: list[str] | dict[str, str],
    *,
    declare_in_gams: bool = True,
) -> None:
    from gamspy._symbols import Equation, Variable

    with open_gdx(container.system_directory, load_from, mode="r") as handle:
        _set_special_values(handle)
        metadata = gdx_get_metadata_by_names(container._gams2np, handle, symbol_names)

    if isinstance(symbol_names, dict):
        for gdx_name, md in zip(symbol_names, metadata):
            gamspy_name = symbol_names[gdx_name]
            md.name = gamspy_name

    metadata = sorted(metadata, key=lambda x: x.gdx_symbol_number)
    for md in metadata:
        if md.name not in container._data:
            create_symbol_from_metadata(container, md, declare_in_gams=declare_in_gams)
            domain = md.domain
            for n, elem in enumerate(domain):
                if elem != "*" and elem in container._data:
                    domain[n] = container._data[elem]  # ty: ignore[invalid-assignment]

            container._data[md.name]._domain = domain  # ty: ignore[invalid-assignment]

        symbol = container._data[md.name]
        if isinstance(symbol, (Variable, Equation)):
            symbol._update_attr_domains()


def get_records(
    container: Container,
    load_from: str | os.PathLike,
    symbols: list[str] | dict[str, str],
    encoding: str | None = None,
) -> dict[str, DataFrame]:
    records_dict = {}
    mapping = {s: s for s in symbols} if isinstance(symbols, list) else symbols
    gdx_symbol_names = list(mapping.keys())

    with open_gdx(container.system_directory, load_from, mode="r") as gdx_handle:
        _set_special_values(gdx_handle)

        symbol_metadata = _fetch_metadata(
            container._gams2np, gdx_handle, gdx_symbol_names, encoding
        )

        gdx_uels = container._gams2np.gdxGetUelList(gdx_handle, encoding=encoding)
        gdx_uels[0] = "*"

        for md in symbol_metadata:
            if md.type == gdx.GMS_DT_ALIAS:
                records_dict[md.name] = None
                continue

            if md.number_records == 0:
                records_dict[md.name] = None
                continue

            gamspy_name = mapping.get(md.name, md.name)
            symobj = cast(
                "Set | Parameter | Variable | Equation | None",
                container._data.get(gamspy_name),
            )

            # Retrieve attributes/domain_names to correctly format DataFrame columns
            if symobj is None:
                if md.type == gdx.GMS_DT_PAR:
                    attributes = ["value"]
                elif md.type in (gdx.GMS_DT_VAR, gdx.GMS_DT_EQU):
                    attributes = ["level", "marginal", "lower", "upper", "scale"]
                elif md.type == gdx.GMS_DT_SET:
                    attributes = ["element_text"]
                else:
                    attributes = []
                domain_names = md.domain
            else:
                attributes = symobj._attributes
                domain_names = symobj.domain_names

            # Fastpath for scalar symbols
            if md.dimension == 0 and md.number_records == 1:
                gdx.gdxDataReadRawStart(gdx_handle, md.gdx_symbol_number)
                _, _, vals, _ = gdx.gdxDataReadRaw(gdx_handle)
                gdx.gdxDataReadDone(gdx_handle)

                df = pd.DataFrame(
                    [vals[: len(attributes)]],
                    columns=attributes,
                    dtype=float,
                )
                records_dict[gamspy_name] = df
            else:
                try:
                    arrkeys, arrvals, unique_uels = container._gams2np.gdxReadSymbolCat(
                        gdx_handle, md.name, gdx_uels, encoding=encoding
                    )
                except Exception as err:
                    raise GdxException(
                        f"Could not properly read symbol `{md.name}` from GDX file: {err}"
                    ) from err

                df = convert_to_categoricals_cat(arrkeys, arrvals, unique_uels)

                if df is not None:
                    df.columns = generate_unique_labels(domain_names) + attributes

                records_dict[gamspy_name] = df

    return records_dict


def read(
    container: Container,
    load_from: str | os.PathLike,
    symbols: list[str] | None,
    records: bool,
    encoding: str | None,
) -> None:
    from gamspy._symbols import Equation, Parameter, Set, Variable

    with open_gdx(container.system_directory, load_from, mode="r") as gdx_handle:
        _set_special_values(gdx_handle)

        symbol_metadata = _fetch_metadata(
            container._gams2np, gdx_handle, symbols, encoding
        )

        link_domains: list[GDXSymbolMetadata] = []
        symbols_with_records = []
        read_symbols = {md.name for md in symbol_metadata}

        # Validate and Create Symbols
        for md in symbol_metadata:
            if md.name in container._data:
                raise ValidationError(
                    f"Attempting to create a new symbol (through a read operation) named `{md.name}` "
                    "but an object with this name already exists in the Container."
                )

            if GAMS_DOMAIN_STATUS.get(md.domain_type) == "regular":
                link_domains.append(md)

            if md.number_records > 0 and md.type != gdx.GMS_DT_ALIAS:
                symbols_with_records.append(md)

            create_symbol_from_metadata(container, md)

        # Link Domain Objects
        for md in link_domains:
            symbol = container._data[md.name]
            if not isinstance(symbol, (Set, Parameter, Variable, Equation)):
                continue

            domain = md.domain
            for n, dom_name in enumerate(domain):
                if dom_name != "*" and dom_name in read_symbols:
                    domain[n] = container._data[dom_name]  # ty: ignore[invalid-assignment]

            symbol._domain = domain

            if isinstance(symbol, (Variable, Equation)):
                symbol._update_attr_domains()

        # Main Records Read
        if not records:
            return

        gdx_uels = container._gams2np.gdxGetUelList(gdx_handle, encoding=encoding)
        gdx_uels[0] = "*"

        for md in symbols_with_records:
            symobj = cast("SymbolWithRecordsType", container._data[md.name])

            # Fastpath for scalar symbols
            if md.dimension == 0 and md.number_records == 1:
                gdx.gdxDataReadRawStart(gdx_handle, md.gdx_symbol_number)
                _, _, vals, _ = gdx.gdxDataReadRaw(gdx_handle)
                gdx.gdxDataReadDone(gdx_handle)

                symobj._records = pd.DataFrame(
                    [vals[: len(symobj._attributes)]],
                    columns=symobj._attributes,
                    dtype=float,
                )
            else:
                try:
                    arrkeys, arrvals, unique_uels = container._gams2np.gdxReadSymbolCat(
                        gdx_handle, md.name, gdx_uels, encoding=encoding
                    )
                except Exception as err:
                    raise GdxException(
                        f"Could not properly read symbol `{md.name}` from GDX file: {err}"
                    ) from err

                df = convert_to_categoricals_cat(arrkeys, arrvals, unique_uels)
                symobj._records = df
                if symobj._records is not None:
                    symobj._records.columns = (
                        generate_unique_labels(symobj.domain_names) + symobj._attributes
                    )

            symobj._should_unload_to_gams = True
            symobj._should_load_from_gams = False


# TODO: fix typing here.
@no_type_check
def create_symbol_from_metadata(
    container: Container, metadata: GDXSymbolMetadata, *, declare_in_gams: bool = True
) -> None:
    from gamspy._symbols import (
        Alias,
        Equation,
        Parameter,
        Set,
        UniverseAlias,
        Variable,
    )

    if metadata.type == gdx.GMS_DT_ALIAS:
        # test for universe alias
        if metadata.userinfo != 0:
            Alias._constructor_bypass(
                container,
                metadata.name,
                container._data[metadata.parent_set],
            )
        else:
            UniverseAlias._constructor_bypass(container, metadata.name)
    # regular set
    elif metadata.type == gdx.GMS_DT_SET and metadata.userinfo == 0:
        Set._constructor_bypass(
            container,
            metadata.name,
            metadata.domain,
            is_singleton=False,
            description=metadata.description,
        )
    # singleton set
    elif metadata.type == gdx.GMS_DT_SET and metadata.userinfo == 1:
        Set._constructor_bypass(
            container,
            metadata.name,
            metadata.domain,
            is_singleton=True,
            description=metadata.description,
        )
    elif metadata.type == gdx.GMS_DT_PAR:
        Parameter._constructor_bypass(
            container,
            metadata.name,
            metadata.domain,
            description=metadata.description,
        )
    elif metadata.type == gdx.GMS_DT_VAR:
        Variable._constructor_bypass(
            container,
            metadata.name,
            GAMS_VARIABLE_SUBTYPES.get(metadata.userinfo, "free"),
            metadata.domain,
            description=metadata.description,
        )
    elif metadata.type == gdx.GMS_DT_EQU:
        Equation._constructor_bypass(
            container,
            metadata.name,
            GAMS_EQUATION_SUBTYPES.get(metadata.userinfo, "eq"),
            metadata.domain,
            description=metadata.description,
        )
    else:
        raise GdxException(
            f"Unknown GDX symbol classification (GAMS Type= {metadata.type}, "
            f"GAMS Subtype= {metadata.userinfo}). ",
            f"Cannot load symbol `{metadata.name}`",
        )

    if not declare_in_gams:
        container._unsaved_statements.pop()


def gdx_get_metadata_by_number(
    gams2np: Gams2Numpy, gdx_handle, symbol_number: int, encoding: str | None = None
) -> GDXSymbolMetadata:
    """Retrieve metadata for a GDX symbol by its symbol number."""
    _, syid, dimen, typ = gdx.gdxSymbolInfo(gdx_handle, symbol_number)
    _, nrecs, userinfo, _ = gdx.gdxSymbolInfoX(gdx_handle, symbol_number)
    domain_type, domain = gdx.gdxSymbolGetDomainX(gdx_handle, symbol_number)
    expltxt = gams2np._gdxGetSymbolExplTxt(gdx_handle, symbol_number, encoding=encoding)

    # gdx specific adjustment for equations
    if typ == gdx.GMS_DT_EQU:
        userinfo = userinfo - gdx.GMS_EQU_USERINFO_BASE

    if typ == gdx.GMS_DT_ALIAS:
        _, parent_set, _, _ = gdx.gdxSymbolInfo(gdx_handle, userinfo)
    else:
        parent_set = None

    # special handling of i(i) -- convert to relaxed domain_type
    if dimen == 1 and syid == domain[0]:
        domain_type = 2

    return GDXSymbolMetadata(
        name=syid,
        gdx_symbol_number=symbol_number,
        dimension=dimen,
        type=typ,
        userinfo=userinfo,
        number_records=nrecs,
        description=expltxt,
        domain_type=domain_type,
        domain=domain,
        parent_set=parent_set,
    )


def gdx_get_metadata_by_names(
    gams2np: Gams2Numpy,
    gdx_handle,
    symbol_names: Iterable[str],
    encoding: str | None = None,
) -> Iterable[GDXSymbolMetadata]:
    """Retrieve metadata for GDX symbols by their names."""
    metadata = []
    for sym in symbol_names:
        _, symnr = gdx.gdxFindSymbol(gdx_handle, sym)
        if symnr == -1:
            raise ValueError(
                f"User specified to read symbol `{sym}`, "
                "but it does not exist in the GDX file."
            )

        _, syid, dimen, typ = gdx.gdxSymbolInfo(gdx_handle, symnr)
        _, nrecs, userinfo, _ = gdx.gdxSymbolInfoX(gdx_handle, symnr)
        domain_type, domain = gdx.gdxSymbolGetDomainX(gdx_handle, symnr)
        expltxt = gams2np._gdxGetSymbolExplTxt(gdx_handle, symnr, encoding=encoding)

        # gdx specific adjustment for equations
        if typ == gdx.GMS_DT_EQU:
            userinfo = userinfo - gdx.GMS_EQU_USERINFO_BASE

        if typ == gdx.GMS_DT_ALIAS:
            _, parent_set, _, _ = gdx.gdxSymbolInfo(gdx_handle, userinfo)
        else:
            parent_set = None

        # special handling of i(i) -- convert to relaxed domain_type
        if dimen == 1 and syid == domain[0]:
            domain_type = 2

        metadata.append(
            GDXSymbolMetadata(
                name=syid,
                gdx_symbol_number=symnr,
                dimension=dimen,
                type=typ,
                userinfo=userinfo,
                number_records=nrecs,
                description=expltxt,
                domain_type=domain_type,
                domain=domain,
                parent_set=parent_set,
            )
        )

    return metadata


def _validate_symbols(
    container: Container, symbols: list[str] | None
) -> tuple[list[str], list]:
    """Validates the input symbols and returns their names and objects."""
    if symbols is None:
        symbols = container.listSymbols()

    symobjs = container.getSymbols(symbols)
    symnames = [sym.name for sym in symobjs]

    container._assert_valid_records(symbols=symbols)

    return symnames, symobjs


@contextlib.contextmanager
def _temporarily_relax_domains(symobjs: list):
    """Temporarily relaxes symbol domains if dependent domains aren't being written."""
    was_relaxed = {}
    try:
        for symobj in symobjs:
            if symobj.domain_type == "regular" and any(
                not utils.isin(domsymobj, symobjs) for domsymobj in symobj.domain
            ):
                was_relaxed[symobj.name] = {"object": symobj, "domain": symobj.domain}
                # Relax the domain
                symobj._domain = [
                    dom.name if isinstance(dom, BaseSymbol) else dom
                    for dom in symobj.domain
                ]
        yield
    finally:
        # Ensure domains are restored in the Container even if writing fails
        for properties in was_relaxed.values():
            symobj = properties["object"]
            symobj._domain = properties["domain"]


def _write_symbols_to_gdx(
    gdx_handle, container: Container, symnames: list[str], symobjs: list
):
    """Iterates through symbols and executes the appropriate GDX write commands."""
    from gamspy._symbols import Alias, Equation, UniverseAlias

    AnyContainerAlias = (Alias, UniverseAlias)

    for symname, symobj in zip(symnames, symobjs, strict=True):
        if isinstance(symobj, AnyContainerAlias):
            if isinstance(symobj, UniverseAlias):
                gdx.gdxAddAlias(gdx_handle, "*", symname)
            else:
                gdx.gdxAddAlias(gdx_handle, symobj.alias_with.name, symname)
        else:
            # Adjust equation userinfo for GDX
            if isinstance(symobj, Equation):
                symobj._gams_subtype = symobj._gams_subtype + gdx.GMS_EQU_USERINFO_BASE

            if symobj.number_records == 0:
                power_write_symbol_no_records(gdx_handle, symobj)
            elif symobj.number_records == 1 and symobj.dimension == 0:
                power_write_symbol_scalar_record(gdx_handle, symobj)
            else:
                power_write_category(container, gdx_handle, symobj)

        if gdx.gdxDataErrorCount(gdx_handle) != 0:
            raise GdxException(
                f"Encountered data errors with symbol `{symname}`. "
                "Possible causes are from duplicate records and/or domain violations"
            )

        symobj._should_unload_to_gams = False


def write(
    container: Container,
    write_to: str | os.PathLike,
    symbols: list[str] | None,
    *,
    compress: bool,
    eps_to_zero: bool,
) -> None:
    symnames, symobjs = _validate_symbols(container, symbols)

    with open_gdx(
        container.system_directory, write_to, mode="w", compress=compress
    ) as gdx_handle:
        _set_special_values(gdx_handle, eps_to_zero=eps_to_zero)

        uels = container._getUELs(symbols=symnames)
        container._gams2np.gdxRegisterUels(gdx_handle, uels)

        with _temporarily_relax_domains(symobjs):
            _write_symbols_to_gdx(gdx_handle, container, symnames, symobjs)


def power_write_symbol_no_records(gdx_handle, symobj: SymbolWithRecordsType) -> None:
    gdx.gdxDataWriteStrStart(
        gdx_handle,
        symobj.name,
        symobj.description,
        symobj.dimension,
        symobj._gams_type,
        symobj._gams_subtype,
    )

    # define domain
    if symobj._domain_status is DomainStatus.regular:
        gdx.gdxSymbolSetDomain(gdx_handle, symobj.domain_names)
    elif symobj._domain_status is DomainStatus.relaxed:
        _, synr = gdx.gdxFindSymbol(gdx_handle, symobj.name)
        gdx.gdxSymbolSetDomainX(gdx_handle, synr, symobj.domain_names)

    gdx.gdxDataWriteDone(gdx_handle)


def power_write_symbol_scalar_record(gdx_handle, symobj: SymbolWithRecordsType) -> None:
    gdx.gdxDataWriteStrStart(
        gdx_handle,
        symobj.name,
        symobj.description,
        symobj.dimension,
        symobj._gams_type,
        symobj._gams_subtype,
    )

    vals = symobj.records.to_numpy().reshape((-1,))  # ty: ignore[unresolved-attribute]

    idx = np.arange(vals.size)
    arr = np.zeros(gdx.GMS_VAL_MAX, dtype=np.float64)
    arr[idx] = vals[idx]

    values = gdx.doubleArray(gdx.GMS_VAL_MAX)
    values[gdx.GMS_VAL_LEVEL] = arr[0]
    values[gdx.GMS_VAL_MARGINAL] = arr[1]
    values[gdx.GMS_VAL_LOWER] = arr[2]
    values[gdx.GMS_VAL_UPPER] = arr[3]
    values[gdx.GMS_VAL_SCALE] = arr[4]

    gdx.gdxDataWriteStr(gdx_handle, [], values)
    gdx.gdxDataWriteDone(gdx_handle)


def power_write_category(
    container: Container, gdx_handle, symobj: SymbolWithRecordsType
) -> None:
    from gamspy import Set

    majorList = [[]] * symobj.dimension
    for i in range(symobj.dimension):
        majorList[i] = symobj._getUELs(i)

    arrkeys, arrvals = get_keys_and_values(symobj)

    if not np.issubdtype(arrkeys.dtype, np.integer):
        arrkeys = arrkeys.astype(int)

    if not isinstance(symobj, Set) and not np.issubdtype(arrvals.dtype, np.floating):
        arrvals = arrvals.astype(float)

    # temporary adjustment to domain argument
    if symobj._domain_status is DomainStatus.regular:
        domain = symobj.domain_names
    elif symobj._domain_status is DomainStatus.relaxed:
        domain = ["*"] * symobj.dimension
    elif symobj._domain_status is DomainStatus.none:
        domain = None

    try:
        container._gams2np.gdxWriteSymbolCat(
            gdx_handle,
            symobj.name,
            symobj.description,
            symobj.dimension,
            symobj._gams_type,
            symobj._gams_subtype,
            arrkeys,
            arrvals,
            majorList,
            domain,
        )
    except Exception as err:
        raise GdxException(
            f"Error encountered when writing symbol `{symobj.name}`. "
            f"GDX Error: '{err}'. \n\n"
            "GDX file was not created successfully."
        ) from err

    # assign actual relaxed domain labels
    if symobj._domain_status is DomainStatus.relaxed:
        _, synr = gdx.gdxFindSymbol(gdx_handle, symobj.name)
        gdx.gdxSymbolSetDomainX(gdx_handle, synr, symobj.domain_names)
