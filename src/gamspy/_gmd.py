from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from gams.core import gmd

from gamspy._algorithms import (
    convert_to_categoricals_cat,
    generate_unique_labels,
    get_keys_and_values,
)
from gamspy._internals import (
    GAMS_EQUATION_SUBTYPES,
    GAMS_VARIABLE_SUBTYPES,
    DomainStatus,
)
from gamspy._special_values import SpecialValues
from gamspy.exceptions import GmdException

if TYPE_CHECKING:
    from gams.core.numpy import Gams2Numpy
    from pandas import DataFrame

    from gamspy._container import Container
    from gamspy._symbols import Alias, UniverseAlias


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


# TODO: legacy code. pay the technical debt
def read(
    container: Container,
    load_from,
    symbols: list[str] | None,
    encoding: str | None,
    *,
    load_records: bool,
):
    initially_empty_container = len(container) == 0

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

    # get number of symbols
    ret = gmd.gmdInfo(load_from, gmd.GMD_NRSYMBOLSWITHALIAS)
    symCount = ret[1]

    # find symbol metadata if not reading in all
    read_all_symbols = True
    SYMBOL_METADATA: list[SymbolMetadata] = []
    LINK_DOMAINS = []
    SYMBOLS_W_RECORDS = []

    rc = gmd.new_intp()

    if symbols is not None:
        read_all_symbols = False
        for sym in symbols:
            symptr = gmd.gmdFindSymbolWithAliasPy(load_from, sym, rc)
            if symptr is None:
                raise ValueError(
                    f"User specified to read symbol `{sym}`, "
                    "but it does not exist in the GMD object."
                )

            ret = gmd.gmdSymbolInfo(load_from, symptr, gmd.GMD_NUMBER)
            symnr = ret[1]
            SYMBOL_METADATA.append(
                gmd_get_metadata_by_number(
                    container._gams2np, load_from, symnr, encoding, rc
                )
            )

        # sort symbols by gmd number in order to read symbols in gmd order not user order
        SYMBOL_METADATA = sorted(SYMBOL_METADATA, key=lambda x: x.gmd_symbol_number)

    # two paths to creating symbol objects
    if read_all_symbols:
        # fastpath if reading in all symbols (by number)
        for i in range(1, symCount + 1):
            md = gmd_get_metadata_by_number(
                container._gams2np, load_from, i, encoding, rc
            )

            # capture metadata
            SYMBOL_METADATA.append(md)

            # check if symbol already exists in container
            if not initially_empty_container and md.name in container:
                raise GmdException(
                    f"Attempting to create a new symbol (through a read operation) named `{md.name}` "
                    "but an object with this name already exists in the Container. "
                )

            # capture symbols to link later
            if (
                any(i is not None for i in md.domains_as_ptrs)
                and md.type != gmd.GMS_DT_ALIAS
            ):
                LINK_DOMAINS.append(md)

            # capture symbols that have records (not aliases)
            if md.number_records > 0 and md.type != gmd.GMS_DT_ALIAS:
                SYMBOLS_W_RECORDS.append(md)

            # create the symbol object in the container
            create_symbol_from_metadata(container, md)
    else:
        for md in SYMBOL_METADATA:
            # check if symbol already exists in container
            if not initially_empty_container and md.name in container:
                raise GmdException(
                    f"Attempting to create a new symbol (through a read operation) named `{md.name}` "
                    "but an object with this name already exists in the Container. "
                )

            # capture symbols to link later
            if (
                any(i is not None for i in md.domains_as_ptrs)
                and md.type != gmd.GMS_DT_ALIAS
            ):
                LINK_DOMAINS.append(md)

            # capture symbols that have records (not aliases)
            if md.number_records > 0 and md.type != gmd.GMS_DT_ALIAS:
                SYMBOLS_W_RECORDS.append(md)

            # create the symbol object in the container
            create_symbol_from_metadata(container, md)

    # link domain objects
    READ_SYMBOLS = [md.name for md in SYMBOL_METADATA]
    for md in LINK_DOMAINS:
        domain = md.domain
        for n, d in enumerate(domain):
            if md.link_domains[n] and d != "*" and d in READ_SYMBOLS:
                domain[n] = container._data[d]

    # main records read
    if load_records:
        # get and store GMD_UELS
        GMD_UELS = container._gams2np.gmdGetUelList(load_from, encoding=encoding)
        GMD_UELS[0] = "*"

        for md in SYMBOLS_W_RECORDS:
            # get symbol object
            symobj = container._data[md.name]

            # fastpath for scalar symbols
            if md.dimension == 0 and md.number_records == 1:
                symptr = gmd.gmdGetSymbolByNumberPy(load_from, md.gmd_symbol_number, rc)
                recptr = gmd.gmdFindFirstRecordPy(load_from, symptr, rc)

                ret = gmd.gmdGetLevel(load_from, recptr)
                level = ret[1]

                marginal = 0.0
                lower = 0.0
                upper = 0.0
                scale = 0.0

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

                symobj._records = pd.DataFrame(
                    [vals[: len(symobj._attributes)]],
                    columns=symobj._attributes,
                    dtype=float,
                )

            else:
                try:
                    (
                        arrkeys,
                        arrvals,
                        unique_uels,
                    ) = container._gams2np.gmdReadSymbolCat(
                        load_from,
                        md.name,
                        GMD_UELS,
                        encoding=encoding,
                    )
                except Exception as err:
                    raise GmdException(
                        f"Could not properly read symbol {md.name} from GMD object. "
                    ) from err

                # convert to categorical dataframe (return None if no data)
                df = convert_to_categoricals_cat(arrkeys, arrvals, unique_uels)
                symobj._records = df
                symobj._records.columns = (
                    generate_unique_labels(symobj.domain_names) + symobj._attributes
                )

    gmd.delete_intp(rc)

    # reset GMD special values
    gmd.gmdSetSpecialValues(load_from, default_special_vaules)


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


def create_symbol_from_metadata(container: Container, metadata: SymbolMetadata):
    from gamspy._symbols import (
        Alias,
        Equation,
        Parameter,
        Set,
        UniverseAlias,
        Variable,
    )

    # create the symbols in the container
    if metadata.type == gmd.GMS_DT_ALIAS:
        # test for universe alias
        if metadata.userinfo > 0:
            try:
                Alias._constructor_bypass(
                    container, metadata.name, container._data[metadata.parent_set]
                )
            except Exception as err:
                raise GmdException(
                    f"Cannot create the Alias symbol `{metadata.name}` "
                    f"because the parent set (`{metadata.parent_set}`) is not "
                    "being read into the in the Container. "
                    "Alias symbols require the parent set object to exist in the Container. "
                    f"Add `{metadata.parent_set}` to the list of symbols to read."
                ) from err
        else:  # symbol number of universe (*) is 0
            UniverseAlias._constructor_bypass(container, metadata.name)

    # regular set
    elif metadata.type == gmd.GMS_DT_SET and metadata.userinfo == 0:
        Set._constructor_bypass(
            container,
            metadata.name,
            metadata.domain,
            is_singleton=False,
            description=metadata.description,
        )
    # singleton set
    elif metadata.type == gmd.GMS_DT_SET and metadata.userinfo == 1:
        Set._constructor_bypass(
            container,
            metadata.name,
            metadata.domain,
            is_singleton=True,
            description=metadata.description,
        )
    elif metadata.type == gmd.GMS_DT_PAR:
        Parameter._constructor_bypass(
            container,
            metadata.name,
            metadata.domain,
            description=metadata.description,
        )
    elif metadata.type == gmd.GMS_DT_VAR:
        Variable._constructor_bypass(
            container,
            metadata.name,
            GAMS_VARIABLE_SUBTYPES.get(metadata.userinfo, "free"),
            metadata.domain,
            description=metadata.description,
        )
    elif metadata.type == gmd.GMS_DT_EQU:
        Equation._constructor_bypass(
            container,
            metadata.name,
            GAMS_EQUATION_SUBTYPES.get(metadata.userinfo, "eq"),
            metadata.domain,
            description=metadata.description,
        )
    else:
        raise GmdException(
            f"Unknown GamsDatabase (GMD) symbol classification (GAMS Type= {metadata.type}, "
            f"GAMS Subtype= {metadata.userinfo}). ",
            f"Cannot load symbol `{metadata.name}`",
        )


def write(
    container: Container, write_to, symbols: list[str] | None, *, eps_to_zero: bool
):
    from gamspy._symbols import Alias, UniverseAlias

    write_all_symbols = symbols is None

    if write_all_symbols:
        symbols = container.listSymbols()

    # assert valid records
    container._assert_valid_records(symbols=symbols)

    # check if domain columns have valid categories
    symobjs = container.getSymbols(symbols)
    symnames = [sym.name for sym in symobjs]

    default_special_vaules = gmd.doubleArray(gmd.GMS_SVIDX_MAX)
    gmd.gmdGetUserSpecialValues(write_to, default_special_vaules)

    # setting special values
    specVals = gmd.doubleArray(gmd.GMS_SVIDX_MAX)
    specVals[gmd.GMS_SVIDX_UNDEF] = SpecialValues.UNDEF
    specVals[gmd.GMS_SVIDX_NA] = SpecialValues.NA
    specVals[gmd.GMS_SVIDX_EPS] = (
        SpecialValues.EPS if not eps_to_zero else gmd.GMS_SV_EPS
    )
    specVals[gmd.GMS_SVIDX_PINF] = SpecialValues.POSINF
    specVals[gmd.GMS_SVIDX_MINF] = SpecialValues.NEGINF

    ret = gmd.gmdSetSpecialValues(write_to, specVals)
    assert ret

    # get number of symbols
    ret = gmd.gmdInfo(write_to, gmd.GMD_NRSYMBOLSWITHALIAS)
    symCount = ret[1]

    # read in all symbol metadata from GMD
    rc = gmd.new_intp()
    GMD_SYMBOL_METADATA: list[SymbolMetadata] = []
    for symnr in range(1, symCount + 1):
        GMD_SYMBOL_METADATA.append(
            gmd_get_metadata_by_number(container._gams2np, write_to, symnr, None, rc)
        )
    GMD_SYMBOLS = [md.name for md in GMD_SYMBOL_METADATA]

    # casefold key symbol lists
    CF_GMD_SYMBOLS = list(map(str.casefold, GMD_SYMBOLS))
    CF_CONTAINER_SYMBOLS = list(map(str.casefold, symnames))

    # check if partial write is possible
    if not write_all_symbols and any(
        i not in CF_GMD_SYMBOLS for i in CF_CONTAINER_SYMBOLS
    ):
        raise GmdException(
            "Writing a subset of symbols from a Container is only enabled for "
            "symbols that currently exist in the GamsDatabase (GMD) object."
            "This restriction may be relaxed in a future release."
        )

    # register the universe
    # get UELS only once
    CONTAINER_UELS = container._getUELs(symbols=symbols)

    # register UELs
    for uel in CONTAINER_UELS:
        try:
            ret = gmd.gmdMergeUel(write_to, uel)
        except Exception as err:
            raise GmdException(
                f"Unable to register UEL `{uel}` to GMD. Reason: {err}"
            ) from err

    # main write
    for symobj in symobjs:
        # write any aliases
        AnyContainerAlias = (Alias, UniverseAlias)
        if isinstance(symobj, AnyContainerAlias):
            power_write_alias(write_to, symobj, rc)

        # all other symbols
        else:
            if symobj.number_records == 0:
                power_write_symbol_no_records(write_to, symobj, rc)

            elif symobj.number_records == 1 and symobj.dimension == 0:
                power_write_symbol_scalar_record(write_to, symobj, eps_to_zero, rc)

            else:
                power_write_category(
                    container._gams2np, write_to, symobj, eps_to_zero, rc
                )

    gmd.delete_intp(rc)

    # reset GMD special values
    gmd.gmdSetSpecialValues(write_to, default_special_vaules)


def power_write_alias(write_to, symobj: Alias | UniverseAlias, rc):
    from gamspy._symbols import Alias

    if isinstance(symobj, Alias):
        _, idx, _, _ = gmd.gmdSymbolInfo(
            write_to,
            gmd.gmdFindSymbolPy(write_to, symobj.alias_with.name, rc),
            gmd.GMD_NUMBER,
        )

        gmd.gmdAddSymbolXPy(
            write_to,
            symobj.name,
            symobj.dimension,
            gmd.GMS_DT_ALIAS,
            idx,
            f"Aliased with {symobj.alias_with.name}",
            [None] * symobj.dimension,
            [""] * symobj.dimension,
            rc,
        )
    else:
        # universe aliases
        gmd.gmdAddSymbolXPy(
            write_to,
            symobj.name,
            symobj.dimension,
            gmd.GMS_DT_ALIAS,
            0,  # universe (*) symbol number in GMD is 0
            f"Aliased with {symobj.alias_with}",
            [None] * symobj.dimension,
            [""] * symobj.dimension,
            rc,
        )


def power_write_symbol_no_records(write_to, symobj, rc):
    # get domain
    if symobj._domain_status is DomainStatus.regular:
        domain = []
        for d in symobj.domain_names:
            ret = gmd.gmdFindSymbolPy(write_to, d, rc)
            domain.append(ret)
    else:
        domain = [None] * symobj.dimension

    # create new symbol
    gmd.gmdAddSymbolXPy(
        write_to,
        symobj.name,
        symobj.dimension,
        symobj._gams_type,
        symobj._gams_subtype,
        symobj.description,
        domain,
        symobj.domain_names,
        rc,
    )


def power_write_symbol_scalar_record(write_to, symobj, eps_to_zero, rc):
    def write_scalar_record(write_to, recptr, symobj):
        vals = symobj.records.to_numpy().reshape((-1,))

        if eps_to_zero:
            vals = [0.0 if val == 0 else val for val in vals]

        if symobj._gams_type == gmd.GMS_DT_PAR:
            gmd.gmdSetLevel(write_to, recptr, vals[0])

        if symobj._gams_type in (gmd.GMS_DT_VAR, gmd.GMS_DT_EQU):
            gmd.gmdSetLevel(write_to, recptr, vals[0])
            gmd.gmdSetMarginal(write_to, recptr, vals[1])
            gmd.gmdSetLower(write_to, recptr, vals[2])
            gmd.gmdSetUpper(write_to, recptr, vals[3])
            gmd.gmdSetScale(write_to, recptr, vals[4])

    symptr = gmd.gmdAddSymbolXPy(
        write_to,
        symobj.name,
        symobj.dimension,
        symobj._gams_type,
        symobj._gams_subtype,
        symobj.description,
        [],
        symobj.domain_names,
        rc,
    )
    recptr = gmd.gmdAddRecordPy(write_to, symptr, [], rc)
    write_scalar_record(write_to, recptr, symobj)


def power_write_category(gams2np, write_to, symobj, eps_to_zero, rc):
    from gamspy._symbols import Set

    # initialize major list
    majorList = [[]] * symobj.dimension

    for i in range(symobj.dimension):
        # create major list
        majorList[i] = symobj._getUELs(i)

    # get keys and values arrays
    arrkeys, arrvals = get_keys_and_values(symobj)

    # final type checking
    if not np.issubdtype(arrkeys.dtype, np.integer):
        arrkeys = arrkeys.astype(int)

    if not isinstance(symobj, Set) and not np.issubdtype(arrvals.dtype, np.floating):
        arrvals = arrvals.astype(float)

    # get domain
    if symobj._domain_status is DomainStatus.regular:
        domain = []
        for d in symobj.domain_names:
            ret = gmd.gmdFindSymbolPy(write_to, d, rc)
            domain.append(ret)
    else:
        domain = [None] * symobj.dimension

    # create new symbol
    symptr = gmd.gmdAddSymbolXPy(
        write_to,
        symobj.name,
        symobj.dimension,
        symobj._gams_type,
        symobj._gams_subtype,
        symobj.description,
        domain,
        symobj.domain_names,
        rc,
    )

    # fill new symbol
    try:
        gams2np.gmdFillSymbolCat(
            write_to,
            symptr,
            arrkeys,
            arrvals,
            majorList,
            merge=True,
            relaxedType=False,
            epsToZero=eps_to_zero,
        )

    except Exception as err:
        # clear symbol records
        gmd.gmdClearSymbol(write_to, symptr)
        raise GmdException(
            f"Unable to successfully write symbol `{symobj.name}`.  Reason: {err}"
        ) from err
