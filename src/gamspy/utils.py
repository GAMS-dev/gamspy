#
# GAMS - General Algebraic Modeling System Python API
#
# Copyright (c) 2023 GAMS Development Corp. <support@gams.com>
# Copyright (c) 2023 GAMS Software GmbH <support@gams.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import os
import platform
import gamspy
import pandas as pd
import gams.transfer as gt
import gamspy._symbols._implicits as implicits
from gams.core import gdx
from gams.transfer._internals.specialvalues import SpecialValues
from collections.abc import Sequence
from typing import Tuple, Union, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from gamspy._symbols._implicits import ImplicitSet
    from gamspy import Alias, Set
    from gamspy import Domain
    from gamspy._algebra._expression import Expression


class GdxException(Exception):
    """Exception raised if there is a GDX related issue"""


def _convert_to_categoricals(arrkeys, arrvals, unique_uels):
    # Temporary function to be removed when Gams Transfer is updated.
    has_domains = arrkeys.size > 0
    has_values = arrvals.size > 0

    dfs = []
    if has_domains:
        dfs.append(pd.DataFrame(arrkeys))

    if has_values:
        dfs.append(pd.DataFrame(arrvals))

    if has_domains and has_values:
        df = pd.concat(dfs, axis=1, copy=False)
        df.columns = pd.RangeIndex(start=0, stop=len(df.columns))
    elif has_domains or has_values:
        df = dfs[0]
        df.columns = pd.RangeIndex(start=0, stop=len(df.columns))
    else:
        df = None  # pragma: no cover

    if has_domains:
        rk, ck = arrkeys.shape
        for i in range(ck):
            dtype = pd.CategoricalDtype(
                categories=unique_uels[i], ordered=True
            )
            df.isetitem(
                i, pd.Categorical(values=df[i], dtype=dtype, fastpath=True)
            )

    return df


def _loadPackageGlobals() -> None:  # pragma: no cover
    gamspy._order = 0  # type: ignore


def _getUniqueName() -> str:
    """
    Generates a unique name for elements with no name (e.g. Expressions).

    Returns
    -------
    str
        Unique name in string format
    """
    gamspy._order += 1  # type: ignore
    return str(gamspy._order)  # type: ignore


def _getSymbolData(gams2np, gdxHandle, symbol_id: str) -> pd.DataFrame:
    """
    Gets the data of single symbol from GDX.

    Parameters
    ----------
    gams2np : gams.numpy.Gams2Numpy
    gdxHandle : gdxHandle
    symbol_id : str

    Returns
    -------
    return pandas.DataFrame (w/categoricals) if records exist

    Raises
    -------
    Exception
        In case there is no such symbol in the gdx
    """
    try:
        arrkeys, arrvals, unique_uels = gams2np.gdxReadSymbolCat(
            gdxHandle, symbol_id
        )
        return _convert_to_categoricals(arrkeys, arrvals, unique_uels)
    except Exception as e:
        raise GdxException(
            f"No symbol with id {symbol_id} in the gdx file! Message: {e}"
        )


def _getSystemDirectory(system_directory: Optional[str] = None) -> str:
    """
    Finds the system directory. If no existing GAMS installation provided,
    returns minigams directory.

    Parameters
    ----------
    system_directory : str, optional

    Returns
    -------
    str
        System directory
    """
    if system_directory:
        return system_directory

    system_directory = os.path.dirname(__file__) + os.sep

    user_os = platform.system().lower()
    system_directory += "minigams" + os.sep + user_os

    if user_os == "darwin":
        system_directory += f"_{platform.machine()}"  # pragma: no cover

    return system_directory


def _closeGdxHandle(handle):
    """
    Closes the handle and unloads the gdx library.

    Parameters
    ----------
    handle : gdxHandle
    """
    gdx.gdxClose(handle)
    gdx.gdxFree(handle)
    gdx.gdxLibraryUnload()


def _replaceEqualitySigns(condition: str) -> str:
    condition = condition.replace("=l=", "<=")
    condition = condition.replace("=e=", "=")
    condition = condition.replace("=g=", ">=")
    return condition


def _set_special_values(gdxHandle):
    """
    Sets the special values

    Parameters
    ----------
    gdxHandle : gdxHandle

    Returns
    -------
    int
    """
    specVals = gdx.doubleArray(gdx.GMS_SVIDX_MAX)
    specVals[gdx.GMS_SVIDX_UNDEF] = SpecialValues.UNDEF
    specVals[gdx.GMS_SVIDX_NA] = SpecialValues.NA
    specVals[gdx.GMS_SVIDX_EPS] = SpecialValues.EPS
    specVals[gdx.GMS_SVIDX_PINF] = SpecialValues.POSINF
    specVals[gdx.GMS_SVIDX_MINF] = SpecialValues.NEGINF

    rc = gdx.gdxSetSpecialValues(gdxHandle, specVals)
    return rc


def _openGdxFile(system_directory: str, load_from: str):
    """
    Opens the gdx file with given path

    Parameters
    ----------
    system_directory : str
    load_from : str

    Returns
    -------
    gdxHandle

    Raises
    ------
    Exception
        Exception while creating the handle or setting the special values
    """
    try:
        gdxHandle = gdx.new_gdxHandle_tp()
        rc = gdx.gdxCreateD(gdxHandle, system_directory, gdx.GMS_SSSIZE)
        assert rc[0], rc[1]
    except AssertionError as e:
        raise GdxException(
            "Could not properly load GDX DLL, check system_directory"
            f" path. {e}"
        )

    try:
        rc = gdx.gdxOpenRead(gdxHandle, load_from)
        assert rc[0]

        rc = _set_special_values(gdxHandle)
        assert rc
    except AssertionError as e:
        raise GdxException(f"Could not open the GDX file at: {load_from}. {e}")

    return gdxHandle


def _toList(
    obj: Union[
        "Set",
        "Alias",
        str,
        Tuple,
        "Domain",
        "Expression",
        list,
        implicits.ImplicitSet,
    ]
) -> list:
    """
    Converts the given object to a list

    Parameters
    ----------
    obj : Set | Alias | str | tuple | list | Domain | Expression | ImplicitSet
        Object to be converted

    Returns
    -------
    list
    """
    if isinstance(obj, tuple):
        return list(obj)

    if not isinstance(obj, list):
        obj = [obj]
    return obj


def isin(symbol, sequence: Sequence) -> bool:
    """
    Checks whether the given symbol in the sequence.
    Needed for symbol comparison since __eq__ magic
    is overloaded by the symbols.

    Parameters
    ----------
    symbol : Symbol
        _Symbol to check
    sequence : Sequence
        Sequence that holds a sequence of symbols

    Returns
    -------
    bool

    Examples
    --------
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i")
    >>> j = gp.Set(m, "j")
    >>> k = gp.Set(m, "k")
    >>> sets = [i, j]
    >>> gp.utils.isin(i, sets)
    True
    >>> gp.utils.isin(k, sets)
    False
    """
    for item in sequence:
        if symbol is item:
            return True
    return False


def checkAllSame(iterable1: Sequence, iterable2: Sequence) -> bool:
    """
    Checks if all elements of a sequence are equal to the all
    elements of another sequence.

    Parameters
    ----------
    iterable1 : Sequence
    iterable2 : Sequence

    Returns
    -------
    bool

    Examples
    --------
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i")
    >>> j = gp.Set(m, "j")
    >>> k = gp.Set(m, "k")
    >>> list1 = [i, j]
    >>> list2 = [i, j]
    >>> utils.checkAllSame(list1, list2)
    True
    >>> list3 = [i, j, k]
    >>> utils.checkAllSame(list1, list3)
    False
    """
    if len(iterable1) != len(iterable2):
        return False

    all_same = True
    for elem1, elem2 in zip(iterable1, iterable2):
        if elem1 is not elem2:
            return False
    return all_same


def _getDomainStr(domain: Union["Set", "Alias", "ImplicitSet", str]) -> str:
    """
    Creates the string format of a given domain

    Parameters
    ----------
    domain : Set | Alias | ImplicitSet | str

    Returns
    -------
    str

    Raises
    ------
    Exception
        Given domain must contain only sets, aliases or strings
    """
    from gamspy._algebra._domain import DomainException

    set_strs = []
    for set in domain:
        if isinstance(set, (gt.Set, gt.Alias, implicits.ImplicitSet)):
            set_strs.append(set.gamsRepr())
        elif isinstance(set, str):
            if set == "*":
                set_strs.append(set)
            else:
                set_strs.append('"' + set + '"')
        else:
            raise DomainException(
                f"Domain type must be str, Set or Alias but found {type(set)}"
            )

    return "(" + ",".join(set_strs) + ")"


def _getMatchingParanthesisIndices(string: str) -> dict:
    """
    Stack based paranthesis matcher.

    Parameters
    ----------
    string : str

    Returns
    -------
    dict

    Raises
    ------
    Exception
        In case there are more closing paranthesis than opening parantheses
    Exception
        In case there are more opening paranthesis than closing parantheses
    """
    stack = []  # stack of indices of opening parentheses
    matching_indices = {}

    for index, character in enumerate(string):
        if character == "(":
            stack.append(index)
        if character == ")":
            try:
                matching_indices[stack.pop()] = index
            except IndexError:
                raise Exception("Too many closing parentheses!")

    if stack:
        raise Exception("Too many opening parentheses!")

    return matching_indices


def _getValidCommandlineOptions() -> List[str]:
    def lowercase(options):
        return [option.lower() for option in options]

    problem_options = [
        "LP",
        "MIP",
        "RMIP",
        "NLP",
        "MCP",
        "MPEC",
        "RMPEC",
        "CNS",
        "DNLP",
        "RMINLP",
        "MINLP",
        "QCP",
        "MIQCP",
        "RMIQCP",
        "EMP",
        "solver",
    ]

    no_value_options = [
        "DmpOpt",
        "DmpSym",
        "DmpUserSym",
        "Eject",
        "Measure",
        "MemoryStat",
        "SubSystems",
    ]

    string_options = [
        "ECImplicitLoad",
        "EpsToZero",
        "gdxUels",
        "SolPrint",
        "SolveOpt",
        "SysOut",
        "ZeroToEps",
    ]

    numeric_options = [
        "AsyncSolLst",
        "Bratio",
        "CheckErrorLevel",
        "Decimals",
        "DispWidth",
        "DomLim",
        "DualCheck",
        "FDDelta",
        "FDOpt",
        "ForLim",
        "HoldFixedAsync",
        "IDCProtect",
        "Integer1",
        "Integer2",
        "Integer3",
        "Integer4",
        "Integer5",
        "IntVarUp",
        "IterLim",
        "LimCol",
        "LimRow",
        "MaxGenericFiles",
        "MCPRHoldfx",
        "OptCA",
        "OptCR",
        "Profile",
        "ProfileTol",
        "Real1",
        "Real2",
        "Real3",
        "Real4",
        "Real5",
        "Reform",
        "ResLim",
        "SavePoint",
        "Seed",
        "SolSlack",
        "SolveLink",
        "strictSingleton",
        "Sys10",
        "Sys11",
        "Sys12",
        "Sys15",
        "Sys16",
        "Sys17",
        "Sys18",
        "Sys19",
        "Threads",
        "ThreadsAs",
    ]

    return (
        lowercase(problem_options)
        + lowercase(no_value_options)
        + lowercase(string_options)
        + lowercase(numeric_options)
    )


COMMANDLINE_OPTIONS = _getValidCommandlineOptions()


def _getDefaultSolvers():
    return {
        "LP": "HIGHS",
        "MIP": "HIGHS",
        "RMIP": "HIGHS",
        "NLP": "CONOPT",
        "MCP": "PATH",
        "MPEC": "NLPEC",
        "CNS": "CONOPT",
        "DNLP": "CONOPT",
        "RMINLP": "CONOPT",
        "MINLP": "SBB",
        "QCP": "CONOPT",
        "MIQCP": "SBB",
        "RMIQCP": "CONOPT",
        "EMP": "CONVERT",
    }


DEFAULT_SOLVERS = _getDefaultSolvers()
