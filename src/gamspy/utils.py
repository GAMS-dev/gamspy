from __future__ import annotations

import os
import platform
from collections.abc import Sequence
from typing import TYPE_CHECKING, Iterable

import gams.transfer as gt
from gams.core import gdx

import gamspy._symbols.implicits as implicits
from gamspy.exceptions import GamspyException, ValidationError

if TYPE_CHECKING:
    import pandas as pd
    from gams.core.numpy import Gams2Numpy

    from gamspy import Alias, Equation, Set, Variable
    from gamspy._symbols.implicits import ImplicitSet

SPECIAL_VALUE_MAP = {
    gt.SpecialValues.NA: "NA",
    gt.SpecialValues.UNDEF: "UNDF",
    gt.SpecialValues.POSINF: "INF",
    gt.SpecialValues.NEGINF: "-INF",
}


def getInstalledSolvers() -> list[str]:
    """
    Returns the list of installed solvers

    Returns
    -------
    List[str]

    Raises
    ------
    ModuleNotFoundError
        In case gamspy_base is not installed.

    Examples
    --------
    >>> import gamspy.utils as utils
    >>> installed_solvers = utils.getInstalledSolvers()

    """
    try:
        import gamspy_base
    except ModuleNotFoundError as e:
        e.msg = "You must first install gamspy_base to use this functionality"
        raise e

    solver_names = []
    capabilities_file = {"Windows": "gmscmpNT.txt", "rest": "gmscmpun.txt"}
    user_platform = "Windows" if platform.system() == "Windows" else "rest"

    with open(
        gamspy_base.directory + os.sep + capabilities_file[user_platform],
        encoding="utf-8",
    ) as capabilities:
        for line in capabilities:
            if line == "DEFAULTS\n":
                break

            if line.isupper():
                solver_names.append(line.split(" ")[0])

    try:
        solver_names.remove("SCENSOLVER")
    except ValueError:
        pass

    try:
        solver_names.remove("GUSS")
    except ValueError:
        pass

    return sorted(solver_names)


def getAvailableSolvers() -> list[str]:
    """
    Returns all available solvers that can be installed.

    Returns
    -------
    List[str]

    Raises
    ------
    ModuleNotFoundError
        In case gamspy_base is not installed.

    Examples
    --------
    >>> import gamspy.utils as utils
    >>> available_solvers = utils.getAvailableSolvers()

    """
    try:
        import gamspy_base
    except ModuleNotFoundError as e:
        e.msg = "You must first install gamspy_base to use this functionality"
        raise e

    return sorted(gamspy_base.available_solvers)


def checkAllSame(iterable1: Sequence, iterable2: Sequence) -> bool:
    """
    Checks if all elements of a sequence are equal to the all
    elements of another sequence.

    Parameters
    ----------
    iterable1 : Sequence of Symbols
    iterable2 : Sequence of Symbols

    Returns
    -------
    bool

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i")
    >>> j = gp.Set(m, "j")
    >>> k = gp.Set(m, "k")
    >>> list1 = [i, j]
    >>> list2 = [i, j]
    >>> gp.utils.checkAllSame(list1, list2)
    True
    >>> list3 = [i, j, k]
    >>> gp.utils.checkAllSame(list1, list3)
    False

    """
    if len(iterable1) != len(iterable2):
        return False

    all_same = True
    for elem1, elem2 in zip(iterable1, iterable2):
        if elem1 is not elem2:
            return False
    return all_same


def isin(symbol, sequence: Sequence) -> bool:
    """
    Checks whether the given symbol in the sequence.
    Needed for symbol comparison since __eq__ magic
    is overloaded by the symbols.

    Parameters
    ----------
    symbol : Symbol
        Symbol to check
    sequence : Sequence
        Sequence that holds a sequence of symbols

    Returns
    -------
    bool

    Examples
    --------
    >>> import gamspy as gp
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
    return any(symbol is item for item in sequence)


def _get_scalar_data(gams2np: Gams2Numpy, gdx_handle, symbol_id: str) -> float:
    _, arrvals = gams2np.gdxReadSymbolRaw(gdx_handle, symbol_id)
    return arrvals[0][0]


def _calculate_infeasibilities(symbol: Variable | Equation) -> pd.DataFrame:
    records = symbol.records
    infeas_rows = records.where(
        (records["level"] < records["lower"])
        | (records["level"] > records["upper"])
    ).dropna()
    lower_distance = (infeas_rows["lower"] - infeas_rows["level"]).abs()
    upper_distance = (infeas_rows["upper"] - infeas_rows["level"]).abs()
    infeas = lower_distance.combine(upper_distance, min)
    infeas_rows["infeasibility"] = infeas

    return infeas_rows


def _filter_gams_string(raw_string: str) -> str:
    FILTERS = (
        "$onMultiR",
        "$onUNDF",
        "$gdxIn",
        "$loadDC",
        "$offUNDF",
        "execute_unload",
        "Parameter autogen",
        "autogen",
        "$offListing",
        "$onListing",
    )
    filtered_lines = [
        line for line in raw_string.split("\n") if not line.startswith(FILTERS)
    ]
    return "\n".join(filtered_lines)


def _get_gamspy_base_directory() -> str:
    """
    Returns the gamspy_base directory.

    Returns
    -------
    str
        System directory
    """
    try:
        import gamspy_base
    except ModuleNotFoundError as e:
        e.msg = "You must first install gamspy_base to use this functionality"
        raise e

    gamspy_base_directory = gamspy_base.__path__[0]
    return gamspy_base_directory


def _get_license_path(system_directory: str) -> str:
    user_license_path = os.path.join(system_directory, "user_license.txt")
    if os.path.exists(user_license_path):
        return user_license_path

    return os.path.join(system_directory, "gamslice.txt")


def _close_gdx_handle(handle):
    """
    Closes the handle and unloads the gdx library.

    Parameters
    ----------
    handle : gdx_handle
    """
    gdx.gdxClose(handle)
    gdx.gdxFree(handle)
    gdx.gdxLibraryUnload()


def _replace_equality_signs(string: str) -> str:
    string = string.replace("=l=", "<=")
    string = string.replace("=e=", "eq")
    string = string.replace("=g=", ">=")
    return string


def _open_gdx_file(system_directory: str, load_from: str):
    """
    Opens the gdx file with given path

    Parameters
    ----------
    system_directory : str
    load_from : str

    Returns
    -------
    gdx_handle

    Raises
    ------
    Exception
        Exception while creating the handle or setting the special values
    """
    try:
        gdx_handle = gdx.new_gdxHandle_tp()
        rc = gdx.gdxCreateD(gdx_handle, system_directory, gdx.GMS_SSSIZE)
        assert rc[0], rc[1]
    except AssertionError as e:
        raise GamspyException("GAMSPy could not create gdx handle.") from e

    try:
        rc = gdx.gdxOpenRead(gdx_handle, load_from)
        assert rc[0]
    except AssertionError as e:
        raise GamspyException(
            "GAMSPy could not open gdx file to read from."
        ) from e

    return gdx_handle


def _to_list(obj: Set | Alias | str | tuple | ImplicitSet) -> list:
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


def _map_special_values(value: float):
    if gt.SpecialValues.isEps(value):
        return "EPS"

    if value in SPECIAL_VALUE_MAP:
        return SPECIAL_VALUE_MAP[value]

    return value


def _get_domain_str(domain: Iterable[Set | Alias | ImplicitSet | str]) -> str:
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
            raise ValidationError(
                f"Domain type must be str, Set or Alias but found {type(set)}"
            )

    return "(" + ",".join(set_strs) + ")"


def _get_matching_paranthesis_indices(string: str) -> int:
    """
    Stack based paranthesis matcher.

    Parameters
    ----------
    string : str

    Returns
    -------
    int

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
            except IndexError as e:
                raise AssertionError("Too many closing parentheses!") from e

    if stack:
        raise AssertionError("Too many opening parentheses!")

    return matching_indices[0]
