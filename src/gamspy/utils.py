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
from __future__ import annotations

import os
import platform
from collections.abc import Sequence
from typing import Iterable
from typing import List
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union

import gams.transfer as gt
from gams.core import gdx

import gamspy._symbols.implicits as implicits
from gamspy.exceptions import GamspyException

if TYPE_CHECKING:
    from gamspy._symbols.implicits import ImplicitSet
    from gamspy import Alias, Set
    from gamspy import Domain
    from gamspy._algebra.expression import Expression


def getInstalledSolvers() -> List[str]:
    """
    Returns the list of installed solvers

    Returns
    -------
    List[str]

    Raises
    ------
    GamspyException
        In case gamspy_base is not installed.

    Examples
    --------
    >>> import gamspy.utils as utils
    >>> utils.getInstalledSolvers()
    ['CONOPT', 'CONVERT', 'CPLEX', 'NLPEC', 'PATH', 'SBB']

    """
    try:
        import gamspy_base
    except ModuleNotFoundError:
        raise GamspyException(
            "You must first install gamspy_base to use this functionality"
        )

    solver_names = []
    capabilities_file = {"Windows": "gmscmpNT.txt", "rest": "gmscmpun.txt"}
    user_platform = "Windows" if platform.system() == "Windows" else "rest"

    with open(
        gamspy_base.directory + os.sep + capabilities_file[user_platform]
    ) as capabilities:
        for line in capabilities:
            if line == "DEFAULTS\n":
                break

            if line.isupper():
                solver_names.append(line.split(" ")[0])

    solver_names.remove("SCENSOLVER")

    return solver_names


def getAvailableSolvers() -> List[str]:
    """
    Returns all available solvers that can be installed.

    Returns
    -------
    List[str]

    Raises
    ------
    GamspyException
        In case gamspy_base is not installed.

    Examples
    --------
    >>> import gamspy.utils as utils
    >>> available_solvers = utils.getAvailableSolvers()

    """
    try:
        import gamspy_base
    except ModuleNotFoundError:
        raise GamspyException(
            "You must first install gamspy_base to use this functionality"
        )
    return gamspy_base.available_solvers


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
    for item in sequence:
        if symbol is item:
            return True
    return False


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
    except ModuleNotFoundError:
        raise GamspyException(
            "You must first install gamspy_base to use this functionality"
        )

    gamspy_base_directory = gamspy_base.__path__[0]
    return gamspy_base_directory


def _reserved_check(word: str) -> str:
    reserved_words = [
        "abort",
        "acronym",
        "acronyms",
        "alias",
        "all",
        "and",
        "binary",
        "break",
        "card",
        "continue",
        "diag",
        "display",
        "do",
        "else",
        "elseif",
        "endfor",
        "endif",
        "endloop",
        "endwhile",
        "eps",
        "equation",
        "equations",
        "execute",
        "execute_load",
        "execute_loaddc",
        "execute_loadhandle",
        "execute_loadpoint",
        "execute_unload",
        "execute_unloaddi",
        "execute_unloadidx",
        "file",
        "files",
        "for",
        "free",
        "function",
        "functions",
        "gdxLoad",
        "if",
        "inf",
        "integer",
        "logic",
        "loop",
        "model",
        "models",
        "na",
        "negative",
        "nonnegative",
        "no",
        "not",
        "option",
        "options",
        "or",
        "ord",
        "parameter",
        "parameters",
        "positive",
        "prod",
        "put",
        "put_utility",
        "putclear",
        "putclose",
        "putfmcl",
        "puthd",
        "putheader",
        "putpage",
        "puttitle",
        "puttl",
        "repeat",
        "sameas",
        "sand",
        "scalar",
        "scalars",
        "semicont",
        "semiint",
        "set",
        "sets",
        "singleton",
        "smax",
        "smin",
        "solve",
        "sor",
        "sos1",
        "sos2",
        "sum",
        "system",
        "table",
        "tables",
        "then",
        "undf",
        "until",
        "variable",
        "variables",
        "while",
        "xor",
        "yes",
    ]

    if word.lower() in reserved_words:
        raise GamspyException(
            "Name cannot be one of the reserved words. List of reserved"
            f" words: {reserved_words}"
        )

    return word


def _close_gdx_handle(handle):
    """
    Closes the handle and unloads the gdx library.

    Parameters
    ----------
    handle : gdxHandle
    """
    gdx.gdxClose(handle)
    gdx.gdxFree(handle)
    gdx.gdxLibraryUnload()


def _replace_equality_signs(string: str) -> str:
    string = string.replace("=l=", "<=")
    string = string.replace("=e=", "=")
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
    except AssertionError:
        raise GamspyException("GAMSPy could not create gdx handle.")

    try:
        rc = gdx.gdxOpenRead(gdxHandle, load_from)
        assert rc[0]
    except AssertionError:
        raise GamspyException("GAMSPy could not open gdx file to read from.")

    return gdxHandle


def _to_list(
    obj: Union[
        Set,
        Alias,
        str,
        Tuple,
        Domain,
        Expression,
        list,
        ImplicitSet,
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


def _get_domain_str(
    domain: Iterable[Union[Set, Alias, ImplicitSet, str]]
) -> str:
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
    from gamspy._algebra.domain import DomainException

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


def _get_matching_paranthesis_indices(string: str) -> dict:
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
                raise AssertionError("Too many closing parentheses!")

    if stack:
        raise AssertionError("Too many opening parentheses!")

    return matching_indices
