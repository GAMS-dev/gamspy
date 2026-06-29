from __future__ import annotations

import base64
import inspect
import os
import platform
import uuid
from typing import TYPE_CHECKING, cast

import gamspy._gdx as gdxio
import gamspy._symbols as syms
import gamspy._symbols.implicits as implicits
import gamspy._validation as validation
from gamspy._config import get_option
from gamspy._special_values import SpecialValues
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import pandas as pd

    from gamspy import Alias, Equation, Model, Set, UniverseAlias, Variable
    from gamspy._algebra.domain import Domain
    from gamspy._algebra.expression import Expression
    from gamspy._symbols.implicits import ImplicitParameter, ImplicitSet
    from gamspy._types import IndexType, SymbolType

SPECIAL_VALUE_MAP = {
    SpecialValues.NA: "NA",
    SpecialValues.UNDEF: "UNDF",
    SpecialValues.POSINF: "INF",
    SpecialValues.NEGINF: "-INF",
}

CAPABILITIES_FILE = "gmscmpNT.txt" if platform.system() == "Windows" else "gmscmpun.txt"

user_dir = os.path.expanduser("~")
DEFAULT_DIR = os.path.join(user_dir, ".local", "share", "GAMSPy")
if platform.system() == "Darwin":
    DEFAULT_DIR = os.path.join(user_dir, "Library", "Application Support", "GAMSPy")
elif platform.system() == "Windows":
    DEFAULT_DIR = os.path.join(user_dir, "Documents", "GAMSPy")

_defaults: dict[str, dict[str, str]] = {}
_capabilities: dict[str, dict[str, list[str]]] = {}
_installed_solvers: dict[str, list[str]] = {}

_cached_system_directory = None


def getDefaultSolvers(system_directory: str) -> dict[str, str]:
    """
    Returns the default solver for each problem type.

    Parameters
    ----------
    system_directory : str

    Returns
    -------
    dict[str, str]

    Examples
    --------
    >>> import gamspy as gp
    >>> import gamspy_base
    >>> default_solvers = gp.utils.getDefaultSolvers(gamspy_base.directory)

    """
    global _defaults
    try:
        return _defaults[system_directory]
    except KeyError:
        ...

    capabilities_path = os.path.join(system_directory, CAPABILITIES_FILE)
    with open(capabilities_path, encoding="utf-8") as file:
        lines = file.read().split("DEFAULTS")[1].splitlines()[1:]

    defaults: dict[str, str] = {}
    for line in lines:
        problem, solver = line.split()
        defaults[problem] = solver

    _defaults[system_directory] = defaults
    return defaults


def getSolverCapabilities(system_directory: str) -> dict[str, list[str]]:
    """
    Returns a dictionary where keys are the solvers and values are the
    capabilities of the solver.

    Parameters
    ----------
    system_directory : str

    Returns
    -------
    dict[str, list[str]]

    Examples
    --------
    >>> import gamspy.utils as utils
    >>> import gamspy_base
    >>> caps = utils.getSolverCapabilities(gamspy_base.directory)
    >>> # caps["CPLEX"] would return list like ['LP', 'MIP', ...]

    """
    global _capabilities
    try:
        return _capabilities[system_directory]
    except KeyError:
        ...

    capabilities_path = os.path.join(system_directory, CAPABILITIES_FILE)
    capabilities: dict[str, list[str]] = {}

    with open(capabilities_path, encoding="utf-8") as file:
        lines = file.read().splitlines()

    while True:
        line = lines.pop(0)
        if line.startswith("*") or line == "":
            continue
        if line == "DEFAULTS":
            break

        solver, _, _, _, _, _, num_lines, *problem_types = line.split()

        for _ in range(int(num_lines) + 1):
            _ = lines.pop(0)

        capabilities[solver] = problem_types

    capabilities.pop("MPSGE", None)
    _capabilities[system_directory] = capabilities
    return capabilities


def getInstalledSolvers(system_directory: str) -> list[str]:
    """
    Returns the list of installed solvers

    Returns
    -------
    list[str]

    Raises
    ------
    ModuleNotFoundError
        In case gamspy_base is not installed.

    Examples
    --------
    >>> import gamspy_base
    >>> import gamspy.utils as utils
    >>> installed_solvers = utils.getInstalledSolvers(gamspy_base.directory)

    """
    global _installed_solvers
    try:
        return _installed_solvers[system_directory]
    except KeyError:
        ...

    capabilities_path = os.path.join(system_directory, CAPABILITIES_FILE)
    solvers: list[str] = []

    with open(capabilities_path, encoding="utf-8") as file:
        lines = file.read().splitlines()

    while True:
        line = lines.pop(0)
        if line.startswith("*") or line == "":
            continue
        if line == "DEFAULTS":
            break

        solver, _, _, _, _, _, num_lines, *_ = line.split()

        for _ in range(int(num_lines) + 1):
            _ = lines.pop(0)

        if solver != "GUSS":
            solvers.append(solver)

    solvers.remove("CONOPT")
    solvers.sort()
    _installed_solvers[system_directory] = solvers
    return solvers


def getAvailableSolvers() -> list[str]:
    """
    Returns all available solvers.

    Returns
    -------
    list[str]

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
    except ModuleNotFoundError as e:  # pragma: no cover
        e.msg = "You must first install gamspy_base to use this functionality"
        raise e

    solvers = sorted(gamspy_base.available_solvers)
    if "CONOPT" in solvers and "CONOPT4" in solvers:
        solvers.remove("CONOPT")

    return solvers


def getInstallableSolvers() -> list[str]:
    """
    Returns all installable solvers.

    Returns
    -------
    list[str]

    Raises
    ------
    ModuleNotFoundError
        In case gamspy_base is not installed.

    Examples
    --------
    >>> import gamspy_base
    >>> import gamspy.utils as utils
    >>> available_solvers = utils.getInstallableSolvers()

    """
    try:
        import gamspy_base
    except ModuleNotFoundError as e:  # pragma: no cover
        e.msg = "You must first install gamspy_base to use this functionality"
        raise e

    return sorted(set(getAvailableSolvers()) - set(gamspy_base.default_solvers))


def checkAllSame(
    iterable1: Sequence[SymbolType], iterable2: Sequence[SymbolType]
) -> bool:
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
    for elem1, elem2 in zip(iterable1, iterable2, strict=True):
        if elem1 is not elem2:
            return False
    return all_same


def isin(symbol: SymbolType | ImplicitParameter, sequence: Sequence) -> bool:
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


def setBaseEqual(set_a: Set | Alias, set_b: Set | Alias) -> bool:
    """
    Checks if two sets are equal considering aliases as equal as well.

    Parameters
    ----------
    set_a : Set | Alias
    set_b : Set | Alias

    Returns
    -------
    bool

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i")
    >>> j = gp.Set(m, "j")
    >>> gp.utils.setBaseEqual(i, j)
    False
    >>> k = gp.Alias(m, "k", i)
    >>> gp.utils.setBaseEqual(k, i)
    True

    """
    a = getattr(set_a, "alias_with", set_a)
    b = getattr(set_b, "alias_with", set_b)
    return a == b


def _get_unique_name() -> str:
    """
    N= 2^122 and the collision probability is: 1 - e^(-(n^2 / 2*N))
    """
    u = uuid.uuid4()
    b64 = base64.urlsafe_b64encode(u.bytes)
    return b64.rstrip(b"=").decode("ascii").replace("-", "_")


def _get_name_from_stack() -> str:
    try:
        frame = inspect.currentframe()
        if frame is None:
            raise RuntimeError("Could not get current frame")

        # Current frame is this function. The first f_back takes us to _get_symbol_name.
        # The second f_back takes us to the __init__ function of the symbol or addX
        # function of Container. The third f_back takes us to the user code.
        for _ in range(3):
            frame = frame.f_back
            if frame is None:
                raise RuntimeError("Call stack is not deep enough")

        info = inspect.getframeinfo(frame)

        if not info.code_context:
            raise RuntimeError("No code context available")

        line = info.code_context[0]

        # Pretty naive but this is the best chance we've got with little overhead.
        # e.g. "i = Set(m)" -> "i"
        name = line.split("=", maxsplit=1)[0].strip()

        return validation.validate_name(name)
    except Exception as e:
        raise ValidationError(
            f"It is not possible to get the Python variable name in this context: {e}"
        ) from e


def _calculate_infeasibilities(symbol: Variable | Equation) -> pd.DataFrame:
    records = symbol.records
    if records is None:
        raise ValidationError(
            "The model is not solved yet. Please solve the model first to compute the infeasibilities."
        )

    infeas_rows = records.where(
        (records["level"] < records["lower"]) | (records["level"] > records["upper"])
    ).dropna()
    lower_distance = (infeas_rows["lower"] - infeas_rows["level"]).abs()
    upper_distance = (infeas_rows["upper"] - infeas_rows["level"]).abs()
    infeas = lower_distance.combine(upper_distance, min)
    infeas_rows["infeasibility"] = infeas

    return infeas_rows


def _filter_gams_string(raw_string: str) -> str:
    FILTERS = (
        "$onMultiR",
        "$offMulti",
        "$onUNDF",
        "$offUNDF",
        "$onDotL",
        "$offDotL",
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
    except ModuleNotFoundError as e:  # pragma: no cover
        e.msg = "You must first install gamspy_base to use this functionality"
        raise e

    return gamspy_base.directory


def _get_license_path(system_directory: str) -> str:
    # Check ci license
    ci_license_path = os.path.join(system_directory, "ci_license.txt")
    if os.path.exists(ci_license_path):
        return ci_license_path

    # Check if a new license was installed.
    gamspy_license_path = os.path.join(DEFAULT_DIR, "gamspy_license.txt")
    if os.path.exists(gamspy_license_path):
        return gamspy_license_path

    # No preinstalled licenses on the machine. Use the demo license.
    return os.path.join(system_directory, "gamslice.txt")


def _replace_equality_signs(string: str) -> str:
    string = string.replace("=l=", "<=")
    string = string.replace("=e=", "eq")
    string = string.replace("=g=", ">=")
    return string


def _to_list(obj: IndexType) -> list:
    """Converts the given object to a list"""
    if type(obj) is tuple:
        return list(obj)

    if type(obj) is not list:
        return [obj]

    return obj


def _map_special_values(value: float):
    if not get_option("MAP_SPECIAL_VALUES"):
        return value

    if SpecialValues.isEps(value):
        return "EPS"

    if value in SPECIAL_VALUE_MAP:
        return SPECIAL_VALUE_MAP[value]

    return value


def _get_domain_str(
    domain: Iterable[Set | Alias | UniverseAlias | ImplicitSet | str],
    *,
    latex: bool = False,
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
    domain_strs = []
    for elem in domain:
        if isinstance(
            elem, (syms.Set, syms.Alias, syms.UniverseAlias, implicits.ImplicitSet)
        ):
            if latex:
                domain_strs.append(elem.latexRepr())
            else:
                domain_strs.append(elem.gamsRepr())
        elif isinstance(elem, str):
            if elem == "*":
                domain_strs.append(elem)
            else:
                if latex:
                    domain_strs.append('"' + elem.replace("_", r"\_") + '"')
                else:
                    domain_strs.append('"' + elem + '"')
        else:
            raise ValidationError(
                f"Domain type must be str, Set or Alias but found {type(elem)}"
            )

    return "(" + ",".join(domain_strs) + ")"


def _permute_domain(domain, dims):
    """
    Returns a new array where original domain's dimensions are permuted.

    Parameters
    ----------
    domain : list[Set | str]
    dims : list[int]

    Returns
    -------
    list[Set | str]

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i")
    >>> j = gp.Set(m, "j")
    >>> k = gp.Set(m, "k")
    >>> domain = [i, j, k]
    >>> new_domain = gp.utils._permute_domain(domain, [2, 0, 1])
    >>> new_domain[0] is domain[2]
    True
    >>> new_domain[1] is domain[0]
    True
    >>> new_domain[2] is domain[1]
    True

    """
    new_domain = [domain[dim] for dim in dims]
    return new_domain


def _get_set(domain: list[Set | Alias | Domain | Expression]):
    from gamspy import Domain

    res = []
    for el in domain:
        if hasattr(el, "left"):
            if hasattr(el.left, "sets"):
                res.extend(el.left.sets)  # type: ignore
            else:
                res.append(el.left)
        elif isinstance(el, Domain):
            res.extend(el.sets)
        else:
            res.append(el)

    return res


def _unpack(domain: list[Set | Alias | ImplicitSet]):
    unpacked = []
    for elem in domain:
        if isinstance(elem, implicits.ImplicitSet):
            if elem.extension is not None:
                unpacked.append(elem.parent)
            else:
                members = []
                for member in elem.domain:
                    if isinstance(member, implicits.ImplicitSet):
                        members.append(member.parent)
                        members.extend(member.domain)
                    else:
                        members.append(member)

                unpacked.extend([*members, elem.parent])
        else:
            unpacked.append(elem)

    return unpacked


def _parse_generated_equations(model: Model, listing_file: str) -> None:
    with open(listing_file) as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if line not in ("\n", "")]

    equation_listing_start_idx = 0
    for idx, line in enumerate(lines):
        if line.startswith("Equation Listing"):
            equation_listing_start_idx = idx
            break

    lines = lines[equation_listing_start_idx + 1 :]

    equation_listing_end_idx = 0
    for idx, line in enumerate(lines):
        if line.startswith(
            "G e n e r a l   A l g e b r a i c   M o d e l i n g   S y s t e m"
        ):
            equation_listing_end_idx = idx
            break

    lines = lines[: equation_listing_end_idx - 1]

    idx = 0
    for equation in model.equations:
        while not lines[idx].startswith(f"---- {equation.name}"):
            idx += 1

        idx += 1
        equation_listing: list[str] = []

        while idx < len(lines) and lines[idx].startswith(equation.name):
            equation_listing.append(lines[idx])
            idx += 1

        equation._equation_listing = equation_listing


def _parse_generated_variables(model: Model, listing_file: str) -> None:
    from gams.core.gdx import GMS_DT_VAR

    variable_names = gdxio._get_symbol_names_from_gdx(
        model.container.system_directory,
        model.container._gdx_out,
        symbol_type=GMS_DT_VAR,
    )
    model._variable_names = variable_names

    with open(listing_file) as file:
        lines = file.readlines()

    variable_listing_start_idx = 0
    for idx, line in enumerate(lines):
        if line.startswith("Column Listing"):
            variable_listing_start_idx = idx
            break

    variable_listing_start_idx += 3

    lines = lines[variable_listing_start_idx:]

    variable_listing_end_idx = 0
    for idx, line in enumerate(lines):
        if line.startswith(
            "G e n e r a l   A l g e b r a i c   M o d e l i n g   S y s t e m"
        ):
            variable_listing_end_idx = idx
            break

    lines = lines[: variable_listing_end_idx - 1]

    for name in variable_names:
        listings: list[str] = []
        idx = 0
        while idx < len(lines) and not lines[idx].startswith(f"---- {name}"):
            idx += 1

        idx += 2
        start_index = idx
        while idx < len(lines):
            line = lines[idx]
            if line.startswith("----"):
                listings = listings[:-1]
                break

            if line == "\n":
                listings.append("".join(lines[start_index:idx]))
                start_index = idx + 1

            idx += 1

        variable = cast("Variable", model.container._data[name])
        variable._column_listing = listings

    return None
