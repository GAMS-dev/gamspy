from __future__ import annotations

import base64
import inspect
import os
import platform
import uuid
from typing import TYPE_CHECKING

import gams.transfer as gt
from gams.core import gdx

import gamspy._model as model
import gamspy._symbols.implicits as implicits
import gamspy._validation as validation
from gamspy._config import get_option
from gamspy.exceptions import FatalError, ValidationError

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import pandas as pd
    from gams.core.numpy import Gams2Numpy

    from gamspy import (
        Alias,
        Container,
        Domain,
        Equation,
        Expression,
        Model,
        Set,
        Variable,
    )
    from gamspy._symbols.implicits import ImplicitSet
    from gamspy._types import EllipsisType

SPECIAL_VALUE_MAP = {
    gt.SpecialValues.NA: "NA",
    gt.SpecialValues.UNDEF: "UNDF",
    gt.SpecialValues.POSINF: "INF",
    gt.SpecialValues.NEGINF: "-INF",
}

CAPABILITIES_FILE = (
    "gmscmpNT.txt" if platform.system() == "Windows" else "gmscmpun.txt"
)

user_dir = os.path.expanduser("~")
if platform.system() == "Linux":
    DEFAULT_DIR = os.path.join(user_dir, ".local", "share", "GAMSPy")
elif platform.system() == "Darwin":
    DEFAULT_DIR = os.path.join(
        user_dir, "Library", "Application Support", "GAMSPy"
    )
elif platform.system() == "Windows":
    DEFAULT_DIR = os.path.join(user_dir, "Documents", "GAMSPy")

_defaults: dict[str, dict[str, str]] = dict()
_capabilities: dict[str, dict[str, list[str]]] = dict()
_installed_solvers: dict[str, list[str]] = dict()


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

    defaults: dict[str, str] = dict()
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
    """
    global _capabilities
    try:
        return _capabilities[system_directory]
    except KeyError:
        ...

    capabilities_path = os.path.join(system_directory, CAPABILITIES_FILE)
    capabilities: dict[str, list[str]] = dict()

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


def getInstallableSolvers(system_directory: str) -> list[str]:
    """
    Returns all installable solvers.

    Parameters
    ----------
    system_directory : str

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
    >>> available_solvers = utils.getInstallableSolvers(gamspy_base.directory)

    """
    try:
        import gamspy_base
    except ModuleNotFoundError as e:  # pragma: no cover
        e.msg = "You must first install gamspy_base to use this functionality"
        raise e

    return sorted(
        list(set(getAvailableSolvers()) - set(gamspy_base.default_solvers))
    )


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
    return float(arrvals[0][0])


def _get_unique_name() -> str:
    """
    N= 2^122 and the collision probability is: 1 - e^(-(n^2 / 2*N))
    """
    u = uuid.uuid4()
    b64 = base64.urlsafe_b64encode(u.bytes)
    return b64.rstrip(b"=").decode("ascii").replace("-", "_")


def _get_name_from_stack() -> str:
    try:
        # Current frame is this function (_get_name_from_stack)
        # The first f_back takes us to _get_symbol_name
        # The second f_back takes us to the __init__ function of
        # the symbol. The third f_back takes us to the user code.
        frame = inspect.currentframe().f_back.f_back.f_back

        # We get the line that defines the symbol. e.g. i = Set(m)
        line = inspect.getframeinfo(frame).code_context[0]

        # Pretty naive but it's the best chance we've got with little overhead.
        name = line.split("=", maxsplit=1)[0].strip()
        name = validation.validate_name(name)
    except Exception as e:
        raise ValidationError(
            f"It is not possible to get the Python variable name in this context: {e}"
        ) from e

    return name


def _get_symbol_name(prefix: str) -> str:
    use_py_var_name = get_option("USE_PY_VAR_NAME")
    if use_py_var_name == "no":
        name = prefix + _get_unique_name() + "gpauto"
    elif use_py_var_name == "yes":
        name = _get_name_from_stack()
    elif use_py_var_name == "yes-or-autogenerate":
        try:
            name = _get_name_from_stack()
        except ValidationError:
            name = prefix + _get_unique_name() + "gpauto"
    else:
        raise ValidationError(
            f'Invalid value `{use_py_var_name}` for `USE_PY_VAR_NAME`. Possible values are "no", "yes", "yes-or-autogenerate"'
        )

    return name


def _get_symbol_names_from_gdx(
    system_directory: str, load_from: str
) -> list[str]:
    gdx_handle = _open_gdx_file(system_directory, load_from)
    _, symbol_count, _ = gdx.gdxSystemInfo(gdx_handle)

    symbol_names = []
    for i in range(1, symbol_count + 1):
        _, symbol_name, _, _ = gdx.gdxSymbolInfo(gdx_handle, i)
        if not symbol_name.startswith(model.Model._generate_prefix):
            symbol_names.append(symbol_name)

    _close_gdx_handle(gdx_handle)

    return symbol_names


def _get_variables_of_model(container: Container):
    names = _get_symbol_names_from_gdx(
        container.system_directory, container._gdx_out
    )

    return [
        container[name]
        for name in names
        if isinstance(container[name], gt.Variable)
    ]


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

    # Check old license installation path.
    user_license_path = os.path.join(system_directory, "user_license.txt")
    if os.path.exists(user_license_path):
        return user_license_path

    # No preinstalled licenses on the machine. Use the demo license.
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
        raise FatalError("GAMSPy could not create the gdx handle.") from e

    try:
        rc = gdx.gdxOpenRead(gdx_handle, load_from)
        assert rc[0]
    except AssertionError as e:
        raise FatalError(
            "GAMSPy could not open the gdx file to read from."
        ) from e

    return gdx_handle


def _to_list(
    obj: EllipsisType
    | slice
    | Set
    | Alias
    | str
    | Iterable
    | Sequence
    | ImplicitSet,
) -> list:
    """Converts the given object to a list"""
    if type(obj) is tuple:
        return list(obj)

    if type(obj) is not list:
        return [obj]

    return obj


def _map_special_values(value: float):
    if not get_option("MAP_SPECIAL_VALUES"):
        return value

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
        if isinstance(
            set, (gt.Set, gt.Alias, gt.UniverseAlias, implicits.ImplicitSet)
        ):
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


# TODO either add description or make private
def _get_set(domain: list[Set | Alias | Domain | Expression]):
    res = []
    for el in domain:
        if hasattr(el, "left"):
            if hasattr(el.left, "sets"):
                res.extend(el.left.sets)  # type: ignore
            else:
                res.append(el.left)
        elif hasattr(el, "sets"):
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
                    else:
                        members.append(member)
                unpacked += [*members, elem.parent]
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
        equation_listing = []

        while idx < len(lines) and lines[idx].startswith(equation.name):
            equation_listing.append(lines[idx])
            idx += 1

        equation._equation_listing = equation_listing


def _parse_generated_variables(model: Model, listing_file: str) -> None:
    variables = _get_variables_of_model(model.container)
    model._variables = variables  # type: ignore

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

    for variable in variables:
        listings: list[str] = []
        idx = 0
        while idx < len(lines) and not lines[idx].startswith(
            f"---- {variable.name}"
        ):
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

        variable._column_listing = listings

    return None
