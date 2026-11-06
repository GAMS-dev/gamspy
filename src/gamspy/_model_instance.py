from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
from gams.core.cfg import GMS_SSSIZE
from gams.core.gev import (
    gevCreateD,
    gevFree,
    gevHandleToPtr,
    gevInitEnvironmentLegacy,
    new_gevHandle_tp,
)
from gams.core.gmo import (
    doubleArray,
    gmoCreateD,
    gmoFree,
    gmoGetEquNameOne,
    gmoGetEquOrderOne,
    gmoGetEquType,
    gmoGetIndicatorMap,
    gmoGetMatrixRow,
    gmoGetRhs,
    gmoGetVarL,
    gmoGetVarLower,
    gmoGetVarNameOne,
    gmoGetVarType,
    gmoGetVarUpper,
    gmoLoadDataLegacy,
    gmoM,
    gmoMinfSet,
    gmoN,
    gmoNZ,
    gmoObjVar,
    gmoorder_L,
    gmoPinfSet,
    gmoRegisterEnvironment,
    intArray,
    new_gmoHandle_tp,
)
from gams.core.opt import (
    new_optHandle_tp,
    optCreateD,
    optFree,
    optHandleToPtr,
    optReadDefinitionFromPChar,
    optReadFromStr,
)

from gamspy._backend.backend import backend_factory
from gamspy.exceptions import GamspyException

if TYPE_CHECKING:
    from gamspy import Container, Model
    from gamspy._options import Options

EQUATION_TYPES = ["=E=", "=G=", "=L=", "=N=", "=X=", "=C=", "=B="]
VARIABLE_TYPES = [
    "continuous",
    "binary",
    "integer",
    "sos1",
    "sos2",
    "semicont",
    "semiint",
]


@dataclass(eq=False)
class ModelInstance:
    """
    Generated model instance of a :meth:`Model <gamspy.Model>` with the current data.

    Attributes
    ----------
    equations : pd.DataFrame
        One row per generated equation with columns name, type, rhs, and nonlinear.
    variables : pd.DataFrame
        One row per generated variable with columns name, type, lower, upper, and level.
    jacobian : pd.DataFrame
        One row per nonzero with columns equation, variable, coefficient, and nonlinear.
        The coefficient of a nonlinear entry is its derivative at the current level values.
    indicators : pd.DataFrame
        One row per native indicator constraint with columns equation, variable, and value.
        The equation is enforced only when the binary variable equals the value.
    objective_variable : str | None
        Name of the objective variable, if the model has one.
    """

    equations: pd.DataFrame
    variables: pd.DataFrame
    jacobian: pd.DataFrame
    indicators: pd.DataFrame
    objective_variable: str | None

    def __repr__(self) -> str:
        return (
            f"ModelInstance(equations={len(self.equations)}, "
            f"variables={len(self.variables)}, nonzeros={len(self.jacobian)})"
        )


def get_control_file(container: Container) -> str:
    return os.path.join(container._process_directory, "gamscntr.dat")


def create_handles(system_directory: str) -> tuple[object, object]:
    gev = new_gevHandle_tp()
    ret = gevCreateD(gev, system_directory, GMS_SSSIZE)
    if not ret[0]:
        raise GamspyException(ret[1])

    gmo = new_gmoHandle_tp()
    ret = gmoCreateD(gmo, system_directory, GMS_SSSIZE)
    if not ret[0]:
        gevFree(gev)
        raise GamspyException(ret[1])

    return gev, gmo


def free_handles(gmo: object, gev: object) -> None:
    gmoFree(gmo)
    gevFree(gev)


def load_instance(gev: object, gmo: object, control_file: str) -> None:
    if gevInitEnvironmentLegacy(gev, control_file) != 0:
        raise GamspyException("Could not initialize model instance")

    gmoRegisterEnvironment(gmo, gevHandleToPtr(gev))
    ret = gmoLoadDataLegacy(gmo)
    if ret[0] != 0:
        raise GamspyException(f"Could not load model instance: {ret[1]}")


def generate_model_instance(model: Model, options: Options) -> ModelInstance:
    container = model.container
    control_file = get_control_file(container)
    indicators = model._get_indicators()

    model._add_runtime_options(options)
    container._add_statement(f"{model.name}.justScrDir = 1;")
    container._add_statement(model._generate_solve_string() + ";")

    with options._temporary_extra_options({"solvercntr": control_file}):
        backend_factory(container, options, output=container.output).run()

    gev, gmo = create_handles(container.system_directory)
    opt = None
    try:
        if indicators:
            opt = _create_indicator_options(container.system_directory, indicators)

        load_instance(gev, gmo, control_file)
        gmoPinfSet(gmo, math.inf)
        gmoMinfSet(gmo, -math.inf)

        return _read_instance(gmo, opt)
    finally:
        if opt is not None:
            optFree(opt)
        free_handles(gmo, gev)


def _create_indicator_options(system_directory: str, indicators: list[str]) -> object:
    opt = new_optHandle_tp()
    ret = optCreateD(opt, system_directory, GMS_SSSIZE)
    if not ret[0]:
        raise GamspyException(ret[1])

    if optReadDefinitionFromPChar(opt, "indic indicator\n")[0] != 0:
        optFree(opt)
        raise GamspyException("Could not read the native indicator constraints")

    for indicator in indicators:
        optReadFromStr(opt, indicator)

    return opt


def _read_indicators(
    gmo: object,
    opt: object | None,
    equation_names: list[str],
    variable_names: list[str],
) -> pd.DataFrame:
    count = 0
    m = gmoM(gmo)
    rows, columns, values = intArray(m), intArray(m), intArray(m)
    if opt is not None:
        ret = gmoGetIndicatorMap(gmo, optHandleToPtr(opt), 1, 0, rows, columns, values)
        if ret[0] != 0:
            raise GamspyException("Could not read the native indicator constraints")
        count = ret[1]

    return pd.DataFrame(
        {
            "equation": [equation_names[rows[k]] for k in range(count)],
            "variable": [variable_names[columns[k]] for k in range(count)],
            "value": [values[k] for k in range(count)],
        }
    )


def _read_instance(gmo: object, opt: object | None) -> ModelInstance:
    m, n, nz = gmoM(gmo), gmoN(gmo), gmoNZ(gmo)

    # TODO: Ask for a bulk api from the maintainers of gmo.
    equation_names = [gmoGetEquNameOne(gmo, i) for i in range(m)]
    variable_names = [gmoGetVarNameOne(gmo, j) for j in range(n)]

    equation_types = intArray(m)
    rhs = doubleArray(m)
    gmoGetEquType(gmo, equation_types)
    gmoGetRhs(gmo, rhs)
    equations = pd.DataFrame(
        {
            "name": equation_names,
            "type": [EQUATION_TYPES[equation_types[i]] for i in range(m)],
            "rhs": [rhs[i] for i in range(m)],
            "nonlinear": [gmoGetEquOrderOne(gmo, i) > gmoorder_L for i in range(m)],
        }
    )

    variable_types = intArray(n)
    lower = doubleArray(n)
    upper = doubleArray(n)
    level = doubleArray(n)
    gmoGetVarType(gmo, variable_types)
    gmoGetVarLower(gmo, lower)
    gmoGetVarUpper(gmo, upper)
    gmoGetVarL(gmo, level)
    variables = pd.DataFrame(
        {
            "name": variable_names,
            "type": [VARIABLE_TYPES[variable_types[j]] for j in range(n)],
            "lower": [lower[j] for j in range(n)],
            "upper": [upper[j] for j in range(n)],
            "level": [level[j] for j in range(n)],
        }
    )

    row_start = intArray(m + 1)
    column_index = intArray(nz)
    values = doubleArray(nz)
    nonlinear = intArray(nz)
    gmoGetMatrixRow(gmo, row_start, column_index, values, nonlinear)
    rows = [i for i in range(m) for _ in range(row_start[i], row_start[i + 1])]
    jacobian = pd.DataFrame(
        {
            "equation": [equation_names[i] for i in rows],
            "variable": [variable_names[column_index[k]] for k in range(nz)],
            "coefficient": [values[k] for k in range(nz)],
            "nonlinear": [nonlinear[k] != 0 for k in range(nz)],
        }
    )

    objective_index = gmoObjVar(gmo)
    objective_variable = (
        variable_names[objective_index] if 0 <= objective_index < n else None
    )

    indicators = _read_indicators(gmo, opt, equation_names, variable_names)

    return ModelInstance(equations, variables, jacobian, indicators, objective_variable)
