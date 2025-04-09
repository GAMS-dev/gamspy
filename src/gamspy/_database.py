from __future__ import annotations

import os
import weakref
from typing import TYPE_CHECKING

from gams.core.gdx import delete_intp, doubleArray, intp_value
from gams.core.gmd import (
    GMD_NRRECORDS,
    GMS_SSSIZE,
    GMS_SV_UNDEF,
    dt_equ,
    dt_par,
    dt_set,
    dt_var,
    gmdAddSymbolPy,
    gmdCreateD,
    gmdFree,
    gmdGetLastError,
    gmdSetSpecialValues,
    gmdSymbolInfo,
    gmdWriteGDX,
    new_gmdHandle_tp,
    new_intp,
)

import gamspy.utils as utils
from gamspy.exceptions import GamspyException

if TYPE_CHECKING:
    from gamspy._workspace import Workspace

SV_UNDEF = GMS_SV_UNDEF
SV_EPS = 4.94066e-324

_spec_values = doubleArray(5)
_spec_values[0] = SV_UNDEF
_spec_values[1] = float("nan")
_spec_values[2] = float("inf")
_spec_values[3] = float("-inf")
_spec_values[4] = SV_EPS


def _int_value_and_free(intP):
    intp_val = intp_value(intP)
    delete_intp(intP)
    return intp_val


class GamsSymbol:
    """
    Representation of a symbol in GAMS. It exists in a Database and contains
    GamsSymbolRecords which one can iterate through.
    """

    def __init__(
        self,
        database: Database,
        name: str,
        dimension: int,
        explanatory_text: str,
    ):
        self.database = database
        self.name = name
        self.dimension = dimension
        self.text = explanatory_text
        self.database.symbols[self.name] = self

    def __len__(self):
        return self.get_number_records()

    def get_number_records(self):
        ret = gmdSymbolInfo(self.database.gmd, self.sym_ptr, GMD_NRRECORDS)
        self.database._check_for_gmd_error(ret[0], self.database.workspace)
        return ret[1]


class GamsSet(GamsSymbol):
    """Representation of a set symbol in GAMS"""

    def __init__(
        self,
        database: Database,
        name: str,
        dimension: int,
        explanatory_text: str = "",
        settype: int = 0,
    ):
        super().__init__(database, name, dimension, explanatory_text)

        self.settype = settype
        rc = new_intp()
        self.sym_ptr = gmdAddSymbolPy(
            self.database.gmd,
            self.name,
            self.dimension,
            dt_set,
            self.settype,
            self.text,
            rc,
        )
        self.database._check_for_gmd_error(
            _int_value_and_free(rc), self.database.workspace
        )


class GamsParameter(GamsSymbol):
    """Representation of a parameter symbol in GAMS"""

    def __init__(
        self,
        database: Database,
        name: str,
        dimension: int,
        explanatory_text: str = "",
    ):
        super().__init__(database, name, dimension, explanatory_text)

        rc = new_intp()
        self.sym_ptr = gmdAddSymbolPy(
            self.database.gmd,
            self.name,
            self.dimension,
            dt_par,
            0,
            self.text,
            rc,
        )
        self.database._check_for_gmd_error(
            _int_value_and_free(rc), self.database.workspace
        )


class GamsVariable(GamsSymbol):
    """Representation of a variable symbol in GAMS"""

    def __init__(
        self,
        database: Database,
        name: str,
        dimension: int,
        vartype: int,
        explanatory_text: str = "",
    ):
        super().__init__(database, name, dimension, explanatory_text)

        self.vartype = vartype
        rc = new_intp()
        self.sym_ptr = gmdAddSymbolPy(
            self.database.gmd,
            self.name,
            self.dimension,
            dt_var,
            self.vartype,
            self.text,
            rc,
        )
        self.database._check_for_gmd_error(
            _int_value_and_free(rc), self.database.workspace
        )


class GamsEquation(GamsSymbol):
    """Representation of an equation symbol in GAMS"""

    def __init__(
        self,
        database: Database,
        name: str,
        dimension: int,
        equtype: int,
        explanatory_text: str = "",
    ):
        super().__init__(database, name, dimension, explanatory_text)

        self.equtype = equtype
        rc = new_intp()
        self.sym_ptr = gmdAddSymbolPy(
            self.database.gmd,
            self.name,
            self.dimension,
            dt_equ,
            self.equtype,
            self.text,
            rc,
        )
        self.database._check_for_gmd_error(
            _int_value_and_free(rc), self.database.workspace
        )


class Database:
    """Communicates data between Python and GAMS"""

    def __init__(self, ws: Workspace):
        self.symbols: dict = dict()
        self.workspace = ws
        self.gmd = new_gmdHandle_tp()

        ret = gmdCreateD(
            self.gmd, utils._get_gamspy_base_directory(), GMS_SSSIZE
        )
        if not ret[0]:
            raise GamspyException(ret[1])

        rc = gmdSetSpecialValues(self.gmd, _spec_values)
        self._check_for_gmd_error(rc, self.workspace)

        weakref.finalize(self, self.cleanup, self.gmd)

    @staticmethod
    def cleanup(gmd):
        gmdFree(gmd)

    def __len__(self):
        return len(self.symbols)

    def __getitem__(self, symbol_name: str):
        return self.symbols[symbol_name]

    def _check_for_gmd_error(self, rc, workspace: Workspace):
        if not rc:
            msg = gmdGetLastError(self.gmd)[1]
            workspace._errors.append(msg)
            raise GamspyException(msg, rc)

    def add_equation(
        self,
        name: str,
        dimension: int,
        equtype: int,
        explanatory_text: str = "",
    ) -> GamsEquation:
        return GamsEquation(self, name, dimension, equtype, explanatory_text)

    def add_variable(
        self,
        name: str,
        dimension: int,
        vartype: int,
        explanatory_text: str = "",
    ) -> GamsVariable:
        return GamsVariable(self, name, dimension, vartype, explanatory_text)

    def add_set(
        self,
        name: str,
        dimension: int,
        explanatory_text: str = "",
        settype: int = 0,
    ) -> GamsSet:
        return GamsSet(
            self, name, dimension, explanatory_text, settype=settype
        )

    def add_parameter(
        self, name: str, dimension: int, explanatory_text: str = ""
    ) -> GamsParameter:
        return GamsParameter(self, name, dimension, explanatory_text)

    def export(self, file_path: str) -> None:
        """Writes database into a GDX file"""
        assert file_path.endswith(".gdx"), (
            f"File path should point to a gdx file but got `{file_path}`"
        )

        if os.path.isabs(file_path):
            rc = gmdWriteGDX(self.gmd, file_path, False)
        else:
            rc = gmdWriteGDX(
                self.gmd,
                os.path.join(self.workspace.working_directory, file_path),
                False,
            )
        self._check_for_gmd_error(rc, self.workspace)
