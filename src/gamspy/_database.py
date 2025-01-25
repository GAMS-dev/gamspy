from __future__ import annotations

import os
from typing import TYPE_CHECKING

from gams.core.gdx import delete_intp, doubleArray, intp_value
from gams.core.gmd import (
    GMD_DIM,
    GMD_EXPLTEXT,
    GMD_NAME,
    GMD_NRRECORDS,
    GMD_NRSYMBOLS,
    GMD_USERINFO,
    GMS_MAX_INDEX_DIM,
    GMS_SSSIZE,
    GMS_SV_UNDEF,
    dt_alias,
    dt_equ,
    dt_par,
    dt_set,
    dt_var,
    gmdAddSymbolPy,
    gmdAddSymbolXPy,
    gmdCheckDBDV,
    gmdCheckSymbolDV,
    gmdCreateD,
    gmdFindSymbolPy,
    gmdFree,
    gmdGetDomain,
    gmdGetLastError,
    gmdGetSymbolByIndexPy,
    gmdHandleToPtr,
    gmdInfo,
    gmdSetSpecialValues,
    gmdSymbolInfo,
    gmdSymbolType,
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
        identifier=None,
        dimension=None,
        explanatory_text="",
        sym_ptr=None,
    ):
        self._database = database
        self._domains = None
        self._domains_as_strings = None

        # receive an already existing symbol from GMD
        if not (identifier or dimension or explanatory_text) and sym_ptr:
            if sym_ptr is None:
                raise GamspyException("Symbol does not exist")

            self._sym_ptr = sym_ptr
            rc, _ = gmdSymbolType(self._database._gmd, self._sym_ptr)
            self._database._check_for_gmd_error(rc)

            rc, _, _, self._name = gmdSymbolInfo(
                self._database._gmd, self._sym_ptr, GMD_NAME
            )
            self._database._check_for_gmd_error(rc)

            rc, self._dim, _, _ = gmdSymbolInfo(
                self._database._gmd, self._sym_ptr, GMD_DIM
            )
            self._database._check_for_gmd_error(rc)

            rc, _, _, self._text = gmdSymbolInfo(
                self._database._gmd, self._sym_ptr, GMD_EXPLTEXT
            )
            self._database._check_for_gmd_error(rc)

        # create a new symbol in GMD
        elif not sym_ptr and identifier and dimension is not None:
            if dimension < 0 or dimension > GMS_MAX_INDEX_DIM:
                raise GamspyException(
                    "Invalid dimension specified "
                    + str(dimension)
                    + " is not in [0,"
                    + str(GMS_MAX_INDEX_DIM)
                    + "]"
                )
            self._name = identifier
            self._dim = dimension
            self._text = explanatory_text

        else:
            raise GamspyException("Invalid combination of parameters")

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._sym_ptr == other._sym_ptr

        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return self.get_number_records()

    @property
    def dimension(self):
        return self._dim

    @property
    def text(self):
        return self._text

    @property
    def name(self):
        return self._name

    @property
    def database(self):
        return self._database

    def get_number_records(self):
        ret = gmdSymbolInfo(self._database._gmd, self._sym_ptr, GMD_NRRECORDS)
        self._database._check_for_gmd_error(ret[0])
        return ret[1]

    def get_domains_as_strings(self):
        if self._domains_as_strings is None:
            self._domains_as_strings = []
            if self._dim == 0:
                return self._domains_as_strings
            ret = gmdGetDomain(self._database._gmd, self._sym_ptr, self._dim)
            self._database._check_for_gmd_error(ret[0])

            for i in range(self._dim):
                self._domains_as_strings.append(ret[2][i])
        return self._domains_as_strings

    def get_domains(self):
        if self._domains is None:
            self._domains = []
            if self._dim == 0:
                return self._domains
            retDom = gmdGetDomain(
                self._database._gmd, self._sym_ptr, self._dim
            )
            self._database._check_for_gmd_error(retDom[0])
            domains = retDom[1]
            for i in range(self._dim):
                if domains[i] is not None:
                    retSym = gmdSymbolInfo(
                        self._database._gmd, domains[i], GMD_NAME
                    )
                    self._database._check_for_gmd_error(retSym[0])
                    name = retSym[3]
                    if name == "*":
                        self._domains.append("*")
                    else:
                        self._domains.append(
                            GamsSet(self._database, sym_ptr=domains[i])
                        )
                else:
                    self._domains.append(retDom[2][i])
        return self._domains

    def _concat_keys(self, keys):
        if not keys or len(keys) == 0:
            return ""
        else:
            ret = ""
            for i in range(len(keys) - 1):
                ret += keys[i] + ","
            ret += keys[len(keys) - 1]
            return ret

    def check_domains(self):
        """Check if all records are within the specified domain of the symbol"""
        has_violation = new_intp()
        rc = gmdCheckSymbolDV(
            self._database._gmd, self._sym_ptr, has_violation
        )
        self._database._check_for_gmd_error(rc)
        return _int_value_and_free(has_violation) != 1


class GamsVariable(GamsSymbol):
    """Representation of a variable symbol in GAMS"""

    def __init__(
        self,
        database,
        identifier=None,
        dimension=None,
        vartype=None,
        explanatory_text="",
        sym_ptr=None,
        domains=None,
    ):
        if (
            identifier
            and (vartype is not None)
            and (domains is not None)
            and not dimension
        ):
            super().__init__(
                database, identifier, len(domains), explanatory_text, sym_ptr
            )
        else:
            super().__init__(
                database, identifier, dimension, explanatory_text, sym_ptr
            )

        # receive an already existing symbol from GMD
        if (
            not (identifier or dimension or vartype or explanatory_text)
            and sym_ptr
        ):
            rc, subtype, _, _ = gmdSymbolInfo(
                self._database._gmd, self._sym_ptr, GMD_USERINFO
            )
            self._database._check_for_gmd_error(rc)
            self._vartype = subtype

        # create new variable in GMD
        elif (
            not sym_ptr
            and identifier
            and dimension is not None
            and vartype is not None
        ):
            self._vartype = vartype
            rc = new_intp()
            self._sym_ptr = gmdAddSymbolPy(
                self._database._gmd,
                self._name,
                self._dim,
                dt_var,
                self._vartype,
                self._text,
                rc,
            )
            self._database._check_for_gmd_error(_int_value_and_free(rc))

        # create new variable with domain information
        elif (
            identifier
            and (vartype is not None)
            and (domains is not None)
            and not dimension
        ):
            if not isinstance(domains, (list, tuple)):
                raise GamspyException(
                    "Parameter domains has to be a list or a tuple"
                )

            self._vartype = vartype
            if len(domains) == 0:
                rc = new_intp()
                self._sym_ptr = gmdAddSymbolPy(
                    self._database._gmd,
                    self._name,
                    self._dim,
                    dt_var,
                    self._vartype,
                    self._text,
                    rc,
                )
                self._database._check_for_gmd_error(_int_value_and_free(rc))
            else:
                dom_ptr = [None] * self._dim
                rel_dom = [""] * self._dim

                for i in range(self._dim):
                    if isinstance(domains[i], GamsSet):
                        dom_ptr[i] = domains[i]._sym_ptr
                    elif isinstance(domains[i], str):
                        rel_dom[i] = domains[i]
                    else:
                        raise GamspyException(
                            "Domain must be GamsSet or string but saw "
                            + str(type(domains[i]))
                            + " on index "
                            + str(i)
                        )
                rc = new_intp()
                self._sym_ptr = gmdAddSymbolXPy(
                    self._database._gmd,
                    self._name,
                    self._dim,
                    dt_var,
                    self._vartype,
                    self._text,
                    dom_ptr,
                    rel_dom,
                    rc,
                )
                self._database._check_for_gmd_error(_int_value_and_free(rc))
        else:
            raise GamspyException("Invalid combination of parameters")

    def get_vartype(self):
        return self._vartype


class GamsParameter(GamsSymbol):
    """Representation of a parameter symbol in GAMS"""

    def __init__(
        self,
        database,
        identifier=None,
        dimension=None,
        explanatory_text="",
        sym_ptr=None,
        domains=None,
    ):
        if identifier and (domains is not None) and not dimension:
            super().__init__(
                database, identifier, len(domains), explanatory_text, sym_ptr
            )
        else:
            super().__init__(
                database, identifier, dimension, explanatory_text, sym_ptr
            )

        # receive an already existing symbol from GMD - nothing to do
        if not (identifier or dimension or explanatory_text) and sym_ptr:
            pass

        # create new parameter in GMD
        elif not sym_ptr and identifier and dimension is not None:
            rc = new_intp()
            self._sym_ptr = gmdAddSymbolPy(
                self._database._gmd,
                self._name,
                self._dim,
                dt_par,
                0,
                self._text,
                rc,
            )
            self._database._check_for_gmd_error(_int_value_and_free(rc))
            if self._sym_ptr is None:
                raise GamspyException("Cannot create parameter " + self._name)

        # create new parameter with domain information
        elif identifier and (domains is not None) and not dimension:
            if not isinstance(domains, (list, tuple)):
                raise GamspyException(
                    "Parameter domains has to be a list or a tuple"
                )

            if len(domains) == 0:
                rc = new_intp()
                self._sym_ptr = gmdAddSymbolPy(
                    self._database._gmd,
                    self._name,
                    self._dim,
                    dt_par,
                    0,
                    self._text,
                    rc,
                )
                self._database._check_for_gmd_error(_int_value_and_free(rc))
            else:
                dom_ptr = [None] * self._dim
                rel_dom = [""] * self._dim

                for i in range(self._dim):
                    if isinstance(domains[i], GamsSet):
                        dom_ptr[i] = domains[i]._sym_ptr
                    elif isinstance(domains[i], str):
                        rel_dom[i] = domains[i]
                    else:
                        raise GamspyException(
                            "Domain must be GamsSet or string but saw "
                            + str(type(domains[i]))
                            + " on index "
                            + str(i)
                        )
                rc = new_intp()
                self._sym_ptr = gmdAddSymbolXPy(
                    self._database._gmd,
                    self._name,
                    self._dim,
                    dt_par,
                    0,
                    self._text,
                    dom_ptr,
                    rel_dom,
                    rc,
                )
                self._database._check_for_gmd_error(_int_value_and_free(rc))

        else:
            raise GamspyException("Invalid combination of parameters")


class GamsSet(GamsSymbol):
    """Representation of a set symbol in GAMS"""

    def __init__(
        self,
        database,
        identifier=None,
        dimension=None,
        explanatory_text="",
        sym_ptr=None,
        domains=None,
        settype=0,
    ):
        if identifier and (domains is not None) and not dimension:
            super().__init__(
                database, identifier, len(domains), explanatory_text, sym_ptr
            )
        else:
            super().__init__(
                database, identifier, dimension, explanatory_text, sym_ptr
            )

        # receive an already existing symbol from GMD
        if not (identifier or dimension or explanatory_text) and sym_ptr:
            rc, subtype, _, _ = gmdSymbolInfo(
                self._database._gmd, self._sym_ptr, GMD_USERINFO
            )
            self._database._check_for_gmd_error(rc)
            self._settype = subtype

        # create new set in GMD
        elif not sym_ptr and identifier and dimension is not None:
            self._settype = settype
            rc = new_intp()
            self._sym_ptr = gmdAddSymbolPy(
                self._database._gmd,
                self._name,
                self._dim,
                dt_set,
                self._settype,
                self._text,
                rc,
            )
            self._database._check_for_gmd_error(_int_value_and_free(rc))

        # create new set with domain information
        elif identifier and (domains is not None) and not dimension:
            if not isinstance(domains, (list, tuple)):
                raise GamspyException(
                    "Parameter domains has to be a list or a tuple"
                )

            self._settype = settype
            if len(domains) == 0:
                rc = new_intp()
                self._sym_ptr = gmdAddSymbolPy(
                    self._database._gmd,
                    self._name,
                    self._dim,
                    dt_set,
                    self._settype,
                    self._text,
                    rc,
                )
                self._database._check_for_gmd_error(_int_value_and_free(rc))
            else:
                dom_ptr = [None] * self._dim
                rel_dom = [""] * self._dim

                for i in range(self._dim):
                    if isinstance(domains[i], GamsSet):
                        dom_ptr[i] = domains[i]._sym_ptr
                    elif isinstance(domains[i], str):
                        rel_dom[i] = domains[i]
                    else:
                        raise GamspyException(
                            "Domain must be GamsSet or string but saw "
                            + str(type(domains[i]))
                            + " on index "
                            + str(i)
                        )
                rc = new_intp()
                self._sym_ptr = gmdAddSymbolXPy(
                    self._database._gmd,
                    self._name,
                    self._dim,
                    dt_set,
                    self._settype,
                    self._text,
                    dom_ptr,
                    rel_dom,
                    rc,
                )
                self._database._check_for_gmd_error(_int_value_and_free(rc))

        else:
            raise GamspyException("Invalid combination of parameters")

    def get_settype(self):
        return self._settype


class GamsEquation(GamsSymbol):
    """Representation of an equation symbol in GAMS"""

    def __init__(
        self,
        database,
        identifier=None,
        dimension=None,
        equtype=None,
        explanatory_text="",
        sym_ptr=None,
        domains=None,
    ):
        if (
            identifier
            and (equtype is not None)
            and (domains is not None)
            and not dimension
        ):
            super().__init__(
                database, identifier, len(domains), explanatory_text, sym_ptr
            )
        else:
            super().__init__(
                database, identifier, dimension, explanatory_text, sym_ptr
            )

        # receive an already existing symbol from GMD
        if (
            not (identifier or dimension or equtype or explanatory_text)
            and sym_ptr
        ):
            rc, subtype, _, _ = gmdSymbolInfo(
                self._database._gmd, self._sym_ptr, GMD_USERINFO
            )
            self._database._check_for_gmd_error(rc)
            self._equtype = subtype

        # create new equation in GMD
        elif (
            not sym_ptr
            and identifier
            and dimension is not None
            and equtype is not None
        ):
            self._equtype = equtype
            rc = new_intp()
            self._sym_ptr = gmdAddSymbolPy(
                self._database._gmd,
                self._name,
                self._dim,
                dt_equ,
                self._equtype,
                self._text,
                rc,
            )
            self._database._check_for_gmd_error(_int_value_and_free(rc))

        # create new equation with domain information
        elif (
            identifier
            and (equtype is not None)
            and (domains is not None)
            and not dimension
        ):
            if not isinstance(domains, (list, tuple)):
                raise GamspyException(
                    "Parameter domains has to be a list or a tuple"
                )

            self._equtype = equtype
            if len(domains) == 0:
                rc = new_intp()
                self._sym_ptr = gmdAddSymbolPy(
                    self._database._gmd,
                    self._name,
                    self._dim,
                    dt_equ,
                    self._equtype,
                    self._text,
                    rc,
                )
                self._database._check_for_gmd_error(_int_value_and_free(rc))
            else:
                dom_ptr = [None] * self._dim
                rel_dom = [""] * self._dim

                for i in range(self._dim):
                    if isinstance(domains[i], GamsSet):
                        dom_ptr[i] = domains[i]._sym_ptr
                    elif isinstance(domains[i], str):
                        rel_dom[i] = domains[i]
                    else:
                        raise GamspyException(
                            "Domain must be GamsSet or string but saw "
                            + str(type(domains[i]))
                            + " on index "
                            + str(i)
                        )
                rc = new_intp()
                self._sym_ptr = gmdAddSymbolXPy(
                    self._database._gmd,
                    self._name,
                    self._dim,
                    dt_equ,
                    self._equtype,
                    self._text,
                    dom_ptr,
                    rel_dom,
                    rc,
                )
                self._database._check_for_gmd_error(_int_value_and_free(rc))

        else:
            raise GamspyException("Invalid combination of parameters")

    def get_equtype(self):
        return self._equtype


class Database:
    """Communicates data between the Python and the GAMS"""

    def __init__(self, ws: Workspace):
        self.workspace = ws
        self._gmd = new_gmdHandle_tp()
        self._record_lock = False
        self._symbol_lock = False

        ret = gmdCreateD(
            self._gmd, utils._get_gamspy_base_directory(), GMS_SSSIZE
        )
        if not ret[0]:
            raise GamspyException(ret[1])

        rc = gmdSetSpecialValues(self._gmd, _spec_values)
        self._check_for_gmd_error(rc)

    def __del__(self):
        try:
            if self._gmd is not None and gmdHandleToPtr(self._gmd) is not None:
                gmdFree(self._gmd)
        except Exception:
            pass

    def __len__(self):
        return self.number_symbols

    def __getitem__(self, symbol_identifier):
        return self.get_symbol(symbol_identifier)

    def __iter__(self):
        self._position = -1
        return self

    def __next__(self):
        return self.next()

    def next(self):
        self._position += 1
        rc, nr_symbols, _, _ = gmdInfo(self._gmd, GMD_NRSYMBOLS)
        if not rc:
            raise StopIteration
        if self._position >= nr_symbols:
            raise StopIteration

        rc = new_intp()
        sym_ptr = gmdGetSymbolByIndexPy(self._gmd, self._position + 1, rc)
        self._check_for_gmd_error(_int_value_and_free(rc))
        rc, type = gmdSymbolType(self._gmd, sym_ptr)
        self._check_for_gmd_error(rc)

        if type < 0:
            raise GamspyException("Cannot retrieve type of symbol")
        if dt_var == type:
            return GamsVariable(self, sym_ptr=sym_ptr)
        if dt_equ == type:
            return GamsEquation(self, sym_ptr=sym_ptr)
        if dt_par == type:
            return GamsParameter(self, sym_ptr=sym_ptr)
        if dt_set == type or dt_alias == type:
            return GamsSet(self, sym_ptr=sym_ptr)
        raise GamspyException("Unknown symbol type " + str(type))

    def _check_for_gmd_error(self, rc, workspace=None):
        if not rc:
            msg = gmdGetLastError(self._gmd)[1]
            raise GamspyException(msg, workspace)

    @property
    def number_symbols(self):
        ret = gmdInfo(self._gmd, GMD_NRSYMBOLS)
        self._check_for_gmd_error(ret[0])
        return ret[1]

    def get_symbol(self, symbol_identifier):
        rc = new_intp()
        sym_ptr = gmdFindSymbolPy(self._gmd, symbol_identifier, rc)
        self._check_for_gmd_error(_int_value_and_free(rc))
        rc, type = gmdSymbolType(self._gmd, sym_ptr)
        self._check_for_gmd_error(rc)

        type_map = {
            dt_equ: self.get_equation,
            dt_var: self.get_variable,
            dt_par: self.get_parameter,
            dt_set: self.get_set,
            dt_alias: self.get_set,
        }

        try:
            return type_map[type](symbol_identifier)
        except KeyError as e:
            raise GamspyException(f"Unknown symbol type {type}") from e

    def get_equation(self, equation_identifier):
        rc = new_intp()
        sym_ptr = gmdFindSymbolPy(self._gmd, equation_identifier, rc)
        self._check_for_gmd_error(_int_value_and_free(rc))
        rc, type = gmdSymbolType(self._gmd, sym_ptr)
        self._check_for_gmd_error(rc)
        if type != dt_equ:
            raise GamspyException(
                "Database: Symbol "
                + equation_identifier
                + " is not an equation"
            )
        return GamsEquation(self, sym_ptr=sym_ptr)

    def get_parameter(self, parameter_identifier):
        rc = new_intp()
        sym_ptr = gmdFindSymbolPy(self._gmd, parameter_identifier, rc)
        self._check_for_gmd_error(_int_value_and_free(rc))
        rc, type = gmdSymbolType(self._gmd, sym_ptr)
        self._check_for_gmd_error(rc)
        if type != dt_par:
            raise GamspyException(
                f"Symbol {parameter_identifier} is not a parameter."
            )
        return GamsParameter(self, sym_ptr=sym_ptr)

    def get_variable(self, variable_identifier):
        rc = new_intp()
        sym_ptr = gmdFindSymbolPy(self._gmd, variable_identifier, rc)
        self._check_for_gmd_error(_int_value_and_free(rc))
        rc, type = gmdSymbolType(self._gmd, sym_ptr)
        self._check_for_gmd_error(rc)
        if type != dt_var:
            raise GamspyException(
                f"Symbol {variable_identifier} is not a variable."
            )
        return GamsVariable(self, sym_ptr=sym_ptr)

    def get_set(self, set_identifier):
        rc = new_intp()
        sym_ptr = gmdFindSymbolPy(self._gmd, set_identifier, rc)
        self._check_for_gmd_error(_int_value_and_free(rc))
        rc, type = gmdSymbolType(self._gmd, sym_ptr)
        if type != dt_set and type != dt_alias:
            raise GamspyException(f"Symbol {set_identifier} is not a set.")
        return GamsSet(self, sym_ptr=sym_ptr)

    def add_equation(
        self, identifier, dimension, equtype, explanatory_text=""
    ):
        if self._symbol_lock:
            raise GamspyException(
                "Cannot add symbols to symbol-locked database"
            )
        return GamsEquation(
            self, identifier, dimension, equtype, explanatory_text
        )

    def add_variable(
        self, identifier, dimension, vartype, explanatory_text=""
    ):
        if self._symbol_lock:
            raise GamspyException(
                "Cannot add symbols to symbol-locked database"
            )
        return GamsVariable(
            self, identifier, dimension, vartype, explanatory_text
        )

    def add_set(self, identifier, dimension, explanatory_text="", settype=0):
        if self._symbol_lock:
            raise GamspyException(
                "Cannot add symbols to symbol-locked database"
            )
        return GamsSet(
            self, identifier, dimension, explanatory_text, settype=settype
        )

    def add_parameter(self, identifier, dimension, explanatory_text=""):
        if self._symbol_lock:
            raise GamspyException(
                "Cannot add symbols to symbol-locked database"
            )
        return GamsParameter(self, identifier, dimension, explanatory_text)

    def export(self, file_path: str):
        """Writes database into a GDX file"""
        if not self.check_domains():
            raise GamspyException("Domain violations in the Database")

        file_path = os.path.splitext(file_path)[0] + ".gdx"
        if os.path.isabs(file_path):
            rc = gmdWriteGDX(self._gmd, file_path, False)
        else:
            rc = gmdWriteGDX(
                self._gmd,
                os.path.join(self.workspace.working_directory, file_path),
                False,
            )
        self._check_for_gmd_error(rc)

    def check_domains(self):
        """Check for all symbols if all records are within the specified domain of the symbol"""
        has_violation = new_intp()
        rc = gmdCheckDBDV(self._gmd, has_violation)
        self._check_for_gmd_error(rc)
        return _int_value_and_free(has_violation) != 1
