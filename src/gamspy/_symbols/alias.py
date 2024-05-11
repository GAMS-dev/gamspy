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

import uuid
import warnings
from typing import TYPE_CHECKING

import gams.transfer as gt
from gams.core.gdx import GMS_DT_ALIAS

import gamspy as gp
import gamspy._algebra.condition as condition
import gamspy._algebra.operable as operable
import gamspy._validation as validation
from gamspy._symbols.set import SetMixin
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from gamspy import Container, Set


class Alias(gt.Alias, operable.Operable, SetMixin):
    """
    Represents an Alias symbol in GAMS.
    https://www.gams.com/latest/docs/UG_SetDefinition.html#UG_SetDefinition_TheAliasStatementMultipleNamesForASet

    Parameters
    ----------
    container : Container
        Container of the alias.
    name : str, optional
        Name of the alias.
    alias_with : Set
        Alias set object.

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i")
    >>> j = gp.Alias(m, "j", i)

    """

    @classmethod
    def _constructor_bypass(
        cls, container: Container, name: str, alias_with: Set
    ):
        # create new symbol object
        obj = Alias.__new__(cls, container, name, alias_with)

        # set private properties directly
        obj._requires_state_check = False
        obj._container = container
        container._requires_state_check = True
        obj._name = name
        obj._alias_with = alias_with
        obj._modified = True

        # typing
        obj._gams_type = GMS_DT_ALIAS
        obj._gams_subtype = 1

        # add to container
        container.data.update({name: obj})

        # gamspy attributes
        obj._is_dirty = False
        obj.where = condition.Condition(obj)
        obj.container._add_statement(obj)
        obj._current_index = 0

        return obj

    def __new__(
        cls,
        container: Container,
        name: str | None = None,
        alias_with: Set | None = None,
    ):
        if not isinstance(container, gp.Container):
            raise TypeError(
                "Container must of type `Container` but found"
                f" {type(container)}"
            )

        if not isinstance(alias_with, (gp.Set, gp.Alias)):
            raise TypeError(
                f"alias_with must be a Set but found {type(alias_with)}"
            )

        if name is None:
            return object.__new__(cls)
        else:
            if not isinstance(name, str):
                raise TypeError(
                    f"Name must of type `str` but found {type(name)}"
                )
            try:
                symobj = container[name]
                if isinstance(symobj, cls):
                    return symobj

                raise TypeError(
                    f"Cannot overwrite symbol `{name}` in container"
                    " because it is not an Alias object)"
                )
            except KeyError:
                return object.__new__(Alias)

    def __init__(
        self, container: Container, name: str | None, alias_with: Set
    ):
        # does symbol exist
        has_symbol = False
        if isinstance(getattr(self, "container", None), gp.Container):
            has_symbol = True

        if has_symbol:
            # reset some properties
            self._requires_state_check = True
            self.container._requires_state_check = True
            self.modified = True
            self.alias_with = alias_with
        else:
            self._is_dirty = False

            if name is not None:
                name = validation.validate_name(name)
            else:
                name = str(uuid.uuid4()).replace("-", "_")

            super().__init__(container, name, alias_with)

            validation.validate_container(self, self.domain)
            self.where = condition.Condition(self)
            self.container._add_statement(self)

            self.container._run()

    def __len__(self):
        if self.records is not None:
            return len(self.records.index)

        return 0

    def __bool__(self):
        raise ValidationError(
            "Alias cannot be used as a truth value. Use len(<symbol>.records) instead."
        )

    def gamsRepr(self) -> str:
        """
        Representation of this Alias in GAMS language.

        Returns
        -------
        str
        """
        return self.name

    def getDeclaration(self):
        """
        Declaration of the Alias in GAMS

        Returns
        -------
        str
        """
        return f"Alias({self.alias_with.name},{self.name});"

    def getStatement(self) -> str:
        """
        Statement of the Alias declaration

        Returns
        -------
        str
        """
        warnings.warn(
            "getStatement is going to be renamed in 0.12.5. Please use getDeclaration instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.getDeclaration()
