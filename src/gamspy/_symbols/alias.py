from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import gams.transfer as gt
from gams.core.gdx import GMS_DT_ALIAS

import gamspy as gp
import gamspy._algebra.condition as condition
import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
import gamspy._symbols.implicits as implicits
import gamspy._validation as validation
import gamspy.utils as utils
from gamspy._symbols.set import SetMixin
from gamspy._symbols.symbol import Symbol
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from gamspy import Container, Set


class Alias(gt.Alias, operable.Operable, Symbol, SetMixin):
    """
    Represents an Alias symbol in GAMS.
    https://gamspy.readthedocs.io/en/latest/user/basics/alias.html

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
        cls, container: Container, name: str, alias_with: Set | Alias
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
        obj.where = condition.Condition(obj)
        obj.container._add_statement(obj)
        obj._current_index = 0

        return obj

    def __new__(
        cls,
        container: Container,
        name: str | None = None,
        alias_with: Set | Alias | None = None,
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
        self,
        container: Container,
        name: str | None = None,
        alias_with: Set | Alias = None,
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
            if name is not None:
                name = validation.validate_name(name)
            else:
                name = "a" + str(uuid.uuid4()).replace("-", "_")

            super().__init__(container, name, alias_with)

            validation.validate_container(self, self.domain)
            self.where = condition.Condition(self)
            self.container._add_statement(self)

            self.container._synch_with_gams()

    def __len__(self):
        if self.records is not None:
            return len(self.records.index)

        return 0

    def __bool__(self):
        raise ValidationError(
            "Alias cannot be used as a truth value. Use len(<symbol>.records) instead."
        )

    def __repr__(self) -> str:
        return f"Alias(name='{self.name}', alias_with={self.alias_with})"

    def __getitem__(self, indices: tuple | str) -> implicits.ImplicitSet:
        domain = validation.validate_domain(self, indices)

        return implicits.ImplicitSet(self, name=self.name, domain=domain)

    def __setitem__(self, indices: tuple | str, rhs):
        # self[domain] = rhs
        domain = validation.validate_domain(self, indices)

        if isinstance(rhs, bool):
            rhs = "yes" if rhs is True else "no"  # type: ignore

        statement = expression.Expression(
            implicits.ImplicitSet(self, name=self.name, domain=domain),
            "=",
            rhs,
        )

        statement._validate_definition(utils._unpack(domain))

        self.container._add_statement(statement)
        self._assignment = statement

        if self.alias_with.synchronize:  # type: ignore
            self.container._synch_with_gams(gams_to_gamspy=True)

    @property
    def synchronize(self):
        raise ValidationError(
            "Each Alias object is tied to a Set. Change the synchronization setting of the set instead."
        )

    @synchronize.setter
    def synchronize(self, value: bool):
        raise ValidationError(
            f"Alias `{self.name}` object is tied to a Set `{self.alias_with.name}`."
            f"Change the synchronization setting of the Set `{self.alias_with.name}` instead."
        )

    def gamsRepr(self) -> str:
        """
        Representation of this Alias in GAMS language.

        Returns
        -------
        str

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", domain=["*"], records=['i1','i2'])
        >>> j = gp.Alias(m, "j", i)
        >>> j.gamsRepr()
        'j'

        """
        return self.name

    def getDeclaration(self):
        """
        Declaration of the Alias in GAMS

        Returns
        -------
        str

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=['i1','i2'])
        >>> j = gp.Alias(m, "j", i)
        >>> j.getDeclaration()
        'Alias(i,j);'

        """
        return f"Alias({self.alias_with.name},{self.name});"
