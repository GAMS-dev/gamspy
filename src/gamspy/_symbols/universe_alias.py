from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING

import gams.transfer as gt
from gams.core.gdx import GMS_DT_ALIAS

import gamspy as gp
import gamspy._algebra.condition as condition
import gamspy._validation as validation
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from gamspy import Container


class UniverseAlias(gt.UniverseAlias):
    """
    Represents a UniverseAlias symbol in GAMS.

    Parameters
    ----------
    container : Container
    name : str

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> universe = gp.UniverseAlias(m)
    >>> universe.records
    Empty DataFrame
    Columns: [uni]
    Index: []
    >>> i = gp.Set(m, "i", records=['i1', 'i2'])
    >>> universe.records
      uni
    0  i1
    1  i2

    """

    @classmethod
    def _constructor_bypass(cls, container: Container, name: str):
        # create new symbol object
        obj = UniverseAlias.__new__(cls, container, name)

        # set private properties directly
        obj._requires_state_check = False
        obj._container = container
        container._requires_state_check = True
        obj._name = name
        obj._modified = True

        # typing
        obj._gams_type = GMS_DT_ALIAS
        obj._gams_subtype = 0

        # add to container
        container.data.update({name: obj})

        # gamspy attributes
        obj.where = condition.Condition(obj)

        # add statement
        obj.container._add_statement(obj)

        return obj

    def __new__(
        cls, container: Container | None = None, name: str = "universe"
    ):
        if container is not None and not isinstance(container, gp.Container):
            raise TypeError(
                "Container must of type `Container` but found"
                f" {type(container)}"
            )

        if not isinstance(name, str):
            raise TypeError(f"Name must of type `str` but found {type(name)}")

        try:
            if not container:
                container = gp._ctx_managers[
                    (os.getpid(), threading.get_native_id())
                ]

            symbol = container[name]
            if isinstance(symbol, cls):
                return symbol

            raise TypeError(
                f"Cannot overwrite symbol `{name}` in container"
                " because it is not a UniverseAlias object)"
            )
        except KeyError:
            return object.__new__(cls)

    def __init__(
        self, container: Container | None = None, name: str = "universe"
    ):
        # check if the name is a reserved word
        name = validation.validate_name(name)
        if container is None:
            try:
                container = gp._ctx_managers[
                    (os.getpid(), threading.get_native_id())
                ]
            except KeyError as e:
                raise ValidationError(
                    "UniverseAlias requires a container."
                ) from e

        super().__init__(container, name)

        # allow conditions
        self.where = condition.Condition(self)

        self.container._add_statement(self)
        self.container._synch_with_gams()

    def _serialize(self) -> dict:
        return dict()

    def _deserialize(self, info: dict) -> None: ...

    def __repr__(self) -> str:
        return f"UniverseAlias(name='{self.name}')"

    def gamsRepr(self) -> str:
        """
        Representation of the UniverseAlias in GAMS language.

        Returns
        -------
        str

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.UniverseAlias(m, name="universe")
        >>> i.gamsRepr()
        'universe'

        """
        return self.name

    def getDeclaration(self) -> str:
        """
        Declaration of the UniverseAlias in GAMS

        Returns
        -------
        str

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.UniverseAlias(m, name="universe")
        >>> i.getDeclaration()
        'Alias(universe,*);'

        """
        return f"Alias({self.name},*);"
