from __future__ import annotations

from typing import TYPE_CHECKING

import gams.transfer as gt
from gams.core.gdx import GMS_DT_ALIAS

import gamspy as gp
import gamspy._algebra.condition as condition
import gamspy._validation as validation

if TYPE_CHECKING:
    from gamspy import Container


class UniverseAlias(gt.UniverseAlias):
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

    def __new__(cls, container: Container, name: str = "universe"):
        if not isinstance(container, gp.Container):
            raise TypeError(
                "Container must of type `Container` but found"
                f" {type(container)}"
            )

        if not isinstance(name, str):
            raise TypeError(f"Name must of type `str` but found {type(name)}")

        try:
            symbol = container[name]
            if isinstance(symbol, cls):
                return symbol

            raise TypeError(
                f"Cannot overwrite symbol `{name}` in container"
                " because it is not a UniverseAlias object)"
            )
        except KeyError:
            return object.__new__(cls)

    def __init__(self, container: Container, name: str = "universe"):
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
        >>> universe = gp.UniverseAlias(m, "universe")
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
        # check if the name is a reserved word
        name = validation.validate_name(name)

        super().__init__(container, name)

        # allow conditions
        self.where = condition.Condition(self)

        # iterator index
        self._current_index = 0

    def __len__(self):
        if not self.records.empty:
            return len(self.records.index)

        return 0
