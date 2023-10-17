from typing import TYPE_CHECKING

import gams.transfer as gt

import gamspy._algebra.condition as condition
import gamspy.utils as utils

if TYPE_CHECKING:
    from gamspy import Container


class UniverseAlias(gt.UniverseAlias):
    def __new__(cls, *args, **kwargs):
        if len(args) == 0:
            return object.__new__(UniverseAlias)

        try:
            symobj = args[0][args[1]]
        except:
            symobj = None

        if symobj is None:
            return object.__new__(UniverseAlias)
        else:
            if isinstance(symobj, UniverseAlias):
                return symobj
            else:
                raise TypeError(
                    f"Cannot overwrite symbol `{symobj.name}` in container because it is not a UniverseAlias object)"
                )

    def __init__(self, container: "Container", name: str = "universe"):
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
        name = utils._reservedCheck(name)

        super().__init__(container, name)

        # allow conditions
        self.where = condition.Condition(self)

        # add statement
        self.container._addStatement(self)

        # iterator index
        self._current_index = 0

    def __len__(self):
        if not self.records.empty:
            return len(self.records.index)

        return 0

    def __next__(self):
        if self._current_index < len(self):
            row = self.records.iloc[self._current_index]
            self._current_index += 1
            return row

        self._current_index = 0
        raise StopIteration

    def __iter__(self):
        return self
