from typing import TYPE_CHECKING

import gams.transfer as gt

import gamspy._algebra.condition as condition

if TYPE_CHECKING:
    from gamspy import Container


class UniverseAlias(gt.UniverseAlias):
    def __init__(self, container: "Container", name: str):
        super().__init__(container, name)

        # allow conditions
        self.where = condition.Condition(self)

        # add statement
        self.container._addStatement(self)

        # iterator index
        self._current_index = 0

    def __len__(self):
        if self.records is not None:
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
