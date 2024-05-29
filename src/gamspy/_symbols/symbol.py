from __future__ import annotations

from typing import TYPE_CHECKING, Union

import gams.transfer as gt

import gamspy._symbols.implicits as implicits
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from gamspy import Alias, Equation, Parameter, Set, Variable

    SymbolType = Union[Alias, Set, Parameter, Variable, Equation]


class Symbol:
    def __bool__(self):
        raise ValidationError(
            "A symbol cannot be used as a truth value. Use len(<symbol>.records) instead."
        )

    @property
    def synchronize(self: SymbolType) -> bool:
        """
        Synchronization state of the symbol. If True, the symbol data
        will be communicated with GAMS. Otherwise, GAMS state will not be updated.

        Returns
        -------
        bool

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=["i1"])
        >>> i.synchronize = False
        >>> i["i2"] = True
        >>> i.records.uni.tolist()
        ['i1']
        >>> i.synchronize = True
        >>> i.records.uni.tolist()
        ['i1', 'i2']

        """
        return self._synchronize

    @synchronize.setter
    def synchronize(self: SymbolType, value: bool):
        """
        If set to True, the current state will be synchronized with GAMS.
        Else, the symbol will not be synchronized with GAMS.
        """
        if value:
            self._synchronize = True
            self.container._run()
        else:
            self._synchronize = False

    def _get_domain_str(self: SymbolType):
        if isinstance(self.domain_forwarding, bool):
            forwarding = [self.domain_forwarding] * self.dimension
        else:
            forwarding = self.domain_forwarding

        set_strs = []
        for elem, is_forwarding in zip(self.domain, forwarding):
            if isinstance(elem, (gt.Set, gt.Alias, implicits.ImplicitSet)):
                set_str = elem.gamsRepr()
                set_strs.append(set_str + "<" if is_forwarding else set_str)
            elif isinstance(elem, str):
                set_strs.append("*")

        return "(" + ",".join(set_strs) + ")"

    def _mark_forwarded_domain_sets(self):
        for elem in self.domain:
            if hasattr(elem, "modified"):
                elem.modified = False
                elem._is_dirty = True
