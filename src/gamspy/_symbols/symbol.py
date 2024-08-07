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

    def latexRepr(self: SymbolType):
        """
        Representation of symbol in Latex.

        Returns
        -------
        str
        """
        name = self.name.replace("_", "\_")
        representation = f"\\text{{{name}}}"
        domain = list(self.domain)

        if hasattr(self, "_scalar_domains"):
            for i, d in self._scalar_domains:
                domain.insert(i, d)

        if domain and domain != ["*"]:
            set_strs = []
            for elem in domain:
                if isinstance(elem, (gt.Set, gt.Alias, implicits.ImplicitSet)):
                    set_strs.append(elem.latexRepr())
                elif isinstance(elem, str):
                    set_strs.append("*")

            domain_str = ",".join(set_strs)
            representation = f"{representation}_{{{domain_str}}}"

        return representation

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
            self.container._synch_with_gams()
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

    def _mark_forwarded_domain_sets(
        self: SymbolType, domain_forwarding: list[bool] | bool
    ):
        if isinstance(domain_forwarding, bool):
            domain_forwarding = [domain_forwarding]

        for elem, forwarding in zip(self.domain, domain_forwarding):
            if hasattr(elem, "modified") and forwarding:
                elem.modified = False
