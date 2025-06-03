from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
import gamspy._symbols as syms
import gamspy._validation as validation
import gamspy.utils as utils
from gamspy._symbols.implicits.implicit_symbol import ImplicitSymbol

if TYPE_CHECKING:
    import pandas as pd

    from gamspy import Alias, Set
    from gamspy._algebra.expression import Expression
    from gamspy._types import EllipsisType


class ImplicitSet(ImplicitSymbol, operable.Operable):
    """
    Implicit Set

    Parameters
    ----------
    container : Container
    name : str
    domain : list[Set | str], optional
    """

    def __init__(
        self,
        parent: Set | Alias,
        name: str,
        domain: list[Set | str] = ["*"],
        scalar_domains: list[tuple[int, Set]] | None = None,
        extension: str | None = None,
    ) -> None:
        super().__init__(
            parent, name, domain, parent_scalar_domains=scalar_domains
        )
        self.extension = extension

    def __ge__(self, other) -> Expression:
        return expression.Expression(self, ">=", other)

    def __le__(self, other) -> Expression:
        return expression.Expression(self, "<=", other)

    def __repr__(self) -> str:
        return f"ImplicitSet(parent={self.parent}, name='{self.name}', domain={self.domain}, extension={self.extension}, parent_scalar_domains={self.parent_scalar_domains})"

    def __getitem__(
        self, indices: Sequence | str | EllipsisType | slice
    ) -> ImplicitSet:
        domain = validation.validate_domain(self, indices)
        return ImplicitSet(
            parent=self.parent,
            name=self.name,
            domain=domain,
            scalar_domains=self._scalar_domains,
        )

    @property
    def records(self) -> pd.DataFrame | None:
        if self.parent.records is None:
            return None

        recs = self.parent.records
        for idx, literal in self._scalar_domains:
            column_name = recs.columns[idx]
            recs = recs[recs[column_name] == literal]

        return recs

    def latexRepr(self):
        name = self.name.replace("_", "\\_")
        representation = name

        if self.extension is not None:
            representation += f"{self.extension}"

        domain = list(self.domain)

        for i, d in self._scalar_domains:
            domain.insert(i, d)

        if domain != ["*"]:
            set_strs = []
            for elem in domain:
                if isinstance(elem, (syms.Set, syms.Alias, ImplicitSet)):
                    set_strs.append(elem.latexRepr())
                elif isinstance(elem, str):
                    set_strs.append(
                        f"\\textquotesingle {elem} \\textquotesingle"
                    )

            domain_str = "{" + ",".join(set_strs) + "}"
            representation = f"{representation}_{domain_str}"

        return representation

    def gamsRepr(self) -> str:
        representation = self.name

        if self.extension is not None:
            representation += f"{self.extension}"

        if self.domain != ["*"]:
            domain = list(self.domain)
            for i, d in self._scalar_domains:
                domain.insert(i, d)

            representation += utils._get_domain_str(domain)

        return representation
