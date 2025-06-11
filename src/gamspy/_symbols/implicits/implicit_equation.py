from __future__ import annotations

from typing import TYPE_CHECKING

import gamspy._symbols.alias as alias
import gamspy._symbols.implicits as implicits
import gamspy._symbols.set as gams_set
from gamspy._symbols.implicits.implicit_symbol import ImplicitSymbol

if TYPE_CHECKING:
    import pandas as pd

    from gamspy import Equation, Set
    from gamspy._algebra.expression import Expression


class ImplicitEquation(ImplicitSymbol):
    def __init__(
        self,
        parent: Equation,
        name: str,
        type: str,
        domain: list[Set | str],
    ) -> None:
        """Implicit Equation

        Parameters
        ----------
        parent : Equation
        name : str
        domain : List[Set | str]
        """
        super().__init__(parent, name, domain)
        self.type = type

    def __repr__(self) -> str:
        return f"ImplicitEquation(parent={self.parent}, name='{self.name}', domain={self.domain}, type={self.type})"

    @property
    def l(self):  # noqa: E741,E743
        return implicits.ImplicitParameter(
            self.parent,
            name=self.parent.l.name,
            domain=self.domain,
            scalar_domains=self._scalar_domains,
        )

    @l.setter
    def l(self, value: int | float | Expression):
        # b[t].l = 30 -> b.l[t] = 30
        self.parent.l[self.domain] = value

    @property
    def m(self):
        return implicits.ImplicitParameter(
            self.parent,
            name=self.parent.m.name,
            domain=self.domain,
            scalar_domains=self._scalar_domains,
        )

    @m.setter
    def m(self, value: int | float | Expression):
        # b[t].m = 30 -> b.m[t] = 30
        self.parent.m[self.domain] = value

    @property
    def lo(self):
        return implicits.ImplicitParameter(
            self.parent,
            name=self.parent.lo.name,
            domain=self.domain,
            scalar_domains=self._scalar_domains,
        )

    @lo.setter
    def lo(self, value: int | float | Expression):
        # b[t].lo = 30 -> b.lo[t] = 30
        self.parent.lo[self.domain] = value

    @property
    def up(self):
        return implicits.ImplicitParameter(
            self.parent,
            name=self.parent.up.name,
            domain=self.domain,
            scalar_domains=self._scalar_domains,
        )

    @up.setter
    def up(self, value: int | float | Expression):
        # b[t].up = 30 -> b.up[t] = 30
        self.parent.up[self.domain] = value

    @property
    def scale(self):
        return implicits.ImplicitParameter(
            self.parent,
            name=self.parent.scale.name,
            domain=self.domain,
            scalar_domains=self._scalar_domains,
        )

    @scale.setter
    def scale(self, value: int | float | Expression):
        # b[t].scale = 30 -> b.scale[t] = 30
        self.parent.scale[self.domain] = value

    @property
    def stage(self):
        return implicits.ImplicitParameter(
            self.parent,
            name=self.parent.stage.name,
            domain=self.domain,
            scalar_domains=self._scalar_domains,
        )

    @stage.setter
    def stage(self, value: int | float | Expression):
        # b[t].stage = 30 -> b.stage[t] = 30
        self.parent.stage[self.domain] = value

    @property
    def range(self):
        return self.parent.range

    @property
    def slacklo(self):
        return self.parent.slacklo

    @property
    def slackup(self):
        return self.parent.slackup

    @property
    def slack(self):
        return self.parent.slack

    @property
    def records(self) -> pd.DataFrame | float | None:
        if self.parent.records is None:
            return None

        recs = self.parent.records
        for idx, literal in self._scalar_domains:
            column_name = recs.columns[idx]
            recs = recs[recs[column_name] == literal]

        return recs

    def gamsRepr(self) -> str:
        representation = f"{self.name}"
        domain = list(self.domain)
        for i, d in self._scalar_domains:
            domain.insert(i, d)

        if len(domain):
            set_strs = []
            for set in domain:
                if isinstance(
                    set, (gams_set.Set, alias.Alias, implicits.ImplicitSet)
                ):
                    set_strs.append(set.gamsRepr())
                elif isinstance(set, str):
                    if set == "*":
                        set_strs.append(set)
                    else:
                        set_strs.append('"' + set + '"')

            domain_str = "(" + ",".join(set_strs) + ")"

            representation += domain_str

        return representation
