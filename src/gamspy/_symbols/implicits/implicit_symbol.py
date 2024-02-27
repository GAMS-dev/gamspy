from __future__ import annotations

from abc import ABC, abstractmethod

import gamspy._algebra.condition as condition
import gamspy.utils as utils


class ImplicitSymbol(ABC):
    def __init__(
        self, parent, name, domain, parent_scalar_domains=None
    ) -> None:
        self.parent = parent
        self.container = parent.container
        self.name = name
        self.domain = domain
        self.where = condition.Condition(self)

        if parent_scalar_domains is None:
            parent_scalar_domains = []

        self.fix_domain_scalars(parent_scalar_domains)

    def fix_domain_scalars(self, parent_scalar_domains):
        if len(self.domain) == 1 and self.domain[0] == "*":
            self._scalar_domains = []
            return

        bare_domain = utils.get_set(self.domain)
        domain = []
        scalars = []
        for i, d in enumerate(bare_domain):
            if isinstance(d, str):
                scalars.append((i, d))
            else:
                domain.append(d)

        self.domain = domain

        scalars.extend(parent_scalar_domains)
        scalars = list(sorted(scalars, key=lambda k: k[0]))
        self._scalar_domains = scalars

    @property
    def dimension(self):
        return self.parent.dimension - len(self._scalar_domains)

    @abstractmethod
    def gamsRepr(self):
        """Representation of the implicit symbol in GAMS"""
