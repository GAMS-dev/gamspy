from __future__ import annotations

from abc import ABC, abstractmethod

import gams.transfer as gt

import gamspy._algebra.condition as condition
import gamspy.utils as utils


class ImplicitSymbol(ABC):
    def __init__(
        self,
        parent,
        name,
        domain,
        permutation=None,
        parent_scalar_domains=None,
    ) -> None:
        self.parent = parent
        self.container = parent.container
        self.name = name
        self.domain = domain
        self.where = condition.Condition(self)

        if permutation is not None:
            permutation = list(permutation)

        self.permutation = permutation

        if parent_scalar_domains is None:
            parent_scalar_domains = []
        else:
            parent_scalar_domains = list(parent_scalar_domains)

        self.parent_scalar_domains = parent_scalar_domains

        self.fix_domain_scalars(parent_scalar_domains)

    def fix_domain_scalars(self, parent_scalar_domains):
        if len(self.domain) == 1 and self.domain[0] == "*":
            self._scalar_domains = []
            return

        bare_domain = utils.get_set(self.domain)
        domain = []
        scalars = []
        permutation_indices_to_del = []
        for i, d in enumerate(bare_domain):
            if isinstance(d, str):
                loc_scalar = (
                    i if self.permutation is None else self.permutation[i]
                )
                scalars.append((loc_scalar, d))
                permutation_indices_to_del.append(i)
            else:
                domain.append(d)

        permutation_indices_to_del = reversed(permutation_indices_to_del)
        if self.permutation is not None:
            for index in permutation_indices_to_del:
                del self.permutation[index]

            self.fix_permutation()

        self.domain = domain

        scalars.extend(parent_scalar_domains)
        scalars = list(sorted(scalars, key=lambda k: k[0]))
        self._scalar_domains = scalars

    def fix_permutation(self):
        numbers_pos = list(zip(self.permutation, range(len(self.permutation))))
        numbers_pos = sorted(numbers_pos, key=lambda k: k[0])
        new_perm = [-1] * len(self.permutation)
        for i, (_, pos) in enumerate(numbers_pos):
            new_perm[pos] = i

        self.permutation = new_perm

    @property
    def dimension(self):
        return self.parent.dimension - len(self._scalar_domains)

    @abstractmethod
    def gamsRepr(self):
        """Representation of the implicit symbol in GAMS"""

    def latexRepr(self):
        from .implicit_set import ImplicitSet

        name = self.name.replace("_", "\_")
        representation = f"\\text{{{name}}}"
        domain = list(self.domain)

        if hasattr(self, "_scalar_domains"):
            for i, d in self._scalar_domains:
                domain.insert(i, d)

        if domain:
            set_strs = []
            for elem in domain:
                if isinstance(elem, (gt.Set, gt.Alias, ImplicitSet)):
                    set_strs.append(elem.latexRepr())
                elif isinstance(elem, str):
                    set_strs.append("*")

            domain_str = ",".join(set_strs)
            representation = f"{representation}_{{{domain_str}}}"

        return representation
