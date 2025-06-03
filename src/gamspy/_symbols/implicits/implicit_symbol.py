from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import gamspy as gp
import gamspy._algebra.condition as condition
import gamspy.utils as utils
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from gamspy import Alias, Product, Sand, Set, Smax, Smin, Sor, Sum
    from gamspy._symbols.implicits import (
        ImplicitParameter,
        ImplicitSet,
        ImplicitVariable,
    )


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

    def __bool__(self):
        raise ValidationError("A symbol cannot be used as a truth value.")

    def fix_domain_scalars(self, parent_scalar_domains):
        if len(self.domain) == 1 and self.domain[0] == "*":
            self._scalar_domains = []
            return

        bare_domain = utils._get_set(self.domain)
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
    def dimension(self) -> int:
        return self.parent.dimension - len(self._scalar_domains)

    @abstractmethod
    def gamsRepr(self):
        """Representation of the implicit symbol in GAMS"""

    def latexRepr(self):
        from .implicit_set import ImplicitSet

        name = self.name.replace("_", "\\_")
        representation = name
        domain = list(self.domain)

        for i, d in self._scalar_domains:
            domain.insert(i, d)

        if domain:
            set_strs = []
            for elem in domain:
                if isinstance(elem, (gp.Set, gp.Alias, ImplicitSet)):
                    set_strs.append(elem.latexRepr())
                elif isinstance(elem, str):
                    set_strs.append(
                        f"\\textquotesingle {elem} \\textquotesingle"
                    )

            domain_str = "{" + ",".join(set_strs) + "}"
            representation = f"{representation}_{domain_str}"

        return representation

    def sum(
        self: ImplicitSet | ImplicitParameter | ImplicitVariable,
        *indices: Set | Alias,
    ) -> Sum:
        """
        Equivalent to Sum(indices, obj[obj.domain]). For example:

        v = Variable(m, domain=[i, j, k])
        v.sum() is equivalent to Sum((i,j), v[i, j, k])
        v.sum(i) is equivalent to Sum(i, v[i, j, k])
        v.sum(i, j) is equivalent to Sum((i, j), v[i, j, k])

        Returns
        -------
        Sum
            Generated Sum gp.

        Raises
        ------
        ValidationError
            In case the symbol is scalar.
        """
        if not self.domain:
            raise ValidationError(
                "Sum operation is not possible on scalar parameters."
            )

        if not indices:
            indices = self.domain

        return gp.Sum(indices, self[self.domain])

    def product(
        self: ImplicitSet | ImplicitParameter | ImplicitVariable,
        *indices: Set | Alias,
    ) -> Product:
        """
        Equivalent to Product(indices, obj[obj.domain]). For example:

        v = Variable(m, domain=[i, j, k])
        v.product() is equivalent to Product((i,j), v[i, j, k])
        v.product(i) is equivalent to Product(i, v[i, j, k])
        v.product(i, j) is equivalent to Product((i, j), v[i, j, k])

        Returns
        -------
        Product
            Generated Product gp.

        Raises
        ------
        ValidationError
            In case the symbol is scalar.
        """
        if not self.domain:
            raise ValidationError(
                "Product operation is not possible on scalar parameters."
            )

        if not indices:
            indices = self.domain

        return gp.Product(indices, self[self.domain])

    def smin(
        self: ImplicitSet | ImplicitParameter | ImplicitVariable,
        *indices: Set | Alias,
    ) -> Smin:
        """
        Equivalent to Smin(indices, obj[obj.domain]). For example:

        v = Variable(m, domain=[i, j, k])
        v.smin() is equivalent to Smin((i,j), v[i, j, k])
        v.smin(i) is equivalent to Smin(i, v[i, j, k])
        v.smin(i, j) is equivalent to Smin((i, j), v[i, j, k])

        Returns
        -------
        Smin
            Generated Smin gp.

        Raises
        ------
        ValidationError
            In case the symbol is scalar.
        """
        if not self.domain:
            raise ValidationError(
                "Smin operation is not possible on scalar parameters."
            )

        if not indices:
            indices = self.domain

        return gp.Smin(indices, self[self.domain])

    def smax(
        self: ImplicitSet | ImplicitParameter | ImplicitVariable,
        *indices: Set | Alias,
    ) -> Smax:
        """
        Equivalent to Smax(indices, obj[obj.domain]). For example:

        v = Variable(m, domain=[i, j, k])
        v.smax() is equivalent to Smax((i,j), v[i, j, k])
        v.smax(i) is equivalent to Smax(i, v[i, j, k])
        v.smax(i, j) is equivalent to Smax((i, j), v[i, j, k])

        Returns
        -------
        Smax
            Generated Smax gp.

        Raises
        ------
        ValidationError
            In case the symbol is scalar.
        """
        if not self.domain:
            raise ValidationError(
                "Smax operation is not possible on scalar parameters."
            )

        if not indices:
            indices = self.domain

        return gp.Smax(indices, self[self.domain])

    def sand(
        self: ImplicitSet | ImplicitParameter | ImplicitVariable,
        *indices: Set | Alias,
    ) -> Sand:
        """
        Equivalent to Sand(indices, obj[obj.domain]). For example:

        v = Variable(m, domain=[i, j, k])
        v.sand() is equivalent to Sand((i,j), v[i, j, k])
        v.sand(i) is equivalent to Sand(i, v[i, j, k])
        v.sand(i, j) is equivalent to Sand((i, j), v[i, j, k])

        Returns
        -------
        Sand
            Generated Sand gp.

        Raises
        ------
        ValidationError
            In case the symbol is scalar.
        """
        if not self.domain:
            raise ValidationError(
                "Sand operation is not possible on scalar parameters."
            )

        if not indices:
            indices = self.domain

        return gp.Sand(indices, self[self.domain])

    def sor(
        self: ImplicitSet | ImplicitParameter | ImplicitVariable,
        *indices: Set | Alias,
    ) -> Sor:
        """
        Equivalent to Sor(indices, obj[obj.domain]). For example:

        v = Variable(m, domain=[i, j, k])
        v.sor() is equivalent to Sor((i,j), v[i, j, k])
        v.sor(i) is equivalent to Sor(i, v[i, j, k])
        v.sor(i, j) is equivalent to Sor((i, j), v[i, j, k])

        Returns
        -------
        Sor
            Generated Sor gp.

        Raises
        ------
        ValidationError
            In case the symbol is scalar.
        """
        if not self.domain:
            raise ValidationError(
                "Sor operation is not possible on scalar parameters."
            )

        if not indices:
            indices = self.domain

        return gp.Sor(indices, self[self.domain])
