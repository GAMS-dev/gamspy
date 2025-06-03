from __future__ import annotations

from typing import TYPE_CHECKING, Union

import gamspy as gp
import gamspy._symbols.implicits as implicits
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from gamspy import (
        Alias,
        Equation,
        Parameter,
        Product,
        Sand,
        Set,
        Smax,
        Smin,
        Sor,
        Sum,
        Variable,
    )

    SymbolType = Union[Alias, Set, Parameter, Variable, Equation]


class Symbol:
    def __bool__(self):
        raise ValidationError("A symbol cannot be used as a truth value.")

    def gamsRepr(self):
        """Representation of the implicit symbol in GAMS"""

    def _serialize(self):
        """Serializes the symbol into a dict"""

    def _deserialize(self, info: dict):
        """Deserializes given info into a symbol"""

    def latexRepr(self: Set | Alias | Parameter | Variable | Equation):
        """
        Representation of symbol in Latex.

        Returns
        -------
        str
        """
        name = self.name.replace("_", "\\_")
        return name

    @property
    def synchronize(
        self: Set | Alias | Parameter | Variable | Equation,
    ) -> bool:
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
    def synchronize(
        self: Set | Alias | Parameter | Variable | Equation, value: bool
    ) -> None:
        """
        If set to True, the current state will be synchronized with GAMS.
        Else, the symbol will not be synchronized with GAMS.
        """
        if value:
            self._synchronize = True

            if self._winner == "python":
                self.modified = True
                self.container._synch_with_gams()
            else:
                self.container.loadRecordsFromGdx(
                    load_from=self.container._gdx_out, symbol_names=[self.name]
                )
        else:
            self._synchronize = False

    def _get_domain_str(
        self: Set | Parameter | Variable | Equation,
        forwardings: bool | list[bool],
    ) -> str:
        if isinstance(forwardings, bool):
            forwardings = [forwardings] * self.dimension

        set_strs = []
        for elem, forwarding in zip(self.domain, forwardings):
            if isinstance(elem, (gp.Set, gp.Alias, implicits.ImplicitSet)):
                elem_str = elem.gamsRepr()
                if forwarding:
                    elem_str += "<"
                set_strs.append(elem_str)
            elif isinstance(elem, (str, gp.UniverseAlias)):
                set_strs.append("*")

        return "(" + ",".join(set_strs) + ")"

    def sum(
        self: Set | Alias | Parameter | Variable,
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
            Generated Sum operation.

        Raises
        ------
        ValidationError
            In case the symbol is scalar.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i")
        >>> j = gp.Set(m, "j")
        >>> x = gp.Parameter(m, "x", domain=[i, j])
        >>> y = gp.Parameter(m, "y")
        >>> x.sum().gamsRepr()
        'sum((i,j),x(i,j))'
        >>> gp.Sum((i, j), x[i, j]).gamsRepr()
        'sum((i,j),x(i,j))'

        """
        if not self.domain:
            raise ValidationError(
                "Sum operation is not possible on scalar symbols."
            )

        op_indices = indices if indices else self.domain

        return gp.Sum(op_indices, self[self.domain])

    def product(
        self: Set | Alias | Parameter | Variable,
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
            Generated Product operation.

        Raises
        ------
        ValidationError
            In case the symbol is scalar.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i")
        >>> j = gp.Set(m, "j")
        >>> x = gp.Parameter(m, "x", domain=[i, j])
        >>> y = gp.Parameter(m, "y")
        >>> x.product().gamsRepr()
        'prod((i,j),x(i,j))'
        >>> gp.Product((i, j), x[i, j]).gamsRepr()
        'prod((i,j),x(i,j))'

        """
        if not self.domain:
            raise ValidationError(
                "Product operation is not possible on scalar symbols."
            )

        op_indices = indices if indices else self.domain

        return gp.Product(op_indices, self[self.domain])

    def smin(
        self: Set | Alias | Parameter | Variable,
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
            Generated Smin operation.

        Raises
        ------
        ValidationError
            In case the symbol is scalar.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i")
        >>> j = gp.Set(m, "j")
        >>> x = gp.Parameter(m, "x", domain=[i, j])
        >>> y = gp.Parameter(m, "y")
        >>> x.smin().gamsRepr()
        'smin((i,j),x(i,j))'
        >>> gp.Smin((i, j), x[i, j]).gamsRepr()
        'smin((i,j),x(i,j))'

        """
        if not self.domain:
            raise ValidationError(
                "Smin operation is not possible on scalar symbols."
            )

        op_indices = indices if indices else self.domain

        return gp.Smin(op_indices, self[self.domain])

    def smax(
        self: Set | Alias | Parameter | Variable,
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
            Generated Smax operation.

        Raises
        ------
        ValidationError
            In case the symbol is scalar.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i")
        >>> j = gp.Set(m, "j")
        >>> x = gp.Parameter(m, "x", domain=[i, j])
        >>> y = gp.Parameter(m, "y")
        >>> x.smax().gamsRepr()
        'smax((i,j),x(i,j))'
        >>> gp.Smax((i, j), x[i, j]).gamsRepr()
        'smax((i,j),x(i,j))'

        """
        if not self.domain:
            raise ValidationError(
                "Smax operation is not possible on scalar symbols."
            )

        op_indices = indices if indices else self.domain

        return gp.Smax(op_indices, self[self.domain])

    def sand(
        self: Set | Alias | Parameter | Variable,
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
            Generated Sand operation.

        Raises
        ------
        ValidationError
            In case the symbol is scalar.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i")
        >>> j = gp.Set(m, "j")
        >>> x = gp.Parameter(m, "x", domain=[i, j])
        >>> y = gp.Parameter(m, "y")
        >>> x.sand().gamsRepr()
        'sand((i,j),x(i,j))'
        >>> gp.Sand((i, j), x[i, j]).gamsRepr()
        'sand((i,j),x(i,j))'

        """
        if not self.domain:
            raise ValidationError(
                "Sand operation is not possible on scalar symbols."
            )

        op_indices = indices if indices else self.domain

        return gp.Sand(op_indices, self[self.domain])

    def sor(
        self: Set | Alias | Parameter | Variable,
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
            Generated Sor operation.

        Raises
        ------
        ValidationError
            In case the symbol is scalar.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i")
        >>> j = gp.Set(m, "j")
        >>> x = gp.Parameter(m, "x", domain=[i, j])
        >>> y = gp.Parameter(m, "y")
        >>> x.sor().gamsRepr()
        'sor((i,j),x(i,j))'
        >>> gp.Sor((i, j), x[i, j]).gamsRepr()
        'sor((i,j),x(i,j))'

        """
        if not self.domain:
            raise ValidationError(
                "Sor operation is not possible on scalar symbols."
            )

        op_indices = indices if indices else self.domain

        return gp.Sor(op_indices, self[self.domain])
