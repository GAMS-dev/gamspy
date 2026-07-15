from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast, no_type_check

import numpy as np
import pandas as pd

import gamspy as gp
import gamspy.utils as utils
from gamspy._algorithms import generate_unique_labels
from gamspy._internals import (
    GAMS_MAX_INDEX_DIM,
    DomainStatus,
)
from gamspy._records_ingestion import VarEquIngestor
from gamspy._special_values import SpecialValues
from gamspy._symbols.equals import equals_variable
from gamspy._symbols.generate_records import generate_records_variable
from gamspy._symbols.pivot import pivot_variable
from gamspy.exceptions import GamspyException, ValidationError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from scipy.sparse import coo_matrix

    from gamspy import (
        Alias,
        Container,
        Equation,
        Parameter,
        Product,
        Sand,
        Set,
        Smax,
        Smin,
        Sor,
        Sum,
        UniverseAlias,
        Variable,
    )
    from gamspy._types import (
        DomainType,
        NormalizedDomainType,
        SymbolType,
        SymbolWithRecordsType,
    )


@dataclass(frozen=True, slots=True)
class DomainViolation:
    symbol: gp.Set | gp.Parameter | gp.Variable | gp.Equation
    dimension: int
    domain: gp.Set | gp.Alias | gp.UniverseAlias
    violations: Any


class BaseSymbol:
    def __bool__(self):
        raise ValidationError("A symbol cannot be used as a truth value.")

    def __len__(self: SymbolType):
        if self.records is not None:
            return len(self.records.index)

        return 0

    def latexRepr(self: SymbolType):
        """
        Representation of symbol in Latex.

        Returns
        -------
        str

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i")
        >>> print(i.latexRepr())
        i

        """
        return self._latex_name

    def toGraph(self: SymbolType):
        """
        Return a ``graphviz.Digraph`` of this symbol's expression tree.

        For an :class:`~gamspy.Equation` this is the ``..`` definition; for a
        :class:`~gamspy.Parameter`/:class:`~gamspy.Variable` (and attribute
        assignments) it is the latest assignment. Requires the optional
        ``graphviz`` dependency (``pip install gamspy[graph]``).

        Returns
        -------
        graphviz.Digraph

        Raises
        ------
        ValidationError
            If the symbol has no assignment or definition to graph.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=["i1", "i2"])
        >>> v = gp.Variable(m, "v", domain=[i])
        >>> a = gp.Parameter(m, "a", domain=[i])
        >>> e = gp.Equation(m, "e", domain=[i])
        >>> e[i] = a[i] <= v[i]
        >>> graph = e.toGraph()  # doctest: +SKIP

        """
        import gamspy._algebra.expression as expression

        definition = getattr(self, "_definition", None)
        if definition is not None:
            return expression.create_graph(definition)

        assignment = getattr(self, "_assignment", None)
        if assignment is not None:
            return expression.create_graph(assignment)

        raise ValidationError(
            f"'{self.name}' has no assignment or definition to graph."
        )

    @property
    def synchronize(self: SymbolType) -> bool:
        """
        Synchronization state of the symbol. If True, the symbol data
        will be communicated with GAMS. Otherwise, GAMS state will not be updated.

        Returns
        -------
        bool
        """
        return True

    @synchronize.setter
    def synchronize(self: SymbolWithRecordsType, value: bool) -> None:
        warnings.warn(
            "`symbol.synchronize` has no effect and will be deprecated in a future release.",
            category=DeprecationWarning,
            stacklevel=2,
        )

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
            raise ValidationError("Sum operation is not possible on scalar symbols.")

        op_indices = indices if indices else self.domain

        return gp.Sum(op_indices, self[self.domain])  # ty: ignore[invalid-argument-type] Invalid indices are caught in the constructor of the operation

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

        return gp.Product(op_indices, self[self.domain])  # ty: ignore[invalid-argument-type] Invalid indices are caught in the constructor of the operation

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
            raise ValidationError("Smin operation is not possible on scalar symbols.")

        op_indices = indices if indices else self.domain

        return gp.Smin(op_indices, self[self.domain])  # ty: ignore[invalid-argument-type] Invalid indices are caught in the constructor of the operation

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
            raise ValidationError("Smax operation is not possible on scalar symbols.")

        op_indices = indices if indices else self.domain

        return gp.Smax(op_indices, self[self.domain])  # ty: ignore[invalid-argument-type] Invalid indices are caught in the constructor of the operation

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
            raise ValidationError("Sand operation is not possible on scalar symbols.")

        op_indices = indices if indices else self.domain

        return gp.Sand(op_indices, self[self.domain])  # ty: ignore[invalid-argument-type] Invalid indices are caught in the constructor of the operation

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
            raise ValidationError("Sor operation is not possible on scalar symbols.")

        op_indices = indices if indices else self.domain

        return gp.Sor(op_indices, self[self.domain])  # ty: ignore[invalid-argument-type] Invalid indices are caught in the constructor of the operation

    @property
    def container(self: SymbolType) -> Container:
        """Container of the symbol"""
        return self._container


class DomainSymbol(BaseSymbol):
    """Base class for Set, Parameter, Variable, and Equation."""

    def _load_from_gams(self: Set | Parameter | Variable | Equation) -> None:
        if self._container._in_loop:
            raise ValidationError(
                "Cannot load symbol records while a loop context manager (e.g. with gp.For, gp.While, gp.Loop) is active."
            )

        from gamspy._gdx import get_records

        self._should_load_from_gams = False
        container = self._container
        gdx_out_name = "_" + utils._get_unique_name() + ".gdx"
        gdx_out_path = os.path.join(container.working_directory, gdx_out_name)
        container._add_statement(f"execute_unload '{gdx_out_path}' , {self.name};")
        container._synch_with_gams()
        records = get_records(container, gdx_out_path, symbols=[self.name])
        self._records = records[self.name]

    def _handle_domain_violations(self: Set | Parameter | Variable | Equation) -> None:
        if gp.get_option("DROP_DOMAIN_VIOLATIONS"):
            self._domain_violations = self._getDomainViolations()
            self._dropDomainViolations()

    def _handle_domain_forwarding(self: Set | Parameter | Variable | Equation) -> None:
        if self._domain_forwarding:
            if isinstance(self._domain_forwarding, bool):
                forwardings = [self._domain_forwarding] * self.dimension
            else:
                forwardings = self._domain_forwarding

            for elem, forwarding in zip(self.domain, forwardings, strict=True):
                if forwarding and elem != "*":
                    elem._should_load_from_gams = True

    def _get_domain_str(
        self: SymbolWithRecordsType, forwardings: bool | list[bool]
    ) -> str:
        import gamspy._symbols.implicits as implicits

        if isinstance(forwardings, bool):
            forwardings = [forwardings] * self.dimension

        set_strs = []
        for elem, forwarding in zip(self.domain, forwardings, strict=False):
            if isinstance(elem, (gp.Set, gp.Alias, implicits.ImplicitSet)):
                elem_str = elem.gamsRepr()
                if forwarding:
                    elem_str += "<"
                set_strs.append(elem_str)
            elif isinstance(elem, (str, gp.UniverseAlias)):
                set_strs.append("*")

        return "(" + ",".join(set_strs) + ")"

    @property
    def domain_names(self) -> list[str]:
        """String version of domain names"""
        AnyContainerDomainSymbol = (gp.Set, gp.Alias, gp.UniverseAlias)

        return [
            i.name if isinstance(i, AnyContainerDomainSymbol) else i
            for i in self.domain
        ]

    @property
    def domain_labels(self: Set | Parameter | Variable | Equation) -> list[str]:
        """The column headings for the records DataFrame"""
        if self.records is None:
            return []

        return self.records.columns.tolist()[: self.dimension]

    @domain_labels.setter
    def domain_labels(self: SymbolWithRecordsType, labels):
        if not isinstance(labels, list):
            labels = [labels]

        if len(labels) != self.dimension:
            raise ValidationError(
                "Attempting to set symbol 'domain_labels', however, len(domain_labels) != symbol dimension."
            )

        # make unique labels if necessary
        labels = generate_unique_labels(labels)

        # set the domain_labels
        if (
            self.domain_labels is not None
            and self.records is not None
            and self.records.columns.tolist() != labels + self._attributes
        ):
            self.records.columns = labels + self._attributes

    def _normalize_domain(
        self: Set | Parameter | Variable | Equation,
        container: Container,
        domain: DomainType | None,
        default: Literal["*"] | None = None,
    ) -> NormalizedDomainType:
        if domain is None:
            domain = ["*"] if default == "*" else []
        elif isinstance(domain, (gp.Set, gp.Alias, gp.UniverseAlias)):
            domain = [domain]
        elif domain == "*":
            domain = ["*"]
        elif isinstance(domain, gp.math.Dim):
            domain = gp.math._generate_dims(container, domain.dims)

        return domain

    @property
    def domain(
        self: Set | Parameter | Variable | Equation,
    ) -> NormalizedDomainType:
        """
        List of domains given either as string (* for universe set) or as reference to the Set/Alias object
        """
        return self._domain

    def _validate_domain(self, domain: NormalizedDomainType) -> NormalizedDomainType:
        AnyContainerDomainSymbol = (gp.Set, gp.Alias, gp.UniverseAlias)

        if not all(isinstance(i, (AnyContainerDomainSymbol, str)) for i in domain):
            raise TypeError(
                "All 'domain' elements must be type Set, Alias, UniverseAlias, or str"
            )

        if not all(
            i.dimension == 1 for i in domain if isinstance(i, AnyContainerDomainSymbol)
        ):
            raise ValueError("All linked 'domain' elements must have dimension == 1")

        if len(domain) > GAMS_MAX_INDEX_DIM:
            raise ValueError(f"Symbol 'domain' length cannot be > {GAMS_MAX_INDEX_DIM}")

        return domain

    @property
    def description(self: Set | Parameter | Variable | Equation) -> str:
        """Description of the symbol"""
        return self._description

    @property
    def dimension(self) -> int:
        """The dimension of symbol"""
        return len(self.domain)

    @property
    def number_records(self: Set | Parameter | Variable | Equation) -> int:
        """Number of records"""
        if self.records is None:
            return 0

        return len(self.records)

    # TODO: Legacy function from GTP. Pay the technical debt.
    @no_type_check
    def _getUELCodes(self, dimension, ignore_unused=False):
        if not isinstance(dimension, int):
            raise TypeError("Argument 'dimension' must be type int")

        if dimension >= self.dimension:
            raise ValueError(
                f"Argument 'dimension' (`{dimension}`) must be < symbol "
                f"dimension (`{self.dimension}`). (NOTE: 'dimension' is indexed from zero)"
            )

        if not isinstance(ignore_unused, bool):
            raise TypeError("Argument 'ignore_unused' must be type bool")

        cats = self._getUELs(dimension, ignore_unused=ignore_unused)
        codes = list(range(len(cats)))
        return dict(zip(cats, codes, strict=True))

    # TODO: Legacy function from GTP. Pay the technical debt.
    @no_type_check
    def _getUELs(
        self: Set | Parameter | Variable | Equation,
        dimensions: int | list | None = None,
        *,
        ignore_unused: bool = False,
    ) -> list[str]:
        """
        Gets UELs from symbol dimensions. If dimensions is None then get UELs from all dimensions (maintains order).
        The argument codes accepts a list of str UELs and will return the corresponding int; must specify a single dimension if passing codes.

        Parameters
        ----------
        dimensions : int | list, optional
            Symbols' dimensions, by default None
        ignore_unused : bool, optional
            Flag to ignore unused UELs, by default False

        Returns
        -------
        list[str]
            Only UELs in the data if ignore_unused=True, otherwise return all UELs.
        """
        if self.records is None:
            return []

        if self.dimension == 0:
            return []

        if not isinstance(dimensions, (list, int, type(None))):
            raise TypeError("Argument 'dimensions' must be type int or NoneType")

        if dimensions is None:
            dimensions = list(range(self.dimension))

        if isinstance(dimensions, int):
            dimensions = [dimensions]

        if any(not isinstance(i, int) for i in dimensions):
            raise TypeError("Argument 'dimensions' must only contain type int")

        for n in dimensions:
            if n >= self.dimension:
                raise ValueError(
                    f"Cannot access symbol 'dimension' `{n}`, because `{n}` is >= symbol "
                    f"dimension (`{self.dimension}`). (NOTE: symbol 'dimension' is indexed from zero)"
                )

        if len(dimensions) == 1:
            n = dimensions[0]
            if not ignore_unused:
                cats = self.records.iloc[:, n].cat.categories.tolist()
            else:
                used_codes = np.sort(self.records.iloc[:, n].cat.codes.unique())
                all_cats = self.records.iloc[:, n].cat.categories.tolist()
                cats = [all_cats[i] for i in used_codes]
        elif len(dimensions) > 1:
            cats = {}
            for n in dimensions:
                if not ignore_unused:
                    cats.update(dict.fromkeys(self.records.iloc[:, n].cat.categories))
                else:
                    used_codes = np.sort(self.records.iloc[:, n].cat.codes.unique())
                    all_cats = self.records.iloc[:, n].cat.categories.tolist()
                    cats.update(dict.fromkeys([all_cats[i] for i in used_codes]))

            cats = list(cats.keys())

        return cats

    def _removeUELs(
        self: Set | Parameter | Variable | Equation,
        uels: dict | list | str | None = None,
        dimensions: int | Iterable | None = None,
    ) -> None:
        if self.records is None:
            return None

        if dimensions is None:
            dimensions = range(self.dimension)
        elif isinstance(dimensions, int):
            dimensions = [dimensions]

        if any(not isinstance(i, int) for i in dimensions):
            raise TypeError("Argument 'dimensions' must only contain type int")

        for i in dimensions:
            if i >= self.dimension:
                raise ValueError(
                    f"Cannot access symbol 'dimension' `{i}`, because `{i}` is >= symbol "
                    f"dimension (`{self.dimension}`). (NOTE: symbol 'dimension' is indexed from zero)"
                )

        if uels is None:
            for n in dimensions:
                try:
                    self.records.isetitem(
                        n, self.records.iloc[:, n].cat.remove_unused_categories()
                    )
                except Exception as err:
                    raise GamspyException(
                        f"Could not remove unused UELs (categories) in symbol "
                        f"dimension `{n}`. Reason: {err}"
                    ) from err
        else:
            for n in dimensions:
                try:
                    self.records.isetitem(
                        n,
                        self.records.iloc[:, n].cat.remove_categories(
                            self.records.iloc[:, n].cat.categories.intersection(
                                set(uels)
                            )
                        ),
                    )
                except Exception as err:
                    raise GamspyException(
                        f"Could not remove unused UELs (categories) in symbol "
                        f"dimension `{n}`. Reason: {err}"
                    ) from err

    def _getDomainViolations(
        self: Set | Parameter | Variable | Equation,
    ) -> list[DomainViolation] | None:
        if self.records is None:
            return None

        AnyContainerDomainSymbol = (gp.Set, gp.Alias, gp.UniverseAlias)

        dvobjs = []
        for n, symobj in enumerate(self.domain):
            if isinstance(symobj, AnyContainerDomainSymbol):
                self_elem = pd.Series(self._getUELs(n, ignore_unused=True))

                # domain violations are generated for all elements if the domain set does not have records
                if symobj.records is not None:
                    domain_elem = pd.Series(symobj._getUELs(ignore_unused=True))
                else:
                    domain_elem = pd.Series([])

                idx = ~self_elem.map(str.casefold).isin(domain_elem.map(str.casefold))

                if any(idx):
                    dvobjs.append(
                        DomainViolation(self, n, symobj, self_elem[idx].tolist())
                    )

        return dvobjs

    def _findDomainViolations(
        self: Set | Parameter | Variable | Equation,
    ) -> pd.DataFrame | None:
        if self.records is None:
            return None

        violations = self._getDomainViolations()

        if violations:
            for n, v in enumerate(violations):
                set_v = set(v.violations)
                if n == 0:
                    idx = self.records.iloc[:, v.dimension].isin(set_v)
                else:
                    idx = (idx) | (self.records.iloc[:, v.dimension].isin(set_v))

            return self.records.loc[idx, :]

        return self.records.loc[pd.Index([]), :]

    def _dropDomainViolations(self: Set | Parameter | Variable | Equation) -> None:
        if self.records is None:
            return None

        violations = self._findDomainViolations()
        if violations is None:
            return None

        self.records.drop(index=violations.index, inplace=True)

    @property
    def domain_type(self):
        """State of the domain links"""
        return self._domain_status.name

    @property
    def _domain_status(self):
        AnyContainerDomainSymbol = (gp.Set, gp.Alias, gp.UniverseAlias)

        if (
            all(isinstance(i, AnyContainerDomainSymbol) for i in self.domain)
            and self.dimension != 0
        ):
            return DomainStatus.regular
        elif all(i == "*" for i in self.domain) or self.dimension == 0:
            return DomainStatus.none

        return DomainStatus.relaxed

    def _assert_valid_records(self: Set | Parameter | Variable | Equation):
        if self.records is None:
            return None

        # make sure and all domains have valid categories
        for i in range(self.dimension):
            if np.any(self.records.iloc[:, i].cat.codes.to_numpy() == -1):
                raise ValidationError(
                    f"Categories are missing from the data in symbol `{self.name}` (dimension {i}) -- "
                    "has resulted in `NaN` domains labels. "
                    "Cannot write symbol until domain labels have been been restored."
                )

    def getSparsity(self) -> float:
        """
        Calculates the sparsity of the symbol's records.

        Sparsity is defined as `1 - (number_of_records / maximum_possible_records)`,
        where the maximum possible records is the product of the number of records
        in each of the symbol's domain sets. A sparsity of 1.0 means the symbol has
        no records (completely empty), while 0.0 means the symbol is fully dense.

        Returns
        -------
        float
            The sparsity of the symbol (between 0.0 and 1.0). Returns `float("nan")` if
            the symbol is a scalar, has a relaxed domain (e.g., `["*"]`), or if any of
            its domain sets have no records.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["A", "B"])
        >>> j = gp.Set(m, name="j", records=["X", "Y", "Z"])
        >>> p = gp.Parameter(m, name="p", domain=[i, j], records=[("A", "X", 10)])
        >>> p.getSparsity()
        0.8333333333333334

        """
        if self.domain_type in {"relaxed", "none"}:
            return float("nan")

        # if there are any domain symbols that do not have records
        if any(not n.number_records for n in self.domain):
            return float("nan")
        else:
            dense = 1
            for i in [n.number_records for n in self.domain]:
                dense *= i

            return 1 - self.number_records / dense


class RecordSymbol(DomainSymbol):
    """Base class for Parameter, Variable, and Equation."""

    # TODO: Legacy function from GTP. Pay the technical debt.
    @property
    @no_type_check
    def shape(self: Parameter | Variable | Equation) -> tuple:
        if self.domain_type == "regular":
            domain = cast("Sequence[Set | Alias | UniverseAlias]", self.domain)
            return tuple(
                [
                    (
                        0
                        if i._getUELs(0, ignore_unused=True) is None
                        else len(i._getUELs(0, ignore_unused=True))
                    )
                    for i in domain
                ]
            )
        else:
            return tuple(
                [
                    (
                        0
                        if self._getUELs(i, ignore_unused=True) is None
                        else len(self._getUELs(i, ignore_unused=True))
                    )
                    for i in range(self.dimension)
                ]
            )

    @property
    def is_scalar(self) -> bool:
        """
        Returns True if the len(self.domain) = 0

        Returns
        -------
        bool
            True if the len(self.domain) = 0
        """
        return self.dimension == 0

    def findEps(
        self: Parameter | Variable | Equation, column: str | None = None
    ) -> pd.DataFrame | None:
        return self.findSpecialValues(SpecialValues.EPS, column=column)

    def findNA(
        self: Parameter | Variable | Equation, column: str | None = None
    ) -> pd.DataFrame | None:
        return self.findSpecialValues(SpecialValues.NA, column=column)

    def findUndef(
        self: Parameter | Variable | Equation, column: str | None = None
    ) -> pd.DataFrame | None:
        return self.findSpecialValues(SpecialValues.UNDEF, column=column)

    def findPosInf(
        self: Parameter | Variable | Equation, column: str | None = None
    ) -> pd.DataFrame | None:
        return self.findSpecialValues(SpecialValues.POSINF, column=column)

    def findNegInf(
        self: Parameter | Variable | Equation, column: str | None = None
    ) -> pd.DataFrame | None:
        return self.findSpecialValues(SpecialValues.NEGINF, column=column)

    def findSpecialValues(
        self: Parameter | Variable | Equation,
        values: float | list[float],
        column: str | None = None,
    ) -> pd.DataFrame | None:
        from gamspy._symbols import Equation, Parameter, Variable

        if self.records is None:
            return None

        if not isinstance(values, (float, list)):
            raise TypeError("Argument 'values' must be type float or list")

        if isinstance(values, float):
            values = [values]

        if column is None:
            if isinstance(self, Parameter):
                column = "value"
            elif isinstance(self, (Variable, Equation)):
                column = "level"

        if not isinstance(column, str):
            raise TypeError(
                f"Argument 'column' must be type str. User passed {type(column)}."
            )

        if column not in self._attributes:
            raise TypeError(
                f"Argument 'column' must be a one of the following: {self._attributes}"
            )

        for n, i in enumerate(values):
            if n == 0:
                if SpecialValues.isEps(i):
                    idx = SpecialValues.isEps(self.records[column])
                elif SpecialValues.isNA(i):
                    idx = SpecialValues.isNA(self.records[column])
                elif SpecialValues.isUndef(i):
                    idx = SpecialValues.isUndef(self.records[column])
                elif SpecialValues.isPosInf(i):
                    idx = SpecialValues.isPosInf(self.records[column])
                elif SpecialValues.isNegInf(i):
                    idx = SpecialValues.isNegInf(self.records[column])
                else:
                    raise ValidationError("Unknown special value detected")
            else:
                if SpecialValues.isEps(i):
                    idx = (idx) | (SpecialValues.isEps(self.records[column]))
                elif SpecialValues.isNA(i):
                    idx = (idx) | (SpecialValues.isNA(self.records[column]))
                elif SpecialValues.isUndef(i):
                    idx = (idx) | (SpecialValues.isUndef(self.records[column]))
                elif SpecialValues.isPosInf(i):
                    idx = (idx) | (SpecialValues.isPosInf(self.records[column]))
                elif SpecialValues.isNegInf(i):
                    idx = (idx) | (SpecialValues.isNegInf(self.records[column]))
                else:
                    raise ValidationError("Unknown special value detected")

        return self.records.loc[idx, :]

    def countNA(
        self: Parameter | Variable | Equation, columns: str | list[str] | None = None
    ) -> int:
        return self._countSpecialValues(SpecialValues.NA, columns=columns)

    def countEps(
        self: Parameter | Variable | Equation, columns: str | list[str] | None = None
    ) -> int:
        return self._countSpecialValues(SpecialValues.EPS, columns=columns)

    def countUndef(
        self: Parameter | Variable | Equation, columns: str | list[str] | None = None
    ) -> int:
        return self._countSpecialValues(SpecialValues.UNDEF, columns=columns)

    def countPosInf(
        self: Parameter | Variable | Equation, columns: str | list[str] | None = None
    ) -> int:
        return self._countSpecialValues(SpecialValues.POSINF, columns=columns)

    def countNegInf(
        self: Parameter | Variable | Equation, columns: str | list[str] | None = None
    ) -> int:
        return self._countSpecialValues(SpecialValues.NEGINF, columns=columns)

    def _countSpecialValues(
        self: Parameter | Variable | Equation, special_value, columns
    ):
        from gamspy._symbols import Equation, Parameter, Variable

        if columns is None:
            if isinstance(self, Parameter):
                columns = "value"
            elif isinstance(self, (Variable, Equation)):
                columns = "level"

        # checks
        if not isinstance(columns, (str, list)):
            raise TypeError(
                f"Argument 'columns' must be type str or list. User passed {type(columns)}."
            )

        if isinstance(columns, str):
            columns = [columns]

        if any(not isinstance(i, str) for i in columns):
            raise TypeError("Argument 'columns' must contain only type str.")

        if any(i not in self._attributes for i in columns):
            raise TypeError(
                f"Argument 'columns' must be a subset of the following: {self._attributes}"
            )

        if self.records is not None:
            if SpecialValues.isEps(special_value):
                return np.sum(SpecialValues.isEps(self.records[columns]))
            elif SpecialValues.isNA(special_value):
                return np.sum(SpecialValues.isNA(self.records[columns]))
            elif SpecialValues.isUndef(special_value):
                return np.sum(SpecialValues.isUndef(self.records[columns]))
            elif SpecialValues.isPosInf(special_value):
                return np.sum(SpecialValues.isPosInf(self.records[columns]))
            elif SpecialValues.isNegInf(special_value):
                return np.sum(SpecialValues.isNegInf(self.records[columns]))
            else:
                raise ValidationError("Unknown special value detected")

    def whereMax(
        self: Parameter | Variable | Equation, column: str | None = None
    ) -> list[str]:
        return self._whereMetric("max", column=column)

    def whereMaxAbs(
        self: Parameter | Variable | Equation, column: str | None = None
    ) -> list[str]:
        return self._whereMetric("absmax", column=column)

    def whereMin(
        self: Parameter | Variable | Equation, column: str | None = None
    ) -> list[str]:
        return self._whereMetric("min", column=column)

    def _whereMetric(self: Parameter | Variable | Equation, metric, column):
        if column is None:
            if isinstance(self, gp.Parameter):
                column = "value"
            elif isinstance(self, (gp.Variable, gp.Equation)):
                column = "level"

        if not isinstance(column, str):
            raise TypeError(
                f"Argument 'column' must be type str. User passed {type(column)}."
            )

        if column not in self._attributes:
            raise TypeError(
                f"Argument 'column' must be a one of the following: {self._attributes}"
            )

        if self.records is not None:
            if metric == "max" and self.dimension > 0:
                try:
                    row_idx = self.records[column].argmax()
                    return self.records.iloc[row_idx, : self.dimension].tolist()
                except Exception:
                    return None

            if metric == "min" and self.dimension > 0:
                try:
                    row_idx = self.records[column].argmin()
                    return self.records.iloc[row_idx, : self.dimension].tolist()
                except Exception:
                    return None

            if metric == "absmax" and self.dimension > 0:
                try:
                    dom = list(
                        self.records[
                            self.records[column] == self.getMaxAbsValue(column)
                        ].to_numpy()[0][: self.dimension]
                    )
                    return dom
                except Exception:
                    return None

    def getMaxValue(
        self: Parameter | Variable | Equation, columns: str | list[str] | None = None
    ) -> float:
        return self._getMetric(metric="max", columns=columns)

    def getMinValue(
        self: Parameter | Variable | Equation, columns: str | list[str] | None = None
    ) -> float:
        return self._getMetric(metric="min", columns=columns)

    def getMeanValue(
        self: Parameter | Variable | Equation, columns: str | list[str] | None = None
    ) -> float:
        return self._getMetric(metric="mean", columns=columns)

    def getMaxAbsValue(
        self: Parameter | Variable | Equation, columns: str | list[str] | None = None
    ) -> float:
        return self._getMetric(metric="absmax", columns=columns)

    def _getMetric(self: Parameter | Variable | Equation, metric, columns):
        from gamspy._symbols import Equation, Parameter, Variable

        if columns is None:
            if isinstance(self, Parameter):
                columns = "value"
            elif isinstance(self, (Variable, Equation)):
                columns = "level"

        if not isinstance(columns, (str, list)):
            raise TypeError(
                f"Argument 'columns' must be type str or list. User passed {type(columns)}."
            )

        if isinstance(columns, str):
            columns = [columns]

        if any(not isinstance(i, str) for i in columns):
            raise TypeError("Argument 'columns' must contain only type str.")

        if any(i not in self._attributes for i in columns):
            raise TypeError(
                f"Argument 'columns' must be a subset of the following: {self._attributes}"
            )

        if self.records is not None:
            if metric == "max":
                return self.records[columns].max().max()
            elif metric == "min":
                return self.records[columns].min().min()
            elif metric == "mean":
                if not (
                    self.records[columns].min().min() == float("-inf")
                    and self.records[columns].max().max() == float("inf")
                ):
                    return self.records[columns].mean().mean()
                else:
                    return float("nan")
            elif metric == "absmax":
                return self.records[columns].abs().max().max()


class VarEquSymbol(RecordSymbol):
    """Base class for Variable and Equation."""

    @property
    def _attributes(self):
        return ["level", "marginal", "lower", "upper", "scale"]

    @property
    def summary(self: Variable | Equation):
        return {
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "domain": self.domain_names,
            "domain_type": self.domain_type,
            "dimension": self.dimension,
            "number_records": self.number_records,
        }

    def dropDefaults(self: Variable | Equation) -> None:
        """
        Drops records from the symbol that are equal to their default values.

        This method removes records where all attributes (level, marginal, lower,
        upper, scale) match the default records for the symbol type.
        """
        if self.records is None:
            return None

        mask = np.all(
            self.records[self._attributes]
            == np.array(list(self._default_records.values())),
            axis=1,
        ) & ~np.any(
            (self.records[self._attributes] == 0.0)
            & SpecialValues.isEps(self.records[self._attributes]),
            axis=1,
        )
        self.records = self.records[~mask].reset_index(drop=True)

    def dropNA(self: Variable | Equation) -> None:
        """
        Drops records from the symbol that contain NA (Not Available) values.

        This method removes any record where at least one of its attributes
        (level, marginal, lower, upper, scale) is a SpecialValues.NA value.
        """
        if self.records is None:
            return None

        mask = SpecialValues.isNA(self.records[self._attributes]).any(axis=1)
        self.records = self.records[~mask].reset_index(drop=True)

    def dropUndef(self: Variable | Equation) -> None:
        """
        Drops records from the symbol that contain UNDEF (Undefined) values.

        This method removes any record where at least one of its attributes
        (level, marginal, lower, upper, scale) is a SpecialValues.UNDEF value.
        """
        if self.records is None:
            return None

        mask = SpecialValues.isUndef(self.records[self._attributes]).any(axis=1)
        self.records = self.records[~mask].reset_index(drop=True)

    def dropEps(self: Variable | Equation) -> None:
        """
        Drops records from the symbol that contain EPS (Epsilon) values.

        This method removes any record where at least one of its attributes
        (level, marginal, lower, upper, scale) is a SpecialValues.EPS value.
        """
        if self.records is None:
            return None

        mask = SpecialValues.isEps(self.records[self._attributes]).any(axis=1)
        self.records = self.records[~mask].reset_index(drop=True)

    def dropMissing(self: Variable | Equation) -> None:
        """
        Drops records from the symbol that contain missing (NaN) values.

        This method removes any record where at least one of its attributes
        (level, marginal, lower, upper, scale) is missing (pandas NaN).
        """
        if self.records is None:
            return None

        mask = pd.isna(self.records[self._attributes]).any(axis=1)
        self.records = self.records[~mask].reset_index(drop=True)

    def toValue(self: Variable | Equation, column: str | None = None) -> float:
        """
        Returns the numerical value of a specified attribute for a scalar symbol.

        Parameters
        ----------
        column : str | None, optional
            The attribute to extract (e.g., "level", "marginal", "lower", "upper", "scale").
            If None, defaults to "level".

        Returns
        -------
        float
            The floating-point value of the requested attribute.

        Raises
        ------
        TypeError
            If the symbol is not a scalar (dimension > 0) or if an invalid column name is provided.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> v = gp.Variable(m, name="v")
        >>> v.l[...] = 15.5
        >>> v.toValue(column="level")
        np.float64(15.5)

        """
        from gamspy._symbols.utils import toValueVariableEquation

        return toValueVariableEquation(self, column=column)

    def toList(self: Variable | Equation, columns: str | None = None) -> list:
        """
        Converts the specified attributes of the symbol records to a Python list.

        Parameters
        ----------
        columns : str | list[str] | None, optional
            The attribute column(s) to include (e.g., "level", "marginal", "lower", "upper", "scale").
            If None, defaults to "level".

        Returns
        -------
        list
            A list containing the requested attribute values. For scalar symbols, it returns a list
            of tuples containing the attributes. For multi-dimensional symbols, it returns a list
            of tuples where domain indices are followed by the requested attributes. Returns an empty
            list if there are no records.

        Examples
        --------
        >>> import gamspy as gp
        >>> import numpy as np
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["A", "B"])
        >>> v = gp.Variable(m, name="v", domain=[i])
        >>> v.setRecords(np.array([1.0, 2.0]))
        >>> v.toList(columns="level")
        [('A', 1.0), ('B', 2.0)]

        """
        from gamspy._symbols.utils import toListVariableEquation

        return toListVariableEquation(self, columns=columns)

    def toDict(
        self: Variable | Equation,
        columns: str | list[str] | None = None,
        orient: str | None = None,
    ) -> dict:
        """
        Converts the records of a non-scalar symbol to a Python dictionary.

        Parameters
        ----------
        columns : str | list[str] | None, optional
            The attribute column(s) to extract (e.g., "level", "marginal"). If None, defaults to "level".
        orient : str | None, optional
            The format of the dictionary. Options are:
            - "natural" (default): Maps domain elements to values. If multiple columns are requested,
              the value becomes a dictionary mapping attributes to their values.
            - "columns": Returns a dictionary of columns, mimicking a pandas DataFrame structure.

        Returns
        -------
        dict
            A dictionary containing the requested symbol attributes. Returns an empty dict if there are no records.

        Raises
        ------
        TypeError
            If the symbol is a scalar, or if an invalid column name is provided.

        Examples
        --------
        >>> import gamspy as gp
        >>> import numpy as np
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["A", "B"])
        >>> v = gp.Variable(m, name="v", domain=[i])
        >>> v.setRecords(np.array([10.0, 20.0]))
        >>> v.toDict(columns="level")
        {'A': 10.0, 'B': 20.0}

        """
        from gamspy._symbols.utils import toDictVariableEquation

        return toDictVariableEquation(self, columns=columns, orient=orient)

    def _setRecords(
        self: Variable | Equation, records, uels_on_axes: bool = False
    ) -> None:
        VarEquIngestor(self).ingest(records, uels_on_axes=uels_on_axes)
        self._handle_domain_violations()

    def pivot(
        self: Variable | Equation,
        index: str | list | None = None,
        columns: str | list | None = None,
        value: str | None = None,
        fill_value: int | float | str | None = None,
    ) -> pd.DataFrame:
        """
        Pivots the specified attribute of the symbol records into a two-dimensional pandas DataFrame.
        This isolates a specific attribute column (e.g., "level" or "marginal") and pivots the data across the
        specified index and column domains.

        Parameters
        ----------
        index : str | list | None, optional
            Column(s) to use for the new frame's index. If None, defaults to all domain labels except the last dimension.
        columns : str | list | None, optional
            Column(s) to use for the new frame's columns. If None, defaults to the last dimension of the domain labels.
        value : str | None, optional
            The specific symbol attribute to pivot (e.g., "level", "marginal", "lower", "upper", "scale"). If None, defaults to "level".
        fill_value : int | float | str | None, optional
            Value used to fill missing data created by the pivot operation. Defaults to 0.0.

        Returns
        -------
        pd.DataFrame
            The pivoted DataFrame representing the specified attribute.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["A", "B"])
        >>> j = gp.Set(m, name="j", records=["X", "Y"])
        >>> v = gp.Variable(m, name="v", domain=[i, j])
        >>> # Assuming variable has been solved/populated
        >>> df_level = v.pivot(value="level")  # doctest: +SKIP

        """
        return pivot_variable(self, index, columns, value, fill_value)

    def generateRecords(
        self: Variable | Equation,
        density: int | float | list | None = None,
        func: dict[str, Callable] | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Automatically generates records for the symbol based on a specified density and optional
        attribute-specific generation functions.

        By default, the "level" attribute is populated with uniformly distributed floats between
        0.0 and 1.0, while all other attributes (e.g., marginal, lower, upper) are initialized to
        the symbol's default record limits.

        Parameters
        ----------
        density : int | float | list | None, optional
            The target density for the generated records on the interval [0, 1].
            * A single numeric value applies to the overall cartesian product.
            * A list applies specific densities to each domain independently.
            * Defaults to 1.0.
        func : dict[str, Callable] | None, optional
            A dictionary mapping attribute strings (e.g., "level", "marginal") to custom callables.
            If provided, each callable is invoked as `func(seed=seed, size=(num_records,))`.
            Attributes not specified in the dictionary fallback to the symbol's defaults.
        seed : int | None, optional
            A random seed for reproducibility during domain sampling and value generation.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["A", "B"])
        >>> v = gp.Variable(m, name="v", type="positive", domain=[i])
        >>> v.generateRecords(seed=42)
        >>> e = gp.Equation(m, name="e", domain=i)
        >>> e.generateRecords(seed=42)

        """
        generate_records_variable(self, density, func, seed)

    def equals(
        self: Variable | Equation,
        other: Variable | Equation,
        columns: str | list[str] | None = None,
        check_meta_data: bool = True,
        rtol: int | float | None = None,
        atol: int | float | None = None,
    ) -> bool:
        """
        Compares this symbol with another symbol to evaluate structural and numerical equality across specified attributes.

        This method verifies dimensions, domain types, and data structure. It then performs an outer merge to match domains
        and checks the specified attribute columns for strict special value equivalence (EPS, NA, UNDEF) and numeric
        closeness (using relative and absolute tolerances).

        Parameters
        ----------
        other : Variable | Equation
            The other Variable or Equation object to compare against.
        columns : str | list[str] | None, optional
            The specific attribute column(s) to evaluate (e.g., ["level", "marginal"]). If None, defaults to all symbol attributes (`_attributes`).
        check_meta_data : bool, optional
            If True, verifies that the symbol names and descriptions match exactly. Defaults to True.
        rtol : int | float | None, optional
            Relative tolerance used for numeric evaluation via `np.isclose`. Defaults to 0.0.
        atol : int | float | None, optional
            Absolute tolerance used for numeric evaluation via `np.isclose`. Defaults to 0.0.

        Returns
        -------
        bool
            True if the symbols are structurally identical and the evaluated attributes are equivalent within the specified tolerances; False otherwise.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["A", "B"])
        >>> v1 = gp.Variable(m, name="v1", type="positive", domain=[i])
        >>> v2 = gp.Variable(m, name="v2", type="positive", domain=[i])
        >>> v1.equals(v2, check_meta_data=False)
        True

        """
        return equals_variable(self, other, columns, check_meta_data, rtol, atol)

    def toSparseCoo(self, column: str = "level") -> coo_matrix | None:
        """
        Converts a specified attribute column of the symbol's records to a SciPy sparse
        COOrdinate format (coo_matrix).

        This method is only available for symbols with 2 or fewer dimensions.
        For scalar symbols (0D), it returns a 1x1 matrix. For 1D symbols,
        it returns a 1xN matrix. For 2D symbols, it returns an MxN matrix.

        Parameters
        ----------
        column : str, optional
            The attribute column to convert (e.g., "level", "marginal", "lower",
            "upper", "scale"). Defaults to "level".

        Returns
        -------
        coo_matrix | None
            A SciPy sparse COO matrix containing the specified attribute values.
            Returns None if there are no records.

        Raises
        ------
        TypeError
            If the `column` argument is not a string, or if it is not a valid attribute
            for the symbol.
        ValidationError
            If the symbol has a dimension greater than 2.

        Examples
        --------
        >>> import gamspy as gp
        >>> import numpy as np
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["A", "B"])
        >>> j = gp.Set(m, name="j", records=["X", "Y"])
        >>> v = gp.Variable(m, name="v", domain=[i, j])
        >>> v.setRecords(np.array([[1.5, 0], [0, 2.5]]))
        >>> sparse_mat = v.toSparseCoo(column="level")  # doctest +SKIP

        """
        from scipy.sparse import coo_matrix

        if not isinstance(column, str):
            raise TypeError("Argument 'column' must be type str")

        if column not in self._attributes:
            raise TypeError(
                f"Argument 'column' must be one of the following: {self._attributes}"
            )

        if self.records is None:
            return None

        if self.is_scalar:
            row = [0]
            col = [0]
            m = 1
            n = 1

        elif self.dimension == 1:
            if self._domain_status is DomainStatus.regular:
                col = (
                    self.records.iloc[:, 0]
                    .map(self.domain[0]._getUELCodes(0, ignore_unused=True))
                    .to_numpy(dtype=int)
                )
            else:
                col = self.records.iloc[:, 0].cat.codes.to_numpy(dtype=int)

            row = np.zeros(len(col), dtype=int)
            m, *n = self.shape
            assert n == []
            n = m
            m = 1

        elif self.dimension == 2:
            if self._domain_status is DomainStatus.regular:
                row = (
                    self.records.iloc[:, 0]
                    .map(self.domain[0]._getUELCodes(0, ignore_unused=True))
                    .to_numpy(dtype=int)
                )
                col = (
                    self.records.iloc[:, 1]
                    .map(self.domain[1]._getUELCodes(0, ignore_unused=True))
                    .to_numpy(dtype=int)
                )
            else:
                row = self.records.iloc[:, 0].cat.codes.to_numpy(dtype=int)
                col = self.records.iloc[:, 1].cat.codes.to_numpy(dtype=int)

            m, n = self.shape

        else:
            raise ValidationError(
                "Sparse coo_matrix formats are only "
                "available for data that has dimension <= 2"
            )

        return coo_matrix(
            (
                self.records.loc[:, column].to_numpy(dtype=float),
                (row, col),
            ),
            shape=(m, n),
            dtype=float,
        )

    def toDense(self: Variable | Equation, column: str = "level") -> np.ndarray:
        """
        Convert column to a dense numpy.array format

        Parameters
        ----------
        column : str, optional
            The column to convert, by default "level"

        Returns
        -------
        np.ndarray
            A column to a dense numpy.array format
        """
        if not isinstance(column, str):
            raise TypeError("Argument 'column' must be type str")

        if column not in self._attributes:
            raise TypeError(
                f"Argument 'column' must be one of the following: {self._attributes}"
            )

        if self.records is None:
            return np.full(self.shape, self._default_records[column])

        if self.is_scalar:
            return self.records.loc[:, column].to_numpy(dtype=float)[0]

        if self.domain_type == "regular":
            # check order of domain UELs in categorical and order of domain UELs in data
            for symobj in cast("list[Set | Alias]", self.domain):
                if symobj.records is None:
                    raise ValidationError(
                        f"The domain element `{symobj.name}` of `{self.name}` has no records. The domain symbols need to have records to get the dense representation."
                    )

                data_cats = symobj.records.iloc[:, 0].unique().tolist()
                cats = symobj.records.iloc[:, 0].cat.categories.tolist()

                if data_cats != cats[: len(data_cats)]:
                    raise ValidationError(
                        f"`toDense` requires that UEL data order of domain set `{symobj.name}` must be "
                        "equal be equal to UEL category order (i.e., the order that set elements "
                        "appear in rows of the dataframe and the order set elements are specified by the categorical). "
                    )
        else:
            # check order of domain UELs in categorical and order of domain UELs in data
            for n in range(self.dimension):
                # check if any invalid codes
                if any(code == -1 for code in self.records.iloc[:, n].cat.codes):
                    raise ValidationError(
                        f"Invalid category detected in dimension `{n}` (code == -1), "
                        "cannot create array until all categories are properly resolved"
                    )

                data_cats = self.records.iloc[:, n].unique().tolist()
                cats = self.records.iloc[:, n].cat.categories.tolist()

                if data_cats != cats[: len(data_cats)]:
                    raise ValidationError(
                        "`toDense` requires (for 'relaxed' symbols) that UEL data order must be "
                        "equal be equal to UEL category order (i.e., the order that set elements "
                        "appear in rows of the dataframe and the order set elements are specified by the categorical). "
                    )

        # create indexing scheme
        if self.domain_type == "regular":
            idx = [
                self.records.iloc[:, n]
                .map(domainobj._getUELCodes(0, ignore_unused=True))
                .to_numpy(dtype=int)
                for n, domainobj in enumerate(cast("list[Set | Alias]", self.domain))
            ]

        else:
            idx = [
                self.records.iloc[:, n].cat.codes.to_numpy(dtype=int)
                for n, domainobj in enumerate(self.domain)
            ]

        # fill the dense array
        a = np.zeros(self.shape)
        val = self.records.loc[:, column].to_numpy(dtype=float)
        a[tuple(idx)] = val

        return a
