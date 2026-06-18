from __future__ import annotations

import itertools
import os
import threading
import weakref
from typing import TYPE_CHECKING, Any, cast, no_type_check

import numpy as np
import pandas as pd
from gams.core.gdx import GMS_DT_PAR

import gamspy as gp
import gamspy._algebra.condition as condition
import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
import gamspy._algebra.operation as operation
import gamspy._symbols.implicits as implicits
import gamspy._validation as validation
import gamspy.utils as utils
from gamspy._records_ingestion import ParameterIngestor
from gamspy._symbols.base import RecordSymbol
from gamspy._symbols.equals import equals_parameter
from gamspy._symbols.generate_records import generate_records_parameter
from gamspy._symbols.pivot import pivot_parameter
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from collections.abc import Callable

    from scipy.sparse import coo_matrix

    from gamspy import Container
    from gamspy._algebra.condition import Condition
    from gamspy._algebra.expression import Expression
    from gamspy._algebra.number import Number
    from gamspy._algebra.operation import Operation
    from gamspy._extrinsic import ExtrinsicFunction
    from gamspy._symbols.implicits import ImplicitParameter
    from gamspy._types import DomainType, IndexType, ParameterRecordsType
    from gamspy.math.misc import MathOp


class Parameter(operable.Operable, RecordSymbol):
    """
    Represents a Parameter symbol in GAMS.

    Parameters are used to hold data (scalars, vectors, or multi-dimensional arrays).
    See https://gamspy.readthedocs.io/en/latest/user/basics/parameter.html

    Parameters
    ----------
    container : Container
        The Container object that this parameter belongs to.
    name : str, optional
        Name of the parameter. If not provided, a unique name is generated automatically.
    domain : DomainType, optional
        The domain of the parameter. Can be a list of Sets/Aliases, a single Set/Alias,
        or strings representing set names. Use "*" for the universe set. Default is [] (scalar).
    records : Sequence | np.ndarray | int | float | pd.DataFrame | pd.Series, optional
        Initial values to populate the parameter. Can be a scalar, a list, a numpy array, or a pandas DataFrame.
    domain_forwarding : bool | list[bool], optional
        If True, adding records to this parameter will implicitly add new elements to the
        domain sets (if they are dynamic). Default is False.
    description : str, optional
        A human-readable description of the parameter.
    uels_on_axes : bool, optional
        If True, implies that the Unique Element Labels (UELs) for the domain are
        contained in the axes (index/columns) of the provided `records` object
        (e.g., pandas DataFrame). Default is False.
    is_miro_input : bool, optional
        If True, flags this parameter as an input symbol for GAMS MIRO. Default is False.
    is_miro_output : bool, optional
        If True, flags this parameter as an output symbol for GAMS MIRO. Default is False.
    is_miro_table : bool, optional
        If True, flags this parameter as a table symbol for GAMS MIRO. Default is False.

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=['i1', 'i2'])
    >>> a = gp.Parameter(m, "a", domain=[i], records=[['i1', 1], ['i2', 2]], description="Input data")

    """

    @classmethod
    def _constructor_bypass(
        cls,
        container: Container,
        name: str,
        domain: DomainType | None = None,
        records: ParameterRecordsType | None = None,
        description: str = "",
    ) -> Parameter:
        obj = object.__new__(cls)

        # legacy gtp attributes
        ## set private properties directly
        obj._container = cast(
            "Container",
            weakref.proxy(container)
            if not isinstance(container, weakref.ProxyType)
            else container,
        )
        obj.name = name
        obj._domain = obj._normalize_domain(obj._container, domain)
        obj._domain_forwarding = False
        obj._description = description
        obj._records = records
        obj._gams_type = GMS_DT_PAR
        obj._gams_subtype = 0
        obj._container._data.update({name: obj})

        # gamspy attributes
        obj._domain_violations = None
        obj.where = condition.Condition(obj)
        obj._latex_name = name.replace("_", r"\_")
        obj.container._add_statement(obj)
        obj._metadata = {}
        obj._should_load_from_gams = False
        obj._should_unload_to_gams = False

        ## miro support
        obj._is_miro_input = False
        obj._is_miro_output = False
        obj._is_miro_table = False

        return obj

    def __new__(
        cls,
        container: Container | None = None,
        name: str | None = None,
        domain: DomainType | None = None,
        records: ParameterRecordsType | None = None,
        domain_forwarding: bool | list[bool] = False,
        description: str = "",
        uels_on_axes: bool = False,
        is_miro_input: bool = False,
        is_miro_output: bool = False,
        is_miro_table: bool = False,
    ):
        if container is not None and not isinstance(container, gp.Container):
            raise TypeError(
                f"Container must of type `Container` but found {type(container)}"
            )

        if name is None:
            return object.__new__(cls)
        else:
            if not isinstance(name, str):
                raise TypeError(f"Name must of type `str` but found {type(name)}")

            try:
                if not container:
                    container = gp._ctx_managers[
                        (os.getpid(), threading.get_native_id())
                    ]

                symbol = container._data[name]
            except KeyError:
                return object.__new__(cls)

        if isinstance(symbol, cls):
            return symbol

        raise TypeError(
            f"Cannot overwrite symbol `{name}` in container"
            " because it is not a Parameter object"
        )

    def __init__(
        self,
        container: Container | None = None,
        name: str | None = None,
        domain: DomainType | None = None,
        records: ParameterRecordsType | None = None,
        domain_forwarding: bool | list[bool] = False,
        description: str = "",
        uels_on_axes: bool = False,
        is_miro_input: bool = False,
        is_miro_output: bool = False,
        is_miro_table: bool = False,
    ):
        self._metadata: dict[str, Any] = {}
        if (is_miro_input or is_miro_output) and name is None:
            raise ValidationError("Please specify a name for miro symbols.")

        # miro support
        self._is_miro_input = is_miro_input
        self._is_miro_output = is_miro_output
        self._is_miro_table = is_miro_table
        self._is_miro_symbol = is_miro_input or is_miro_output or is_miro_table
        self._domain_violations = None

        # does symbol exist
        has_symbol = False
        if isinstance(getattr(self, "container", None), gp.Container):
            has_symbol = True

        if has_symbol:
            domain = self._normalize_domain(self.container, domain)
            if any(d1 != d2 for d1, d2 in itertools.zip_longest(self._domain, domain)):
                raise ValueError(
                    "Cannot overwrite symbol in container unless symbol"
                    " domains are equal"
                )

            if self._domain_forwarding != domain_forwarding:
                raise ValueError(
                    "Cannot overwrite symbol in container unless"
                    " 'domain_forwarding' is left unchanged"
                )

            # reset some properties
            if description != "":
                self._description = description

            self._records: pd.DataFrame | None = None

            previous_state = self._container._options.miro_protect
            self._container._options.miro_protect = False
            if records is not None:
                self.setRecords(records, uels_on_axes=uels_on_axes)
            self._container._options.miro_protect = previous_state
        else:
            if container is None:
                try:
                    container = gp._ctx_managers[
                        (os.getpid(), threading.get_native_id())
                    ]
                except KeyError as e:
                    raise ValidationError("Parameter requires a container.") from e

            self._container = cast("Container", weakref.proxy(container))

            if name is not None:
                name = validation.validate_name(name)

                if is_miro_input or is_miro_output:
                    name = name.lower()
            else:
                name = self._container._get_symbol_name(prefix="p")

            self.name = name
            domain = self._normalize_domain(self.container, domain)
            self._domain = self._validate_domain(domain)
            self._domain_forwarding = domain_forwarding
            self._description = description
            self._records = None
            self._gams_type = GMS_DT_PAR
            self._gams_subtype = 0
            self._latex_name = self.name.replace("_", r"\_")
            self._should_load_from_gams = False
            self._should_unload_to_gams = False
            self._container._data.update({name: self})

            if is_miro_input:
                self._already_loaded = False
                self._container._miro_input_symbols.append(self.name)

            if is_miro_output:
                self._container._miro_output_symbols.append(self.name)

            validation.validate_container(self, self._domain)
            self.where = condition.Condition(self)
            self._assignment: Expression | None = None
            self._container._add_statement(self)

            previous_state = self._container._options.miro_protect
            self._container._options.miro_protect = False
            if records is not None:
                self._setRecords(records, uels_on_axes=uels_on_axes)
                if self.dimension == 0 and not self._is_miro_symbol:
                    self._should_unload_to_gams = False
            else:
                if self._is_miro_symbol:
                    self._should_unload_to_gams = True

            self._container._synch_with_gams()

            self._container._options.miro_protect = previous_state

    def _serialize(self) -> dict:
        info: dict[str, Any] = {
            "_domain_forwarding": self._domain_forwarding,
            "_is_miro_input": self._is_miro_input,
            "_is_miro_output": self._is_miro_output,
            "_is_miro_table": self._is_miro_table,
            "_metadata": self._metadata,
        }
        if self._assignment is not None:
            info["_assignment"] = self._assignment.getDeclaration()

        return info

    def _deserialize(self, info: dict) -> None:
        for key, value in info.items():
            if key == "_assignment":
                left, right = value.split(" = ")
                value = expression.Expression(left, "=", right[:-1])

            setattr(self, key, value)

        # Relink domain symbols
        new_domain = []
        for elem in self._domain:
            if elem == "*":
                new_domain.append(elem)
                continue
            new_domain.append(self._container[elem.name])

        self._domain = new_domain

    def __getitem__(self, indices: IndexType) -> ImplicitParameter:
        domain = validation.validate_domain(self, indices)

        return implicits.ImplicitParameter(self, name=self.name, domain=domain)

    def __setitem__(
        self,
        indices: IndexType,
        rhs: Operation
        | Expression
        | MathOp
        | Condition
        | Parameter
        | ImplicitParameter
        | float
        | int
        | Number
        | ExtrinsicFunction,
    ):
        # self[domain] = rhs
        domain = validation.validate_domain(self, indices)

        if self._is_miro_input and self._container._options.miro_protect:
            raise ValidationError(
                f"Cannot assign to protected miro input symbol {self.name}. `miro_protect`"
                " attribute of the container can be set to False to allow"
                " assigning to MIRO input symbols"
            )

        if isinstance(rhs, float):
            rhs = utils._map_special_values(rhs)

        statement = expression.Expression(
            implicits.ImplicitParameter(self, name=self.name, domain=domain),
            "=",
            rhs,
        )

        # Cannot validate definition if we are in a gp.Loop since the control indices can be provided by the gp.Loop
        if not self._container._in_loop:
            statement._validate_definition(utils._unpack(domain))

        self._container._add_statement(statement)
        self._assignment = statement

        self.container._synch_with_gams()
        self._should_load_from_gams = True

    def __eq__(self, other):
        op = "eq"
        if isinstance(
            other,
            (
                implicits.ImplicitVariable,
                expression.Expression,
                operation.Operation,
            ),
        ):
            op = "=e="
        return expression.Expression(self, op, other)

    def __ne__(self, other):
        return expression.Expression(self, "ne", other)

    def __repr__(self) -> str:
        return f"Parameter(name='{self.name}', domain={self.domain})"

    @property
    def T(self) -> ImplicitParameter:
        """
        Alias for the `.t()` method.

        Returns
        -------
        ImplicitParameter
        """
        return self.t()

    def t(self) -> ImplicitParameter:
        """
        Returns an ImplicitParameter derived from this
        parameter by swapping its last two indices. This operation
        does not generate a new parameter in GAMS.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=['i1','i2'])
        >>> j = gp.Set(m, "j", records=['j1','j2'])
        >>> v = gp.Parameter(m, "v", domain=[i, j])
        >>> v_t = v.t()
        >>> v_t.domain
        [Set(name='j', domain=['*']), Set(name='i', domain=['*'])]
        >>> v_t[i, j] # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        gamspy.exceptions.ValidationError:
        >>> v_t["j1", "i1"].gamsRepr()
        'v("i1","j1")'

        """
        from gamspy.math.matrix import permute

        dims = list(range(len(self.domain)))
        if len(dims) < 2:
            raise ValidationError(
                "Parameter must contain at least 2 dimensions to transpose"
            )

        x = dims[-1]
        dims[-1] = dims[-2]
        dims[-2] = x
        return permute(self, dims)  # type: ignore

    @property
    def _attributes(self):
        return ["value"]

    @property
    def summary(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "domain": self.domain_names,
            "domain_type": self.domain_type,
            "dimension": self.dimension,
            "number_records": self.number_records,
        }

    def toValue(self) -> float:
        """
        Returns the numerical value of a scalar Parameter.

        Returns
        -------
        float | None
            The floating-point value of the scalar parameter.

        Raises
        ------
        TypeError
            If the parameter is not a scalar (dimension > 0).

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> p = gp.Parameter(m, name="p", records=42.5)
        >>> p.toValue()
        np.float64(42.5)

        """
        from gamspy._symbols.utils import toValueParameter

        return toValueParameter(self)

    def toList(self) -> list:
        """
        Converts the records of the Parameter to a Python list.

        Returns
        -------
        list | None
            A list containing the parameter's data. For scalars, it returns a list with a single
            numerical value. For multi-dimensional parameters, it returns a list of tuples where
            the last element of each tuple is the value. Returns an empty list if there are no records.

        Examples
        --------
        >>> import gamspy as gp
        >>> import numpy as np
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["seattle", "san-diego"])
        >>> d = gp.Parameter(m, name="d", domain=[i], records=np.array([10, 25]))
        >>> d.toList()
        [('seattle', 10.0), ('san-diego', 25.0)]

        """
        from gamspy._symbols.utils import toListParameter

        return toListParameter(self)

    def toDict(self, orient: str | None = None) -> dict | None:
        """
        Converts the records of a non-scalar Parameter to a Python dictionary.

        Parameters
        ----------
        orient : str | None, optional
            The format of the dictionary. Options are:
            - "natural" (default): Maps domain elements to values (e.g., `{'A': 10.0}`).
              For multi-dimensional parameters, keys are tuples (e.g., `{(A, X): 10.0}`).
            - "columns": Returns a dictionary of columns (e.g., `{'i': {0: 'A'}, 'value': {0: 10.0}}`).

        Returns
        -------
        dict | None
            A dictionary containing the parameter's data, or None if there are no records.

        Raises
        ------
        TypeError
            If the parameter is a scalar.

        Examples
        --------
        >>> import gamspy as gp
        >>> import numpy as np
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["seattle", "san-diego"])
        >>> p = gp.Parameter(m, name="p", domain=[i], records=np.array([10.0, 20.0]))
        >>> p.toDict()
        {'seattle': 10.0, 'san-diego': 20.0}

        """
        from gamspy._symbols.utils import toDictParameter

        return toDictParameter(self, orient=orient)

    # TODO: Legacy function from GTP. Pay the technical debt.
    @no_type_check
    def toSparseCoo(self) -> coo_matrix | None:
        """
        Converts the parameter records to a SciPy sparse COOrdinate format (coo_matrix).

        This method is only available for parameters with 2 or fewer dimensions.
        For scalar parameters (0D), it returns a 1x1 matrix. For 1D parameters,
        it returns a 1xN matrix. For 2D parameters, it returns an MxN matrix.

        Returns
        -------
        coo_matrix | None
            A SciPy sparse COO matrix containing the parameter values. Returns None
            if there are no records.

        Raises
        ------
        ValueError
            If the parameter has a dimension greater than 2.

        Examples
        --------
        >>> import gamspy as gp
        >>> import numpy as np
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["A", "B"])
        >>> j = gp.Set(m, name="j", records=["X", "Y"])
        >>> p = gp.Parameter(m, name="p", domain=[i, j])
        >>> p.setRecords(np.array([[1, 0], [0, 2]]))
        >>> sparse_mat = p.toSparseCoo()  # doctest +SKIP

        """
        from scipy.sparse import coo_matrix

        if self.records is None:
            return None

        if self.is_scalar:
            row, col, m, n = [0], [0], 1, 1
        elif self.dimension == 1:
            if self.domain_type == "regular":
                col = (
                    self.records.iloc[:, 0]
                    .map(self.domain[0]._getUELCodes(0, ignore_unused=True))
                    .to_numpy(dtype=int)
                )
            else:
                col = self.records.iloc[:, 0].cat.codes.to_numpy(dtype=int)

            row = np.zeros(len(col), dtype=int)
            m, *n_arr = self.shape
            assert not n_arr
            n, m = m, 1
        elif self.dimension == 2:
            if self.domain_type == "regular":
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
            raise ValueError(
                "Sparse coo_matrix formats are only available for data that has dimension <= 2"
            )

        return coo_matrix(
            (self.records.iloc[:, -1].to_numpy(dtype=float), (row, col)),
            shape=(m, n),
            dtype=float,
        )

    # TODO: Legacy function from GTP. Pay the technical debt.
    @no_type_check
    def toDense(self) -> np.ndarray | None:
        """
        Convert symbol records to a dense numpy.array format

        Returns
        -------
        ndarray | None
            A numpy array with symbol records, None if no records were assigned

        Examples
        --------
        >>> import numpy as np
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> j = gp.Set(m, "j", records=["new-york", "chicago", "topeka"])
        >>> s = gp.Parameter(m, "s", [j], records=np.array([3,4,5]))
        >>> print(s.toDense())
        [3. 4. 5.]

        """
        if self.records is None:
            return None

        if self.is_scalar:
            return self.records.to_numpy(dtype=float).reshape(self.shape)

        if self.domain_type == "regular":
            for symobj in self.domain:
                data_cats = symobj.records.iloc[:, 0].unique().tolist()
                cats = symobj.records.iloc[:, 0].cat.categories.tolist()

                if data_cats != cats[: len(data_cats)]:
                    raise ValidationError(
                        f"`toDense` requires that UEL data order of domain set `{symobj.name}` must be "
                        "equal be equal to UEL category order (i.e., the order that set elements "
                        "appear in rows of the dataframe and the order set elements are specified by the categorical). "
                    )
        else:
            for n in range(self.dimension):
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

        if self.domain_type == "regular":
            idx = [
                self.records.iloc[:, n]
                .map(domainobj._getUELCodes(0, ignore_unused=True))
                .to_numpy(dtype=int)
                for n, domainobj in enumerate(self.domain)
            ]
        else:
            idx = [
                self.records.iloc[:, n].cat.codes.to_numpy(dtype=int)
                for n, domainobj in enumerate(self.domain)
            ]

        a = np.zeros(self.shape)
        val = self.records.iloc[:, -1].to_numpy(dtype=float)
        a[tuple(idx)] = val

        return a

    def pivot(
        self,
        index: str | list | None = None,
        columns: str | list | None = None,
        fill_value: int | float | str | None = None,
    ) -> pd.DataFrame:
        """
        Convenience function to pivot records into a new shape (only symbols with >1D can be pivoted).

        Parameters
        ----------
        index : str | list, optional
            If index is None then it is set to dimensions [0..dimension-1], by default None
        columns : str | list, optional
            If columns is None then it is set to the last dimension, by default None
        fill_value : int | float | str, optional
            Missing values in the pivot will take the value provided by fill_value, by default None

        Returns
        -------
        pd.DataFrame
            The pivoted DataFrame representing the parameter records.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["A", "B"])
        >>> j = gp.Set(m, name="j", records=["X", "Y"])
        >>> p = gp.Parameter(m, name="p", domain=[i, j], records=[("A", "X", 10), ("B", "Y", 20)])
        >>> df = p.pivot()

        """
        return pivot_parameter(self, index, columns, fill_value)

    def generateRecords(
        self,
        density: int | float | list | None = None,
        func: Callable | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Generates records with the Cartesian product of all domain sets.

        Parameters
        ----------
        density : int | float | list, optional
            Takes any value on the interval [0,1]. If density is <1 then randomly selected records will be removed.
            `density` will accept a `list` of length `dimension` which allows users to specify a density per symbol dimension,
            by default None.
        func : Callable, optional
            Functions to generate the records, by default None; numpy.random.uniform(0,1)
        seed : int, optional
            Random number state can be set with `seed` argument, by default None

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["A", "B"])
        >>> p = gp.Parameter(m, name="p", domain=[i])
        >>> p.generateRecords(seed=42)

        """
        generate_records_parameter(self, density, func, seed)

    def equals(
        self,
        other: Parameter,
        check_meta_data: bool = True,
        rtol: int | float | None = None,
        atol: int | float | None = None,
    ) -> bool:
        """
        Used to compare the symbol to another symbol

        Parameters
        ----------
        other : Parameter
            Other Symbol to compare with
        check_uels : bool, optional
            If True, check both used and unused UELs and confirm same order, otherwise only check used UELs in data and do not check UEL order. by default True
        check_meta_data : bool, optional
            If True, check that symbol name and description are the same, otherwise skip. by default True
        rtol : int | float, optional
            relative tolerance, by default None
        atol : int | float, optional
            absolute tolerance, by default None

        Returns
        -------
        bool
            True if symbols are equal, False otherwise

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> p1 = gp.Parameter(m, name="p1", records=10.0001)
        >>> p2 = gp.Parameter(m, name="p2", records=10.0002)
        >>> p1.equals(p2, check_meta_data=False, atol=1e-3)
        True

        """
        return equals_parameter(self, other, check_meta_data, rtol, atol)

    @property
    def records(self) -> pd.DataFrame | None:
        """
        Returns the records (data) of the Parameter as a DataFrame.

        Returns
        -------
        pd.DataFrame | None
            The dataframe containing the parameter's data.

        Examples
        --------
        >>> import gamspy as gp
        >>> import numpy as np
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["seattle", "san-diego"])
        >>> d = gp.Parameter(m, name="d", domain=[i])
        >>> d.setRecords(np.array([10, 25]))
        >>> d.toList()
        [('seattle', 10.0), ('san-diego', 25.0)]

        """
        if self._should_load_from_gams:
            self._load_from_gams()

        return self._records

    @records.setter
    def records(self, records: pd.DataFrame | None):
        if (
            hasattr(self, "_is_miro_input")
            and self._is_miro_input
            and self._container._options.miro_protect
        ):
            raise ValidationError(
                "Cannot assign to protected miro input symbols. `miro_protect`"
                " attribute of the container can be set to False to allow"
                " assigning to MIRO input symbols"
            )

        if records is not None and not isinstance(records, pd.DataFrame):
            raise TypeError("Symbol 'records' must be type DataFrame")

        self._records = records
        self._should_unload_to_gams = True
        self._handle_domain_forwarding()

    def __hash__(self):
        return id(self)

    def _setRecords(self, records: Any, *, uels_on_axes: bool = False) -> None:
        ParameterIngestor(self).ingest(records, uels_on_axes=uels_on_axes)
        self._handle_domain_violations()

    def setRecords(
        self, records: ParameterRecordsType | None, uels_on_axes: bool = False
    ) -> None:
        """
        Sets the records (data) of the Parameter.

        This is a convenience method to load data into the parameter. It handles various
        input formats like scalars, lists, numpy arrays, and pandas DataFrames.
        If `uels_on_axes=True`, it assumes that all domain information is contained
        in the axes (index/columns) of the pandas object, and the data will be flattened if necessary.

        Parameters
        ----------
        records : Sequence | np.ndarray | int | float | pd.DataFrame | pd.Series
            The data to load. Common formats:


            - Scalar: `10.5`
            - List of tuples/lists: `[['i1', 1], ['i2', 2]]`
            - numpy array: `np.array([1, 2])`
            - pandas DataFrame
        uels_on_axes : bool, optional
            If True, assumes domain elements are in the axes of the DataFrame. Default is False.

        Examples
        --------
        >>> import gamspy as gp
        >>> import numpy as np
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i")
        >>> i.setRecords(["seattle", "san-diego"])
        >>> i.toList()
        ['seattle', 'san-diego']

        """
        if self._is_miro_input and self._container._options.miro_protect:
            raise ValidationError(
                f"Cannot assign to protected miro input symbol {self.name}. `miro_protect`"
                " attribute of the container can be set to False to allow"
                " assigning to MIRO input symbols"
            )

        if records is None:
            self._container._add_statement(f"option clear={self.name};")
            self._container._synch_with_gams()
            self._records = None
        elif isinstance(records, (int, float)):
            self._container._add_statement(f"{self.name} = {records};")
            self._container._synch_with_gams()
            self._should_load_from_gams = True
        else:
            self._setRecords(records, uels_on_axes=uels_on_axes)
            self._container._synch_with_gams()

    def gamsRepr(self) -> str:
        """
        Returns the string representation of this Parameter in the GAMS language.

        The representation includes the parameter name and its domain (e.g., 'p(i, j)').

        Returns
        -------
        str
            The GAMS string representation.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=['i1','i2'])
        >>> d = gp.Parameter(m, "d", domain=i)
        >>> d.gamsRepr()
        'd(i)'

        """
        representation = self.name
        if self.domain:
            representation += self._get_domain_str(self._domain_forwarding)

        return representation

    def getDeclaration(self) -> str:
        """
        Returns the GAMS declaration statement for this Parameter.

        This string is used internally to declare the parameter in the GAMS execution environment.

        Returns
        -------
        str
            The GAMS declaration statement (e.g., 'Parameter p(i);').

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=['i1','i2'])
        >>> a = gp.Parameter(m, "a", [i], records=[['i1',1],['i2',2]])
        >>> a.getDeclaration()
        'Parameter a(i);'

        """
        output = f"Parameter {self.gamsRepr()}"

        if self.description:
            output += ' "' + self.description + '"'

        if self._records is None:
            output += " / /"

        if self.dimension == 0 and self._records is not None:
            value = self._records["value"][0]
            value = utils._map_special_values(value)
            if isinstance(value, float) and np.isnan(value):
                value = "Undf"

            output += f" / {value} /"

        output += ";"

        return output

    def getAssignment(self) -> str:
        """
        Returns the latest GAMS assignment statement for this Parameter.

        This string represents the last assignment operation performed on the parameter
        (e.g., 'p(i) = 5;').

        Returns
        -------
        str
            The GAMS assignment statement.

        Raises
        ------
        ValidationError
            If the parameter has not been assigned yet.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=['i1','i2'])
        >>> a = gp.Parameter(m, "a", [i], records=[['i1',1],['i2',2]])
        >>> a[i] = a[i] * 5
        >>> a.getAssignment()
        'a(i) = a(i) * 5;'

        """
        if self._assignment is None:
            raise ValidationError("Parameter was not assigned!")

        return self._assignment.getDeclaration()
