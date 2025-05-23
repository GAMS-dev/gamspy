from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import gamspy._algebra.condition as condition
import gamspy._algebra.domain as domain
import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
import gamspy._symbols as syms
import gamspy._symbols.implicits as implicits
import gamspy._validation as validation
import gamspy.utils as utils
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    import pandas as pd

    from gamspy._algebra import Domain
    from gamspy._algebra.condition import Condition
    from gamspy._algebra.expression import Expression
    from gamspy._symbols import Alias, Parameter, Set
    from gamspy._symbols.implicits import (
        ImplicitParameter,
        ImplicitSet,
        ImplicitVariable,
    )
    from gamspy.math.misc import MathOp


class Operation(operable.Operable):
    def __init__(
        self,
        domain: Set
        | Alias
        | ImplicitSet
        | Sequence[Set | Alias]
        | Domain
        | Condition,
        rhs: (
            Expression
            | Operation
            | MathOp
            | ImplicitSet
            | ImplicitVariable
            | ImplicitParameter
            | int
            | bool
        ),
        op_name: str,
    ):
        self.op_domain = utils._to_list(domain)  # type: ignore
        assert len(self.op_domain) > 0, "Operation requires at least one index"
        self.rhs = rhs
        self._op_name = op_name
        self.raw_domain = self._get_raw_domain()

        # allow conditions
        self.where = condition.Condition(self)

        self._bare_op_domain = utils._get_set(self.op_domain)
        self.container = self.raw_domain[0].container
        self.domain: list[Set | Alias] = []

        self._operation_indices = []
        if isinstance(rhs, condition.Condition):
            rhs = rhs.conditioning_on  # type: ignore

        if not isinstance(rhs, (bool, float, int)):
            for i, x in enumerate(rhs.domain):  # type: ignore
                try:
                    sum_index = self._bare_op_domain.index(x)
                    self._operation_indices.append((i, sum_index))
                except ValueError:
                    self.domain.append(x)

        self.dimension: int = validation.get_dimension(self.domain)
        controlled_domain = [d for d in self._bare_op_domain]
        controlled_domain.extend(getattr(rhs, "controlled_domain", []))
        self.controlled_domain = list(set(controlled_domain))

    def __getitem__(self, indices: Sequence | str):
        domain = validation.validate_domain(self, indices)
        for index, sum_index in self._operation_indices:
            domain.insert(index, self._bare_op_domain[sum_index])

        if isinstance(self.rhs, (bool, float, int)):
            return Operation(self.op_domain, self.rhs, self._op_name)

        return Operation(self.op_domain, self.rhs[domain], self._op_name)  # type: ignore

    @property
    def records(self) -> pd.DataFrame | None:
        """
        Evaluates the operation and returns the resulting records.

        Returns
        -------
        pd.DataFrame | None
        """
        assert self.container is not None
        temp_name = "a" + utils._get_unique_name()
        temp_param = syms.Parameter._constructor_bypass(
            self.container, temp_name, self.domain
        )
        temp_param[...] = self
        del self.container.data[temp_name]
        return temp_param.records

    def toValue(self) -> float | None:
        """
        Convenience method to return the records of the operation as a Python float. Only possible if there is a single record as a result of the operation.

        Returns
        -------
        float | None
        """
        records = self.records
        if records is not None:
            return records["value"][0]

        return records

    def toList(self) -> list | None:
        """
        Convenience method to return the records of the operation as a list.

        Returns
        -------
        list | None
        """
        records = self.records
        if records is not None:
            return records.values.tolist()

        return None

    def _get_raw_domain(self) -> list[Set | Alias | ImplicitSet]:
        raw_domain = []
        for elem in self.op_domain:
            if isinstance(elem, condition.Condition):
                if isinstance(elem.conditioning_on, implicits.ImplicitSet):
                    raw_domain.append(elem.conditioning_on.parent)
                elif isinstance(elem.conditioning_on, (syms.Set, syms.Alias)):
                    raw_domain.append(elem.conditioning_on)
                elif isinstance(elem.conditioning_on, domain.Domain):
                    raw_domain += elem.conditioning_on.sets
            elif isinstance(elem, domain.Domain):
                raw_domain += elem.sets
            elif isinstance(elem, implicits.ImplicitSet):
                raw_domain.append(elem)
            else:
                raw_domain.append(elem)

        return raw_domain

    def _validate_operation(
        self, control_stack: list[Set | Alias | ImplicitSet]
    ) -> None:
        for elem in self.raw_domain:
            if isinstance(elem, implicits.ImplicitSet):
                control_stack += [
                    member
                    for member in elem.domain
                    if member not in control_stack
                ] + [elem.parent]

            if elem in control_stack:
                raise ValidationError(f"Set {elem} is already in control")

        stack = control_stack + self.raw_domain
        if isinstance(self.rhs, expression.Expression):
            self.rhs._validate_definition(utils._unpack(stack))
        elif isinstance(self.rhs, Operation):
            self.rhs._validate_operation(utils._unpack(stack))

    def _get_index_str(self) -> str:
        if len(self.op_domain) == 1:
            op_domain = self.op_domain[0]
            representation = op_domain.gamsRepr()
            if isinstance(op_domain, condition.Condition):
                # sum((l(root,s,s1,s2) $ od(root,s)),1); -> not valid
                # sum(l(root,s,s1,s2) $ od(root,s),1); -> valid
                return representation[1:-1]

            return representation

        return (
            "("
            + ",".join([index.gamsRepr() for index in self.op_domain])
            + ")"
        )

    def __eq__(self, other):
        return expression.Expression(self, "=e=", other)

    def __ne__(self, other):
        return expression.Expression(self, "ne", other)

    def _replace_operations(self, output: str) -> str:
        output = output.replace("=l=", "<=")
        output = output.replace("=g=", ">=")
        output = output.replace("=e=", "eq")

        return output

    def gamsRepr(self) -> str:
        # Ex: sum((i,j), c(i,j) * x(i,j))
        output = f"{self._op_name}("

        index_str = self._get_index_str()

        output += index_str
        output += ","

        if isinstance(self.rhs, float):
            self.rhs = utils._map_special_values(self.rhs)

        if isinstance(self.rhs, bool):
            self.rhs = (
                "yes" if self.rhs is True else "no"  # type: ignore
            )

        expression_str = (
            str(self.rhs)
            if isinstance(self.rhs, (bool, float, int, str))
            else self.rhs.gamsRepr()
        )

        output += expression_str
        output += ")"

        output = self._replace_operations(output)

        return output

    def latexRepr(self) -> str:
        """
        Representation of this operation in Latex.

        Returns
        -------
        str
        """
        op_map = {
            "sum": "sum",
            "prod": "prod",
            "smax": "max",
            "smin": "min",
            "sand": "sand",
            "sor": "sor",
        }

        indices = []
        given_condition = None
        for index in self.op_domain:
            if isinstance(index, condition.Condition):
                indices.append(index.conditioning_on)
                given_condition = index.condition
            else:
                indices.append(index)

        index_str = ",".join([index.latexRepr() for index in indices])

        if given_condition is not None:
            condition_str = str(given_condition)
            if hasattr(given_condition, "latexRepr"):
                condition_str = given_condition.latexRepr()
            index_str += " ~ | ~ " + condition_str

        expression_str = (
            str(self.rhs)
            if isinstance(self.rhs, (int, float, str))
            else self.rhs.latexRepr()
        )
        representation = (
            f"\\{op_map[self._op_name]}_\\text{{{index_str}}} {expression_str}"
        )
        return representation


class Sum(Operation):
    """
    Represents a sum operation over a domain.

    Parameters
    ----------
    domain : Set | Alias | ImplicitSet | Sequence[Set | Alias], Domain, Condition
    expression : (
            Expression
            | MathOp
            | ImplicitVariable
            | ImplicitParameter
            | int
            | bool
            | Variable
            | Parameter
            | Operation
        )

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=['i1','i2', 'i3'])
    >>> v = gp.Variable(m, "v")
    >>> e = gp.Equation(m, "e", type="eq")
    >>> d = gp.Parameter(m, "d", domain=[i], records=[("i1", 1), ("i2", 2), ("i3", 4)])
    >>> e[...] = gp.Sum(i, d[i]) <= v

    """

    def __init__(
        self,
        domain: Set
        | Alias
        | ImplicitSet
        | Sequence[Set | Alias]
        | Domain
        | Condition,
        expression: Operation
        | Expression
        | MathOp
        | ImplicitSet
        | ImplicitParameter
        | ImplicitVariable
        | int
        | bool,
    ):
        super().__init__(domain, expression, "sum")

    def __repr__(self) -> str:
        return f"Sum(domain={self.domain}, expression={self.rhs})"

    def gamsRepr(self):
        """
        Representation of the Sum operation in GAMS language.

        Returns
        -------
        str

        Examples
        --------
        >>> from gamspy import Container, Set, Parameter, Variable, Sum
        >>> m = Container()
        >>> i = Set(m, "i", records=['i1','i2', 'i3'])
        >>> c = Parameter(m, "c", domain=i)
        >>> v = Variable(m, "v", domain=i)
        >>> Sum(i, c[i]*v[i]).gamsRepr()
        'sum(i,(c(i) * v(i)))'

        """
        repr = super().gamsRepr()
        return repr


class Product(Operation):
    """
    Represents a product operation over a domain.

    Parameters
    ----------
    domain : Set | Alias | Sequence[Set | Alias], Domain, Expression
    expression : (
            Expression
            | MathOp
            | ImplicitVariable
            | ImplicitParameter
            | int
            | bool
            | Variable
            | Parameter
            | Operation
        )

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=['i1','i2', 'i3'])
    >>> v = gp.Variable(m, "v")
    >>> e = gp.Equation(m, "e", type="eq")
    >>> p = gp.Parameter(m, "p", domain=[i], records=[("i1", 1), ("i2", 2), ("i3", 4)])
    >>> e[...] = gp.Product(i, p[i]) <= v

    """

    def __init__(
        self,
        domain: Set
        | Alias
        | ImplicitSet
        | Sequence[Set | Alias]
        | Domain
        | Condition,
        expression: Operation
        | Expression
        | MathOp
        | ImplicitSet
        | ImplicitParameter
        | ImplicitVariable
        | int
        | bool,
    ):
        super().__init__(domain, expression, "prod")

    def __repr__(self) -> str:
        return f"Product(domain={self.domain}, expression={self.rhs})"

    def gamsRepr(self):
        """
        Representation of the Product operation in GAMS language.

        Returns
        -------
        str

        Examples
        --------
        >>> from gamspy import Container, Set, Parameter, Variable, Product
        >>> m = Container()
        >>> i = Set(m, "i", records=['i1','i2', 'i3'])
        >>> c = Parameter(m, "c", domain=i)
        >>> v = Variable(m, "v", domain=i)
        >>> Product(i, c[i]*v[i]).gamsRepr()
        'prod(i,(c(i) * v(i)))'

        """
        repr = super().gamsRepr()
        return repr


class Smin(Operation):
    """
    Represents a smin operation over a domain.

    Parameters
    ----------
    domain : Set | Alias | Sequence[Set | Alias], Domain, Expression
    expression : (
            Expression
            | MathOp
            | ImplicitVariable
            | ImplicitParameter
            | int
            | bool
            | Variable
            | Parameter
            | Operation
        )

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=['i1','i2', 'i3'])
    >>> v = gp.Variable(m, "v")
    >>> e = gp.Equation(m, "e", type="eq")
    >>> p = gp.Parameter(m, "p", domain=[i], records=[("i1", 1), ("i2", 2), ("i3", 4)])
    >>> e[...] = gp.Smin(i, p[i]) <= v

    """

    def __init__(
        self,
        domain: Set
        | Alias
        | ImplicitSet
        | Sequence[Set | Alias]
        | Domain
        | Condition,
        expression: Operation
        | Expression
        | MathOp
        | ImplicitSet
        | ImplicitParameter
        | ImplicitVariable
        | int
        | bool,
    ):
        super().__init__(domain, expression, "smin")

    def __repr__(self) -> str:
        return f"Smin(domain={self.domain}, expression={self.rhs})"

    def gamsRepr(self):
        """
        Representation of the Smin operation in GAMS language.

        Returns
        -------
        str

        Examples
        --------
        >>> from gamspy import Container, Set, Parameter, Variable, Smin
        >>> m = Container()
        >>> i = Set(m, "i", records=['i1','i2', 'i3'])
        >>> c = Parameter(m, "c", domain=i)
        >>> v = Variable(m, "v", domain=i)
        >>> Smin(i, c[i]*v[i]).gamsRepr()
        'smin(i,(c(i) * v(i)))'

        """
        repr = super().gamsRepr()
        return repr


class Smax(Operation):
    """
    Represents a smax operation over a domain.

    Parameters
    ----------
    domain : Set | Alias | Sequence[Set | Alias], Domain, Expression
    expression : (
            Expression
            | MathOp
            | ImplicitVariable
            | ImplicitParameter
            | int
            | bool
            | Variable
            | Parameter
            | Operation
        )

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=['i1','i2', 'i3'])
    >>> v = gp.Variable(m, "v")
    >>> e = gp.Equation(m, "e", type="eq")
    >>> p = gp.Parameter(m, "p", domain=[i], records=[("i1", 1), ("i2", 2), ("i3", 4)])
    >>> e[...] = gp.Smax(i, p[i]) <= v

    """

    def __init__(
        self,
        domain: Set
        | Alias
        | ImplicitSet
        | Sequence[Set | Alias]
        | Domain
        | Condition,
        expression: Operation
        | Expression
        | MathOp
        | ImplicitSet
        | ImplicitParameter
        | ImplicitVariable
        | int
        | bool,
    ):
        super().__init__(domain, expression, "smax")

    def __repr__(self) -> str:
        return f"Smax(domain={self.domain}, expression={self.rhs})"

    def gamsRepr(self):
        """
        Representation of the Smax operation in GAMS language.

        Returns
        -------
        str

        Examples
        --------
        >>> from gamspy import Container, Set, Parameter, Variable, Smax
        >>> m = Container()
        >>> i = Set(m, "i", records=['i1','i2', 'i3'])
        >>> c = Parameter(m, "c", domain=i)
        >>> v = Variable(m, "v", domain=i)
        >>> Smax(i, c[i]*v[i]).gamsRepr()
        'smax(i,(c(i) * v(i)))'

        """
        repr = super().gamsRepr()
        return repr


class Sand(Operation):
    """
    Represents a sand operation over a domain.

    Parameters
    ----------
    domain : Set | Alias | Sequence[Set | Alias], Domain, Expression
    expression : (
            Expression
            | MathOp
            | ImplicitVariable
            | ImplicitParameter
            | int
            | bool
            | Variable
            | Parameter
            | Operation
        )

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=['i1','i2', 'i3'])
    >>> p = gp.Parameter(m, domain=i)
    >>> p[i] = gp.math.uniformInt(0,1)
    >>> result = gp.Parameter(m)
    >>> result[:] = gp.Sand(i, p[i])

    """

    def __init__(
        self,
        domain: Set
        | Alias
        | ImplicitSet
        | Sequence[Set | Alias]
        | Domain
        | Condition,
        expression: Operation
        | Expression
        | MathOp
        | ImplicitSet
        | ImplicitParameter
        | ImplicitVariable
        | int
        | bool,
    ):
        super().__init__(domain, expression, "sand")

    def __repr__(self) -> str:
        return f"Sand(domain={self.domain}, expression={self.rhs})"

    def gamsRepr(self):
        """
        Representation of the Sand operation in GAMS language.

        Returns
        -------
        str

        Examples
        --------
        >>> from gamspy import Container, Set, Parameter, Variable, Sor, Sand
        >>> m = Container()
        >>> i = Set(m, "i", records=['i1','i2', 'i3'])
        >>> v = Variable(m, "v", domain=i, type="binary")
        >>> Sand(i, v[i]).gamsRepr()
        'sand(i,v(i))'

        """
        repr = super().gamsRepr()
        return repr


class Sor(Operation):
    """
    Represents a sor operation over a domain.

    Parameters
    ----------
    domain : Set | Alias | Sequence[Set | Alias], Domain, Expression
    expression : (
            Expression
            | MathOp
            | ImplicitVariable
            | ImplicitParameter
            | int
            | bool
            | Variable
            | Parameter
            | Operation
        )

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=['i1','i2', 'i3'])
    >>> p = gp.Parameter(m, domain=i)
    >>> p[i] = gp.math.uniformInt(0,1)
    >>> result = gp.Parameter(m)
    >>> result[:] = gp.Sor(i, p[i])

    """

    def __init__(
        self,
        domain: Set
        | Alias
        | ImplicitSet
        | Sequence[Set | Alias]
        | Domain
        | Condition,
        expression: Operation
        | Expression
        | MathOp
        | ImplicitSet
        | ImplicitParameter
        | ImplicitVariable
        | int
        | bool,
    ):
        super().__init__(domain, expression, "sor")

    def __repr__(self) -> str:
        return f"Sor(domain={self.domain}, expression={self.rhs})"

    def gamsRepr(self):
        """
        Representation of the Sor operation in GAMS language.

        Returns
        -------
        str

        Examples
        --------
        >>> from gamspy import Container, Set, Parameter, Variable, Sor
        >>> m = Container()
        >>> i = Set(m, "i", records=['i1','i2', 'i3'])
        >>> v = Variable(m, "v", domain=i, type="binary")
        >>> Sor(i, v[i]).gamsRepr()
        'sor(i,v(i))'

        """
        repr = super().gamsRepr()
        return repr


class Ord(operable.Operable):
    """
    The operator ord may be used only with one-dimensional sets.

    Parameters
    ----------
    set : Set | Alias

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> t = gp.Set(m, name="t", description="time periods", records=[str(x) for x in range(1985, 1996)])
    >>> val = gp.Parameter(m, name="val", domain=[t])
    >>> val[t] = gp.Ord(t)

    """

    def __init__(self, symbol: Set | Alias):
        if not isinstance(symbol, (syms.Set, syms.Alias)):
            raise ValidationError(
                "Ord operation is only for Set and Alias objects!"
            )

        self._symbol = symbol
        self.container = symbol.container
        self.domain: list[Set | Alias] = []
        self.where = condition.Condition(self)

    def __eq__(self, other):
        return expression.Expression(self, "eq", other)

    def __ge__(self, other):
        return expression.Expression(self, ">=", other)

    def __le__(self, other):
        return expression.Expression(self, "<=", other)

    def __ne__(self, other):
        return expression.Expression(self, "ne", other)

    def gamsRepr(self) -> str:
        """
        Representation of the Ord operation in GAMS language.

        Returns
        -------
        str

        Examples
        --------
        >>> from gamspy import Container, Set, Ord
        >>> m = Container()
        >>> i = Set(m, "i", records=['i1','i2', 'i3'])
        >>> Ord(i).gamsRepr()
        'ord(i)'

        """
        return f"ord({self._symbol.name})"

    def latexRepr(self) -> str:
        """
        Representation of Ord function in Latex.

        Returns
        -------
        str
        """
        return f"ord({self._symbol.name})"


class Card(operable.Operable):
    """
    The operator card may be used with any symbol and returns its number of records.

    Parameters
    ----------
    symbol : Set | Alias | Parameter | Variable | Equation | Model

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> t = gp.Set(m, name="t", description="time periods", records=[str(x) for x in range(1985, 1996)])
    >>> s = gp.Parameter(m, name="s")
    >>> s[...] = gp.Card(t)

    """

    def __init__(
        self,
        symbol: Set | Alias | Parameter,
    ) -> None:
        if not isinstance(symbol, (syms.Set, syms.Alias, syms.Parameter)):
            raise ValidationError(
                "Card operation is only for Set, Alias and Parameter objects!"
            )

        self._symbol = symbol
        self.container = symbol.container
        self.domain: list[Set | Alias] = []
        self.where = condition.Condition(self)

    def __eq__(self, other) -> Expression:  # type: ignore
        return expression.Expression(self, "eq", other)

    def __ge__(self, other):
        return expression.Expression(self, ">=", other)

    def __le__(self, other):
        return expression.Expression(self, "<=", other)

    def __ne__(self, other):  # type: ignore
        return expression.Expression(self, "ne", other)

    def __bool__(self):
        raise ValidationError(
            "Card operation cannot be used as a truth value. Use len(<symbol>.records) instead."
        )

    def gamsRepr(self) -> str:
        """
        Representation of the Card operation in GAMS language.

        Returns
        -------
        str

        Examples
        --------
        >>> from gamspy import Container, Set, Card
        >>> m = Container()
        >>> i = Set(m, "i", records=['i1','i2', 'i3'])
        >>> Card(i).gamsRepr()
        'card(i)'

        """
        return f"card({self._symbol.name})"

    def latexRepr(self) -> str:
        """
        Representation of Card function in Latex.

        Returns
        -------
        str
        """
        return f"card({self._symbol.name})"
