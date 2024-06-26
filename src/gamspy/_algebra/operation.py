from __future__ import annotations

from typing import TYPE_CHECKING

import gamspy._algebra.condition as condition
import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
import gamspy.utils as utils
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from gams.transfer import Alias, Parameter, Set

    from gamspy._algebra import Domain
    from gamspy._algebra.expression import Expression
    from gamspy._symbols.implicits import ImplicitParameter, ImplicitVariable


class Operation(operable.Operable):
    def __init__(
        self,
        domain: Set | Alias | tuple[Set | Alias] | Domain | Expression,
        expression: (
            Expression | ImplicitVariable | ImplicitParameter | int | bool
        ),
        op_name: str,
    ):
        self.domain = utils._to_list(domain)
        assert len(self.domain) > 0, "Operation requires at least one index"
        self.expression = expression
        self._op_name = op_name

        # allow conditions
        self.where = condition.Condition(self)

    def _get_index_str(self) -> str:
        if len(self.domain) == 1:
            domain = self.domain[0]
            representaiton = domain.gamsRepr()
            if isinstance(domain, expression.Expression):
                # sum((l(root,s,s1,s2) $ od(root,s)),1); -> not valid
                # sum(l(root,s,s1,s2) $ od(root,s),1); -> valid
                return representaiton[1:-1]

            return representaiton

        return (
            "(" + ",".join([index.gamsRepr() for index in self.domain]) + ")"
        )

    def __eq__(self, other):  # type: ignore
        return expression.Expression(self, "=e=", other)

    def __ne__(self, other):  # type: ignore
        return expression.Expression(self, "ne", other)

    def __neg__(self):
        return expression.Expression(None, "-", self)

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

        if isinstance(self.expression, float):
            self.expression = utils._map_special_values(self.expression)

        if isinstance(self.expression, bool):
            self.expression = (
                "yes" if self.expression is True else "no"  # type: ignore
            )

        expression_str = (
            str(self.expression)
            if isinstance(self.expression, (int, str))
            else self.expression.gamsRepr()
        )

        output += expression_str
        output += ")"

        output = self._replace_operations(output)

        return output


class Sum(Operation):
    """
    Represents a sum operation over a domain.

    Parameters
    ----------
    domain : Set | Alias | Tuple[Set | Alias], Domain, Expression
    expression : Expression | int | bool

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=['i1','i2', 'i3'])
    >>> v = gp.Variable(m, "v")
    >>> e = gp.Equation(m, "e", type="eq")
    >>> e[...] = gp.Sum(i, 3) <= v
    """

    def __init__(
        self,
        domain: Set | Alias | tuple[Set | Alias] | Domain | Expression,
        expression: Expression | int | bool,
    ):
        super().__init__(domain, expression, "sum")

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
    domain : Set | Alias | Tuple[Set | Alias], Domain, Expression
    expression : Expression | int | bool

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=['i1','i2', 'i3'])
    >>> v = gp.Variable(m, "v")
    >>> e = gp.Equation(m, "e", type="eq")
    >>> e[...] = gp.Product(i, 3) <= v
    """

    def __init__(
        self,
        domain: Set | Alias | tuple[Set | Alias] | Domain | Expression,
        expression: Expression | int | bool,
    ):
        super().__init__(domain, expression, "prod")

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
    domain : Set | Alias | Tuple[Set | Alias], Domain, Expression
    expression : Expression | int | bool

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=['i1','i2', 'i3'])
    >>> v = gp.Variable(m, "v")
    >>> e = gp.Equation(m, "e", type="eq")
    >>> e[...] = gp.Smin(i, 3) <= v
    """

    def __init__(
        self,
        domain: Set | Alias | tuple[Set | Alias] | Domain | Expression,
        expression: Expression | int | bool,
    ):
        super().__init__(domain, expression, "smin")

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
    domain : Set | Alias | Tuple[Set | Alias], Domain, Expression
    expression : Expression | int | bool

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=['i1','i2', 'i3'])
    >>> v = gp.Variable(m, "v")
    >>> e = gp.Equation(m, "e", type="eq")
    >>> e[...] = gp.Smax(i, 3) <= v
    """

    def __init__(
        self,
        domain: Set | Alias | tuple[Set | Alias] | Domain | Expression,
        expression: Expression | int | bool,
    ):
        super().__init__(domain, expression, "smax")

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


class Ord(operable.Operable):
    """
    The operator ord may be used only with one-dimensional sets.

    Parameters
    ----------
    set : Set | Alias

    Examples
    --------
    >>> import gamspy as gp
    >>>
    >>> m = gp.Container()
    >>> t = gp.Set(
    >>>     m,
    >>>     name="t",
    >>>     description="time periods",
    >>>     records=[str(x) for x in range(1985, 1996)],
    >>> )
    >>> val = gp.Parameter(m, name="val", domain=[t])
    >>> val[t] = gp.Ord(t)
    """

    def __init__(self, set: Set | Alias):
        self._set = set

    def __eq__(self, other) -> Expression:  # type: ignore
        return expression.Expression(self, "eq", other)

    def __ge__(self, other):
        return expression.Expression(self, ">=", other)

    def __le__(self, other):
        return expression.Expression(self, "<=", other)

    def __ne__(self, other):  # type: ignore
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
        return f"ord({self._set.name})"


class Card(operable.Operable):
    """
    The operator card may be used with any symbol and returns its number of records.

    Parameters
    ----------
    symbol : Set | Alias | Parameter | Variable | Equation | Model

    Examples
    --------
    >>> import gamspy as gp
    >>>
    >>> m = gp.Container()
    >>>
    >>> t = gp.Set(
    >>>     m,
    >>>     name="t",
    >>>     description="time periods",
    >>>     records=[str(x) for x in range(1985, 1996)],
    >>> )
    >>> s = gp.Parameter(m, name="s")
    >>> s[...] = gp.Card(t)
    """

    def __init__(
        self,
        symbol: Set | Alias | Parameter,
    ) -> None:
        self._symbol = symbol

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
