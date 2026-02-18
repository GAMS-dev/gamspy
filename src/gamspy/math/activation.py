from __future__ import annotations

from typing import TYPE_CHECKING

import gamspy._symbols.implicits as implicits
import gamspy.math
import gamspy.utils as utils
from gamspy._algebra.number import Number
from gamspy._container import Container
from gamspy._symbols.equation import Equation
from gamspy._symbols.variable import Variable
from gamspy.exceptions import ValidationError
from gamspy.formulations.result import FormulationResult
from gamspy.math.matrix import next_alias

if TYPE_CHECKING:
    from gamspy._algebra.expression import Expression
    from gamspy._algebra.operation import Operation
    from gamspy._symbols.parameter import Parameter


def _get_random_name(prefix: str) -> str:
    return f"{prefix}_{utils._get_unique_name()}"


def _get_lb(x: Variable, default_lb: float):
    lb = x.container.addParameter(
        _get_random_name("lb"),
        domain=x.domain,
    )
    lb[...] = x.lo[...]
    lb[...].where[lb[...] == "-inf"] = default_lb
    return lb


def _get_ub(x: Variable, default_ub: float):
    ub = x.container.addParameter(
        _get_random_name("ub"),
        domain=x.domain,
    )
    ub[...] = x.up[...]
    ub[...].where[ub[...] == "inf"] = default_ub
    return ub


def tanh(x: Variable) -> tuple[Variable, list[Equation]]:
    """
    Convenience wrapper that uses gamspy.math.tanh. Unlike gamspy.math.tanh,
    this function creates a new variable and the equation that
    sets it to follow formulations structure.

    Parameters
    ----------
    x : Variable

    Returns
    -------
    tuple[Variable, list[Equation]]

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> v1 = gp.Variable(m)
    >>> v2, eqs = gp.math.activation.tanh(v1)

    """
    y = x.container.addVariable(domain=x.domain)
    set_y = x.container.addEquation(domain=x.domain)
    set_y[...] = y == gamspy.math.tanh(x)
    y.lo[...] = gamspy.math.tanh(x.lo[...])
    y.up[...] = gamspy.math.tanh(x.up[...])
    return y, [set_y]


def relu_with_sos1_var(
    x: (
        Parameter
        | Variable
        | implicits.ImplicitParameter
        | implicits.ImplicitVariable
        | Expression
        | Operation
    ),
    return_slack_var: bool = False,
):
    """
    Implements the ReLU activation function using
    `SOS1 <https://www.gams.com/47/docs/UG_LanguageFeatures.html?search=sos#UG_LanguageFeatures_SpecialOrderSetsOfType1-SOS1>`_ variables.
    The ReLU function is defined as ReLU(x) = max(x, 0). This implementation
    **generates** one SOS1 type variable which is necessary to represent the
    mathematical relationship and one equation. The SOS1 variable contains the
    activation variable and the slack variable.

    Unlike ``relu_with_binary_var``, this function does not require lower and
    upper bounds for the formulation. It is claimed that when providing tight
    bounds is not straightforward, using ``relu_with_sos1_var`` might perform
    better than ``relu_with_binary_var``, as the relaxation of
    ``relu_with_binary_var`` can be weak.

    Usage of SOS1 variables require MIP or MINLP and a solver that supports SOS1
    variables. Main intended use case of this function is embedding the trained
    neural network into MIP models, we do not suggest using it in training since
    you would need a MINLP solver that support SOS1 variables.

    Returns the activation variable and the equation list if ``return_slack_var`` is
    False, otherwise returns activation, slack variable and the equation list in
    order. Since activation variable and slack variable are the same variable
    only separated by the last domain this function returns ImplicitVariable
    instead of Variable.

    Based on paper:
    `PySCIPOpt-ML: Embedding trained machine learning models into mixed-integer programs. <https://arxiv.org/pdf/2312.08074>`_

    Parameters
    ----------
    x : Parameter | Variable | implicits.ImplicitParameter | implicits.ImplicitVariable | Expression | Operation
    return_slack_var: bool

    Returns
    -------
    tuple[implicits.ImplicitVariable, list[Equation]] | tuple[implicits.ImplicitVariable, implicits.ImplicitVariable, list[Equation]]

    Examples
    --------
    >>> from gamspy import Container, Variable, Set
    >>> from gamspy.math.activation import relu_with_sos1_var
    >>> m = Container()
    >>> i = Set(m, "i", records=range(3))
    >>> x = Variable(m, "x", domain=[i])
    >>> y, eqs = relu_with_sos1_var(x)
    >>> y, s, eqs = relu_with_sos1_var(x, return_slack_var=True)
    >>> y.domain # implicit activation variable has the same domain
    [Set(name='i', domain=['*'])]
    >>> s.domain # implicit slack variable has the same domain as well
    [Set(name='i', domain=['*'])]
    >>> y.name == s.name # In the background that y and s are parts of the same variable
    True
    >>> id(y.parent) == id(s.parent)
    True

    """
    domain = x.domain

    assert isinstance(x.container, Container)
    last_dim = gamspy.math._generate_dims(x.container, [2])[0]

    y = x.container.addVariable(
        _get_random_name("sos1"), type="sos1", domain=[*domain, last_dim]
    )
    eq = x.container.addEquation(
        _get_random_name("eq"),
        domain=domain,
    )

    activation_domain = [*domain, "0"]
    sos1_var_domain = [*domain, "1"]

    eq[...] = y[activation_domain] == x + y[sos1_var_domain]

    if return_slack_var:
        return y[activation_domain], y[sos1_var_domain], [eq]

    return y[activation_domain], [eq]


def leaky_relu_with_binary_var(
    x: (
        Parameter
        | Variable
        | implicits.ImplicitParameter
        | implicits.ImplicitVariable
        | Expression
        | Operation
    ),
    negative_slope: float,
    default_lb: float = -(10**6),
    default_ub: float = 10**6,
    *,
    return_binary_var: bool = False,
):
    """
    Implements the LeakyReLU activation function using binary variables. The LeakyReLU
    function is defined as LeakyReLU(x, negative_slope) = max(x, 0) + negative_slope * min(0, x).
    This implementation **generates** one binary variable, one output variable and four
    equations. The binary variable is necessary to represent the mathematical relationship,
    while the output variable serves as the activation variable. Both the binary and
    ouput variables share the same domain as the input.

    The formulation of this function requires having lower and upper bounds
    for the input ``x``. This function utilizes the bounds from the variables
    if provided. If not, it defaults to the bounds defined by ``default_lb``
    and ``default_ub``. Providing tighter and **correct** bounds can enhance
    the quality of linear relaxations.

    Returns the activation variable and the equation list if ``return_binary_var`` is False,
    otherwise returns activation, binary variable and equation list in order.

    Parameters
    ----------
    x : Parameter | Variable | implicits.ImplicitParameter | implicits.ImplicitVariable | Expression | Operation
    negative_slope: float
    default_ub : float
    default_lb : float
    return_binary_var: bool

    Returns
    -------
    tuple[Variable, list[Equation]] | tuple[Variable, Variable, list[Equation]]

    Examples
    --------
    >>> from gamspy import Container, Variable, Set
    >>> from gamspy.math.activation import leaky_relu_with_binary_var
    >>> m = Container()
    >>> i = Set(m, "i", records=range(3))
    >>> x = Variable(m, "x", domain=[i])
    >>> y, eqs = leaky_relu_with_binary_var(x, 0.01)
    >>> len(eqs)
    4
    >>> y, b, eqs = leaky_relu_with_binary_var(x, 0.01, return_binary_var=True)
    >>> b.type
    'binary'
    >>> y.domain # i many activation variables
    [Set(name='i', domain=['*'])]
    >>> b.domain # i many binary variables
    [Set(name='i', domain=['*'])]

    """
    assert isinstance(x.container, Container)

    if negative_slope <= 0 or negative_slope >= 1:
        raise ValidationError("negative_slope must be in the range (0, 1).")

    domain = x.domain
    sigma = Variable._constructor_bypass(
        x.container,
        _get_random_name("bin"),
        type="binary",
        domain=domain,
    )

    y = Variable._constructor_bypass(
        x.container,
        _get_random_name("y"),
        domain=domain,
        type="free",
    )

    eq = [
        Equation._constructor_bypass(
            x.container,
            _get_random_name("eq"),
            domain=domain,
        )
        for _ in range(4)
    ]

    eq[0][...] = y >= x
    eq[1][...] = y >= negative_slope * x

    if isinstance(x, Variable):
        eq[2][...] = y <= x - (1 - sigma) * (1 - negative_slope) * _get_lb(
            x, default_lb
        )
        eq[3][...] = y <= negative_slope * x + sigma * (1 - negative_slope) * _get_ub(
            x, default_ub
        )
        y.lo[...] = x.lo[...] * negative_slope
        y.up[...] = gamspy.math.Max(0, x.up[...])
    else:
        eq[2][...] = y <= x - (1 - sigma) * (1 - negative_slope) * default_lb
        eq[3][...] = y <= negative_slope * x + sigma * default_ub * (1 - negative_slope)

    if return_binary_var:
        return y, sigma, eq
    else:
        return y, eq


def relu_with_binary_var(
    x: (
        Parameter
        | Variable
        | implicits.ImplicitParameter
        | implicits.ImplicitVariable
        | Expression
        | Operation
    ),
    default_lb: float = -(10**6),
    default_ub: float = 10**6,
    return_binary_var: bool = False,
) -> FormulationResult:
    """
    Implements the ReLU activation function using binary variables. The ReLU
    function is defined as ReLU(x) = max(x, 0). This implementation **generates**
    one binary variable, one positive variable and three equations. The binary
    variable is necessary to represent the mathematical relationship, while the
    positive variable serves as the activation variable. Both the binary and
    positive variables share the same domain as the input.

    The formulation of this function requires having lower and upper bounds
    for the input ``x``. This function utilizes the bounds from the variables
    if provided. If not, it defaults to the bounds defined by ``default_lb``
    and ``default_ub``. Providing tighter and **correct** bounds can enhance
    the quality of linear relaxations.

    Returns the activation variable and the equation list if ``return_binary_var`` is False,
    otherwise returns activation, binary variable and equation list in order.

    Adapted from `OMLT <https://github.com/cog-imperial/OMLT/blob/e60563859a66ac5dd3348bf1763de57eec95171e/src/omlt/neuralnet/activations/relu.py>`_

    FormulationResult:
        - variables_created: ["output", "binary"]
        - equations_created: ["y_gte_x", "y_lte_x_1", "y_lte_x_2"]

    Parameters
    ----------
    x : Parameter | Variable | implicits.ImplicitParameter | implicits.ImplicitVariable | Expression | Operation
    default_ub : float
    default_lb : float
    return_binary_var: bool

    Returns
    -------
    FormulationResult

    Examples
    --------
    >>> from gamspy import Container, Variable, Set
    >>> from gamspy.math.activation import relu_with_binary_var
    >>> m = Container()
    >>> i = Set(m, "i", records=range(3))
    >>> x = Variable(m, "x", domain=[i])
    >>> y, eqs = relu_with_binary_var(x) # FormulationResult can be unpacked for backwards compat
    >>> y.type
    'positive'
    >>> len(eqs)
    3
    >>> y, b, eqs = relu_with_binary_var(x, return_binary_var=True)
    >>> b.type
    'binary'
    >>> y.domain # i many activation variables
    [Set(name='i', domain=['*'])]
    >>> b.domain # i many binary variables
    [Set(name='i', domain=['*'])]
    >>> output = relu_with_binary_var(x) # You can use FormulationResult too
    >>> binary_var = output.variables_created["binary"]
    >>> y = output.result

    """
    assert isinstance(x.container, Container)
    domain = x.domain
    sigma = x.container.addVariable(
        _get_random_name("bin"),
        type="binary",
        domain=domain,
    )

    y = x.container.addVariable(
        _get_random_name("y"),
        type="positive",
        domain=domain,
    )

    eq = [
        x.container.addEquation(
            _get_random_name("eq"),
            domain=domain,
        )
        for _ in range(3)
    ]

    # y >= 0 implied by positive variable
    eq[0][...] = y >= x

    if isinstance(x, Variable):
        eq[1][...] = y <= x - (1 - sigma) * _get_lb(x, default_lb)
        eq[2][...] = y <= sigma * _get_ub(x, default_ub)
        y.lo[...] = gamspy.math.Max(0, x.lo[...])
        y.up[...] = gamspy.math.Max(0, x.up[...])
    else:
        eq[1][...] = y <= x - (1 - sigma) * default_lb
        eq[2][...] = y <= sigma * default_ub
        y.lo[...] = 0

    result = FormulationResult(
        y,
        {
            "y_gte_x": eq[0],
            "y_lte_x_1": eq[1],
            "y_lte_x_2": eq[2],
        },
    )
    result.variables_created = {"binary": sigma, "output": y}
    result.extra_return = sigma if return_binary_var else None
    return result


def relu_with_equilibrium(
    x: (
        Parameter
        | Variable
        | implicits.ImplicitParameter
        | implicits.ImplicitVariable
        | Expression
        | Operation
    ),
) -> FormulationResult:
    """
    Implements the ReLU activation function using Equilibrium Constraints.
    This implementation is suitable for models of type Mathematical Program with
    Equilibrium Constraints (MPEC) or Mixed Complementarity Problem (MCP).
    One positive variable is **generated**, which serves as the activation
    variable and no equations. The activation variable shares the same domain as
    the input. Lower and upper bounds are not required for this formulation.

    Returns FormulationResult which can be unpacked as activation variable,
    matches dictionary and the equation list (empty).

    FormulationResult:
        variables_created: ["output"]
        equations_created: []
        matches: {(output - x) : output}
        extra_return: yes (matches)

    or if the provided input was not a Variable, this formulation assigns it to a
    new variable and uses the new variable instead

    FormulationResult:
        variables_created: ["output", "new_input"]
        equations_created: ["set_new_input"]
        matches: {(output - new_input) : output}
        extra_return: yes (matches)


    Parameters
    ----------
    x : Variable

    Returns
    -------
    FormulationResult

    Examples
    --------
    >>> from gamspy import Container, Variable, Set
    >>> from gamspy.math.activation import relu_with_equilibrium
    >>> m = Container()
    >>> i = Set(m, "i", records=range(3))
    >>> x = Variable(m, "x", domain=[i])
    >>> y, matches, eqs = relu_with_equilibrium(x)
    >>> y.type
    'positive'
    >>> len(eqs)
    0
    >>> len(matches)
    1
    >>> result = relu_with_equilibrium(x)
    >>> type(result)
    <class 'gamspy.formulations.result.FormulationResult'>
    >>> result = relu_with_equilibrium(x - 5)
    >>> new_input = result.variables_created["new_input"]

    """
    assert isinstance(x.container, Container)
    domain = x.domain

    y = x.container.addVariable(
        type="positive",
        domain=domain,
    )

    new_input = None
    set_new_input = None
    if not isinstance(x, Variable):
        new_input = Variable._constructor_bypass(
            x.container,
            _get_random_name("new_input"),
            domain=domain,
        )
        set_new_input = Equation._constructor_bypass(
            x.container,
            _get_random_name("set_new_input"),
            domain=domain,
        )
        set_new_input[...] = new_input == x
    else:
        new_input = x

    eq = Equation._constructor_bypass(
        x.container,
        _get_random_name("matches_eq"),
        domain=domain,
    )

    eq[...] = y - new_input >= Number(0)

    result = FormulationResult(y, {})
    result.variables_created["output"] = y
    if set_new_input is not None:
        result.variables_created["new_input"] = new_input
        result.equations_created["set_new_input"] = set_new_input

    result.extra_return = {eq: y}
    result.matches = {eq: y}
    return result


def relu_with_complementarity_var(
    x: (
        Parameter
        | Variable
        | implicits.ImplicitParameter
        | implicits.ImplicitVariable
        | Expression
        | Operation
    ),
):
    """
    Implements the ReLU activation function using complementarity conditions.
    The ReLU function is defined as ReLU(x) = max(x, 0). This implementation
    **generates** one positive variable, which serves as the activation variable
    and two equations. The activation variable shares the same domain as the
    input. Unlike ``relu_with_binary_var``, this function does not require lower
    and upper bounds for the formulation.

    Returns the activation variable and the equation list.

    Adapted from `OMLT <https://github.com/cog-imperial/OMLT/blob/e60563859a66ac5dd3348bf1763de57eec95171e/src/omlt/neuralnet/activations/relu.py>`_

    Parameters
    ----------
    x : Parameter | Variable | implicits.ImplicitParameter | implicits.ImplicitVariable | Expression | Operation

    Returns
    -------
    tuple[Variable, list[Equation]]

    Examples
    --------
    >>> from gamspy import Container, Variable, Set
    >>> from gamspy.math.activation import relu_with_complementarity_var
    >>> m = Container()
    >>> i = Set(m, "i", records=range(3))
    >>> x = Variable(m, "x", domain=[i])
    >>> y, eqs = relu_with_complementarity_var(x)
    >>> y.type
    'positive'
    >>> len(eqs)
    2

    """
    assert isinstance(x.container, Container)
    domain = x.domain

    y = x.container.addVariable(
        _get_random_name("y"),
        type="positive",
        domain=domain,
    )

    eq = [
        x.container.addEquation(
            _get_random_name("eq"),
            domain=domain,
        )
        for _ in range(2)
    ]

    # y >= 0 implied by positive variable
    eq[0][...] = y * (y - x) == 0
    eq[1][...] = y - x >= 0

    if isinstance(x, Variable):
        y.lo[...] = gamspy.math.Max(0, x.lo[...])
        y.up[...] = gamspy.math.Max(0, x.up[...])
    else:
        y.lo[...] = 0

    return y, eq


def log_softmax(x: Variable, dim: int = -1, skip_intrinsic: bool = False):
    """
    Implements the log_softmax activation function. This function strictly
    requires a GAMSPy Variable, `y = log_softmax(x)`. The ``dim`` parameter
    specifies the **index** of the softmax dimension. If not provided, it
    calculates log_softmax for the last dimension. This function is preferred
    over the :meth:`softmax <gamspy.math.softmax>` function because, when
    the softmax dimension has 20 or fewer elements, it uses the
    :meth:`lse_max <gamspy.math.lse_max>` (log-sum-exp) intrinsic function for
    improved numerical stability which usually leads to faster solve times.
    Some solvers do not support :meth:`lse_max <gamspy.math.lse_max>`, in that
    case you can set ``skip_intrinsic`` parameter to True to not use intrinsic
    functions even when possible.

    To learn more about `Log-Sum-Exp trick <https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/>`_ .

    This function is usually combined with Negative Log Likelihood loss for
    classification problems.

    Returns the activation variable and the equation list.

    Parameters
    ----------
    x : Variable
    dim : int
    skip_intrinsic: bool

    Returns
    -------
    Variable, list[Equation]

    Examples
    --------
    >>> from gamspy import Container, Variable
    >>> from gamspy.math import dim
    >>> from gamspy.math.activation import log_softmax
    >>> m = Container()
    >>> x = Variable(m, "x", domain=dim([500, 10]))
    >>> y, eqs1 = log_softmax(x) # uses LSE because 10 <= 20
    >>> y.domain
    [Set(name='DenseDim500_1', domain=['*']), Set(name='DenseDim10_1', domain=['*'])]
    >>> y2, eqs2 = log_softmax(x, dim=0) # cannot use LSE because 500 > 20
    >>> y3, eqs3 = log_softmax(x, skip_intrinsic=True) # don't use LSE because of skip_intrinsic

    """
    if not isinstance(x, Variable):
        raise ValidationError("log_softmax expects a variable")

    if dim < 0:
        dim = len(x.domain) + dim

    sum_domain = next_alias(x.domain[dim])

    y = x.container.addVariable(
        _get_random_name("y"),
        domain=x.domain,
    )

    eq = x.container.addEquation(
        _get_random_name("eq"),
        domain=x.domain,
    )

    if not skip_intrinsic and len(sum_domain) != 0 and len(sum_domain) <= 20:
        # Use built-in LSE if possible
        scalars = list(sum_domain.records["uni"])
        variables = []
        for scalar in scalars:
            expr_domain = [*x.domain[:dim], scalar, *x.domain[dim + 1 :]]
            variables.append(x[expr_domain])

        log_sum_exp = gamspy.math.lse_max(*variables)
        eq[...] = y[...] == x - log_sum_exp
    else:
        expr_domain = [d if i != dim else sum_domain for (i, d) in enumerate(x.domain)]
        sum_expr = gamspy.Sum(sum_domain, gamspy.math.exp(x[expr_domain]))
        eq[...] = y[...] == x - gamspy.math.log(sum_expr)

    return y, [eq]


def softplus(
    x: Variable,
    beta=1.0,
    *,
    skip_intrinsic: bool = False,
) -> tuple[Variable, list[Equation]]:
    """
    Implements the softplus activation function. This function uses the
    :meth:`lse_max_sc <gamspy.math.lse_max_sc>` (log-sum-exp) intrinsic function for
    improved numerical stability which usually leads to faster solve times. Some solvers
    do not support :meth:`lse_max_sc <gamspy.math.lse_max_sc>`, in that case you can set
    ``skip_intrinsic`` parameter to True to not use intrinsic functions.

    ``beta`` value controls the smoothness and slope of the function, by default equals
    to 1.

    ``skip_intrinsic`` (Default `False`)

    Parameters
    ----------
    x : Variable
    beta : float

    skip_intrinsic: bool

    Returns
    -------
    tuple[Variable, list[Equation]]

    Examples
    --------
    >>> from gamspy import Container, Variable
    >>> from gamspy.math import dim
    >>> from gamspy.math.activation import softplus
    >>> m = Container()
    >>> x = Variable(m, "x", domain=dim([500, 10]))
    >>> y, eqs1 = softplus(x)
    >>> y.domain
    [Set(name='DenseDim500_1', domain=['*']), Set(name='DenseDim10_1', domain=['*'])]
    >>> y2, eqs2 = softplus(x, skip_intrinsic=True) # don't use LSE because of skip_intrinsic

    """
    y = Variable._constructor_bypass(
        x.container,
        _get_random_name("y"),
        domain=x.domain,
    )
    eq = Equation._constructor_bypass(
        x.container,
        _get_random_name("eq"),
        domain=x.domain,
    )

    if skip_intrinsic:
        eq[...] = y == (1 / beta) * gamspy.math.log(1 + gamspy.math.exp(beta * x))
    else:
        eq[...] = y == gamspy.math.lse_max_sc(beta, 0, x)

    return y, [eq]


def softmax(x: Variable, dim: int = -1):
    """
    Implements the softmax activation function. This function strictly requires
    a GAMSPy Variable, `y = softmax(x)`. The ``dim`` parameter specifies the
    index of the softmax dimension. If not provided, the softmax is calculated
    for the last dimension. This function is implemented for completeness;
    however, in many cases, you can use :meth:`log_softmax <gamspy.math.log_softmax>`
    for better numerical stability.

    Use :meth:`log_softmax <gamspy.math.log_softmax>` if you need to take the
    logarithm of the softmax function.

    Returns the activation variable and the equation list.

    Parameters
    ----------
    x : Variable
    dim : int

    Returns
    -------
    tuple[Variable, list[Equation]]

    Examples
    --------
    >>> from gamspy import Container, Variable
    >>> from gamspy.math import dim
    >>> from gamspy.math.activation import softmax
    >>> m = Container()
    >>> x = Variable(m, "x", domain=dim([500, 10]))
    >>> y, eqs = softmax(x)
    >>> y.domain
    [Set(name='DenseDim500_1', domain=['*']), Set(name='DenseDim10_1', domain=['*'])]

    """
    if not isinstance(x, Variable):
        raise ValidationError("softmax expects a variable")

    if dim < 0:
        dim = len(x.domain) + dim

    sum_domain = next_alias(x.domain[dim])
    expr_domain = [d if i != dim else sum_domain for (i, d) in enumerate(x.domain)]

    y = x.container.addVariable(
        _get_random_name("y"),
        domain=x.domain,
    )

    eq = x.container.addEquation(
        _get_random_name("eq"),
        domain=x.domain,
    )

    sum_expr = gamspy.Sum(sum_domain, gamspy.math.exp(x[expr_domain]))
    eq[...] = y[...] == gamspy.math.exp(x) / sum_expr

    return y, [eq]
