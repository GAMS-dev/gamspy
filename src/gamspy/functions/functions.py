from typing import Union, TYPE_CHECKING
import gamspy._algebra.expression as expression

if TYPE_CHECKING:
    from gamspy._algebra.expression import Expression
    from gamspy._symbols.symbol import Symbol


def beta(x: Union[int, float], y: Union[int, float]) -> "Expression":
    """
    Beta function

    Parameters
    ----------
    x : int | float
    y : int | float

    Returns
    -------
    Expression
    """
    return expression.Expression("beta(", f"{x},{y}", ")")


def regularized_beta(
    x: Union[int, float], y: Union[int, float], z: Union[int, float]
) -> "Expression":
    """
    Beta function

    Parameters
    ----------
    x : int | float
    y : int | float
    z : int | float

    Returns
    -------
    Expression
    """
    return expression.Expression("betaReg(", f"{x},{y},{z}", ")")


def gamma(x: Union[int, float], y: Union[int, float]) -> "Expression":
    """
    Gamma function

    Parameters
    ----------
    x : int | float
    y : int | float

    Returns
    -------
    Expression
    """
    return expression.Expression("gamma(", f"{x},{y}", ")")


def lse_max(x: "Symbol") -> "Expression":
    """
    Smoothed Max via the Logarithm of the Sum of Exponentials

    Parameters
    ----------
    x : Symbol

    Returns
    -------
    Expression
    """
    return expression.Expression("lseMax(", x.gamsRepr(), ")")


def lse_max_sc(t: "Symbol", x: "Symbol") -> "Expression":
    """
    Scaled smoothed Max via the Logarithm of the Sum of Exponentials

    Parameters
    ----------
    x : Symbol

    Returns
    -------
    Expression
    """
    return expression.Expression(
        "lseMaxSc(", f"{t.gamsRepr()},{x.gamsRepr()}", ")"
    )


def lse_min(x: "Symbol") -> "Expression":
    """
    Smoothed Min via the Logarithm of the Sum of Exponentials

    Parameters
    ----------
    x : Symbol

    Returns
    -------
    Expression
    """
    return expression.Expression("lseMin(", x.gamsRepr(), ")")


def lse_min_sc(t: "Symbol", x: "Symbol") -> "Expression":
    """
    Scaled smoothed Min via the Logarithm of the Sum of Exponentials

    Parameters
    ----------
    x : Symbol

    Returns
    -------
    Expression
    """
    return expression.Expression(
        "lseMinSc(", f"{t.gamsRepr()},{x.gamsRepr()}", ")"
    )


def ncp_cm(
    x: "Symbol",
    y: "Symbol",
    z: Union[float, int],
) -> "Expression":
    """
    Chen-Mangasarian smoothing

    Parameters
    ----------
    x : Symbol
    y : Symbol
    z : int | float

    Returns
    -------
    Expression
    """
    return expression.Expression(
        "ncpCM(", ",".join([x.gamsRepr(), y.gamsRepr(), str(z)]), ")"
    )


def ncp_f(
    x: "Symbol",
    y: "Symbol",
    z: Union[int, float] = 0,
) -> "Expression":
    """
    Fisher-Burmeister smoothing

    Parameters
    ----------
    x : Symbol
    y : Symbol
    z : int | float, optional

    Returns
    -------
    Expression
    """
    return expression.Expression(
        "ncpF(", ",".join([x.gamsRepr(), y.gamsRepr(), str(z)]), ")"
    )


def ncpVUpow(
    r: "Symbol",
    s: "Symbol",
    mu: Union[int, float] = 0,
) -> "Expression":
    """
    NCP Veelken-Ulbrich: smoothed min(r,s)

    Parameters
    ----------
    r : Symbol
    s : Symbol
    mu : int | float, optional

    Returns
    -------
    Expression
    """
    return expression.Expression(
        "ncpVUpow(", ",".join([r.gamsRepr(), s.gamsRepr(), str(mu)]), ")"
    )


def ncpVUsin(
    r: "Symbol",
    s: "Symbol",
    mu: Union[int, float] = 0,
) -> "Expression":
    """
    NCP Veelken-Ulbrich: smoothed min(r,s)

    Parameters
    ----------
    r : Symbol
    s : Symbol
    mu : int | float, optional

    Returns
    -------
    Expression
    """
    return expression.Expression(
        "ncpVUsin(", ",".join([r.gamsRepr(), s.gamsRepr(), str(mu)]), ")"
    )


def poly(x: "Symbol", *args) -> "Expression":
    """
    Polynomial function

    Parameters
    ----------
    x : Symbol

    Returns
    -------
    Expression
    """
    args_str = ",".join(args)

    return expression.Expression("poly(", f"{x.gamsRepr()}, {args_str}", ")")


def rand_binomial():
    ...


def rand_linear():
    ...


def rand_triangle():
    ...


def regularized_gamma(
    x: Union[int, float], y: Union[int, float], z: Union[int, float]
) -> "Expression":
    """
    Gamma function

    Parameters
    ----------
    x : int | float
    y : int | float
    z : int | float

    Returns
    -------
    Expression
    """
    return expression.Expression("gammaReg(", f"{x},{y},{z}", ")")


def slrec():
    ...


def sqrec():
    ...


def entropy(x: Union[int, float, "Symbol"]) -> "Expression":
    """
    L2 Norm of x

    Parameters
    ----------
    x : int | float | Symbol

    Returns
    -------
    Expression
    """
    x_str = str(x) if isinstance(x, (int, float)) else x.gamsRepr()
    return expression.Expression("entropy(", x_str, ")")


def errorf(x: Union[int, float, "Symbol"]) -> "Expression":
    """
    Integral of the standard normal distribution

    Parameters
    ----------
    x : int, float, Symbol

    Returns
    -------
    Expression
    """
    x_str = str(x) if isinstance(x, (int, float)) else x.gamsRepr()
    return expression.Expression("errorf(", x_str, ")")
