import gamspy._algebra._expression as expression
from typing import Union, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from gamspy._algebra._expression import Expression


def centropy(x, y, z: Optional[float] = None):
    if z:
        return expression.Expression(
            "centropy(", ",".join([x.gamsRepr(), y.gamsRepr(), str(z)]), ")"
        )

    return expression.Expression(
        "centropy(", ",".join([x.gamsRepr(), y.gamsRepr()]), ")"
    )


def uniform(lower_bound, upper_bound) -> "Expression":
    return expression.Expression(
        "uniform(", f"{lower_bound},{upper_bound}", ")"
    )


def uniformInt(lower_bound, upper_bound) -> "Expression":
    return expression.Expression(
        "uniformInt(", f"{lower_bound},{upper_bound}", ")"
    )


def normal(mean: Union[int, float], dev: Union[int, float]) -> "Expression":
    return expression.Expression("normal(", f"{mean},{dev}", ")")
