from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np

import gamspy as gp
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from gamspy._types import SymbolWithRecordsType
    from gamspy._universe import Universe


def _encode_infinity(x: np.ndarray) -> np.ndarray:
    """
    Encode infinity values as complex numbers.
    - Replace -np.inf with 0 - 1j.
    - Replace np.inf with 0 + 1j.
    """
    x = np.where(x == -np.inf, 0 - 1j, x)
    x = np.where(x == np.inf, 0 + 1j, x)
    return x


def _decode_complex_array(z: np.ndarray) -> np.ndarray:
    """
    Decode complex numbers back to infinities
    - Replace 0 - 1j -> -np.inf
    - Replace 0 + 1j -> np.inf
    """
    real = z.real.copy()
    real[z.imag > 0] = np.inf
    real[z.imag < 0] = -np.inf
    return real


def _apply_real_linear(
    fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray
) -> np.ndarray:
    """
    Apply a linear map with real coefficients to `x`. Complex (infinity
    encoded) inputs are split into two real products, which is several
    times faster than complex BLAS and avoids its multi-threading.
    """
    if np.iscomplexobj(x):
        return fn(x.real) + 1j * fn(x.imag)

    return fn(x)


def _generate_name(sym_type: str, prefix: str, name: str) -> str:
    rand = gp.utils._get_unique_name()[:5]
    return "_".join([sym_type, prefix, name, rand])


def _check_tuple_int(
    value: int | tuple[int] | tuple[int, int],
    name: str,
    *,
    allow_zero: bool = False,
) -> tuple[int, int]:
    if not isinstance(value, (int, tuple)):
        raise ValidationError(f"{name} must be an integer or a tuple of integer")

    if isinstance(value, int):
        value = (value, value)
    elif len(value) == 1:
        value = (value[0], value[0])

    cmp = (lambda a: a >= 0) if allow_zero else (lambda a: a > 0)
    text = "or equal to " if allow_zero else ""

    if not (isinstance(value[0], int) and cmp(value[0])):
        raise ValidationError(f"{name} must be greater than {text}0")

    if not (isinstance(value[1], int) and cmp(value[1])):
        raise ValidationError(f"{name} must be a greater than {text}0")

    return value


def _check_padding(value: int | tuple[int, int]) -> tuple[int, int, int, int]:
    if not isinstance(value, (int, tuple)):
        raise ValidationError("Padding must be an integer or a tuple of two integers")

    if isinstance(value, int):
        padding = (value, value, value, value)

    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValidationError(
                f"Padding can only be a tuple of 2 integers. Not {len(value)}."
            )

        # padding is represented as (top, left, bottom, right)
        padding = (value[0], value[1], value[0], value[1])

    if not (isinstance(padding[0], int) and (padding[0] >= 0)):
        raise ValidationError("Padding must be greater than or equal to 0")

    if not (isinstance(padding[1], int) and (padding[1] >= 0)):
        raise ValidationError("Padding must be a greater than or equal to 0")

    return padding


def _calc_same_padding(kernel_size: Sequence[int]) -> tuple[int, ...]:
    """
    Padding that keeps the spatial size with stride 1, laid out as all
    leading paddings followed by all trailing paddings, e.g. (top, left, bottom, right).
    """
    before = tuple(max(k - 1, 0) // 2 for k in kernel_size)
    after = tuple(max(k - 1, 0) - b for k, b in zip(kernel_size, before))
    return before + after


def _calc_output_size(
    padding: Sequence[int],
    kernel_size: Sequence[int],
    stride: Sequence[int],
    sizes: Sequence[int],
) -> tuple[int, ...]:
    n = len(sizes)
    return tuple(
        math.floor(
            1
            + (
                (sizes[d] + (padding[d] + padding[n + d]) - (kernel_size[d] - 1) - 1)
                / stride[d]
            )
        )
        for d in range(n)
    )


def _calc_hw(
    padding: Sequence[int] | str,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    h_in: int,
    w_in: int,
) -> tuple[int, int]:
    if isinstance(padding, str):
        return h_in, w_in

    if len(padding) == 2:
        # padding is represented as (top, left, bottom, right)
        padding = (padding[0], padding[1], padding[0], padding[1])

    h_out, w_out = _calc_output_size(padding, kernel_size, stride, (h_in, w_in))
    return h_out, w_out


def _get_domain(symbol: SymbolWithRecordsType) -> list[gp.Set | gp.Alias]:
    """
    Domain of the symbol narrowed down to the sets and aliases that formulations
    index over. Formulations only operate on symbols indexed over concrete sets/aliases.
    """
    return cast("list[gp.Set | gp.Alias]", list(symbol.domain))


def _domain_name(symbol: gp.Set | gp.Alias | gp.UniverseAlias | Universe | str) -> str:
    return symbol if isinstance(symbol, str) else symbol.name


def _next_domains(
    input_domain: Sequence[gp.Alias | gp.Set],
    check_domains: Sequence[gp.Alias | gp.Set],
) -> list[gp.Set | gp.Alias]:
    names = {x.name for x in check_domains}
    output = []
    for domain in input_domain:
        while domain.name in names:
            domain = gp.math.next_alias(domain)

        output.append(domain)
        names.add(domain.name)

    return output
