from __future__ import annotations

import math

import numpy as np

import gamspy as gp
from gamspy.exceptions import ValidationError


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


def _generate_name(sym_type: str, prefix: str, name: str) -> str:
    rand = gp.utils._get_unique_name()[:5]
    return "_".join([sym_type, prefix, name, rand])


def _check_tuple_int(
    value: int | tuple[int, int],
    name: str,
    allow_zero=False,
) -> tuple[int, int]:
    if not isinstance(value, (int, tuple)):
        raise ValidationError(
            f"{name} must be an integer or a tuple of integer"
        )

    if isinstance(value, int):
        value = (value, value)

    cmp = (lambda a: a >= 0) if allow_zero else (lambda a: a > 0)
    text = "or equal to " if allow_zero else ""

    if not (isinstance(value[0], int) and cmp(value[0])):
        raise ValidationError(f"{name} must be greater than {text}0")

    if not (isinstance(value[1], int) and cmp(value[1])):
        raise ValidationError(f"{name} must be a greater than {text}0")

    return value


def _check_padding(value: int | tuple[int, int]) -> tuple[int, int, int, int]:
    if not isinstance(value, (int, tuple)):
        raise ValidationError(
            "Padding must be an integer or a tuple of two integers"
        )

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


def _calc_same_padding_2d(
    kernel_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    # assumes stride = 1
    pad_h_total = max(kernel_size[0] - 1, 0)
    pad_w_total = max(kernel_size[1] - 1, 0)

    # Calculate padding for height
    pad_top = pad_h_total // 2
    pad_bottom = pad_h_total - pad_top

    # Calculate padding for width
    pad_left = pad_w_total // 2
    pad_right = pad_w_total - pad_left

    return (pad_top, pad_left, pad_bottom, pad_right)


def _calc_same_padding_1d(
    kernel_size: int,
) -> tuple[int, int]:
    # assumes stride = 1
    pad_w_total = max(kernel_size - 1, 0)

    # Calculate padding for width
    pad_left = pad_w_total // 2
    pad_right = pad_w_total - pad_left

    return (pad_left, pad_right)


def _calc_w(
    padding: tuple[int, int] | str,
    kernel_size: int,
    stride: int,
    w_in: int,
) -> int:
    # same padding
    if isinstance(padding, str):
        return w_in

    w_out = math.floor(
        1
        + ((w_in + (padding[0] + padding[1]) - (kernel_size - 1) - 1) / stride)
    )

    return w_out


def _calc_hw(
    padding: tuple[int, int, int, int] | tuple[int, int] | str,
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

    h_out = math.floor(
        1
        + (
            (h_in + (padding[0] + padding[2]) - (kernel_size[0] - 1) - 1)
            / stride[0]
        )
    )
    w_out = math.floor(
        1
        + (
            (w_in + (padding[1] + padding[3]) - (kernel_size[1] - 1) - 1)
            / stride[1]
        )
    )

    return h_out, w_out


def _next_domains(
    input_domain: list[gp.Set], check_domains: list[gp.Set]
) -> list[gp.Set]:
    names = {x.name for x in check_domains}
    output = []
    for domain in input_domain:
        while domain.name in names:
            domain = gp.math.next_alias(domain)

        output.append(domain)
        names.add(domain.name)

    return output
