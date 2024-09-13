from __future__ import annotations

import math

from gamspy.exceptions import ValidationError


def _check_tuple_int(
    value: int | tuple[int, int], name: str, allow_zero=False
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


def _calc_hw(
    padding: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    h_in: int,
    w_in: int,
) -> tuple[int, int]:
    h_out = math.floor(
        1 + ((h_in + (2 * padding[0]) - (kernel_size[0] - 1) - 1) / stride[0])
    )
    w_out = math.floor(
        1 + ((w_in + (2 * padding[1]) - (kernel_size[1] - 1) - 1) / stride[1])
    )
    return h_out, w_out
