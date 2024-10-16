from __future__ import annotations

import math

import gamspy as gp
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


def _calc_same_padding(
    kernel_size: tuple[int, int],
    h_in: int,
    w_in: int,
) -> tuple[int, int]:
    # assumes stride = 1
    pad_h = math.floor((kernel_size[0] - 1) / 2)
    pad_w = math.floor((kernel_size[1] - 1) / 2)
    return pad_h, pad_w


def _calc_hw(
    padding: tuple[int, int] | str,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    h_in: int,
    w_in: int,
) -> tuple[int, int]:
    if isinstance(padding, str):
        return h_in, w_in

    h_out = math.floor(
        1 + ((h_in + (2 * padding[0]) - (kernel_size[0] - 1) - 1) / stride[0])
    )
    w_out = math.floor(
        1 + ((w_in + (2 * padding[1]) - (kernel_size[1] - 1) - 1) / stride[1])
    )
    return h_out, w_out


def _next_domains(
    input_domain: list[gp.Set], check_domains: list[gp.Set]
) -> list[gp.Set]:
    names = set([x.name for x in check_domains])
    output = []
    for domain in input_domain:
        while domain.name in names:
            domain = gp.math.next_alias(domain)

        output.append(domain)
        names.add(domain.name)

    return output
