from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
from typing import cast

import gamspy as gp
from gamspy.exceptions import ValidationError
from gamspy.formulations._piecewise_data import (
    _copy_row_with_key,
    _get_explicit_domain_symbols,
    _match_curve_domain,
    _normalize_sequence_data,
    _PwlData,
    _PwlRow,
    _to_finite_number,
)


@dataclass(frozen=True)
class PWLCurve:
    """Compact description of a piecewise-linear curve.

    Parameters
    ----------
    points : Sequence[Sequence[float] | None]
        Ordered coordinate pairs. ``None`` disconnects the points on its two
        sides, equal consecutive x-coordinates describe a discontinuity, and a
        decreasing x-coordinate starts a new independent chain.
    left_gradient : float | None, optional
        Gradient of an unbounded ray extending from the first point to the
        left.
    right_gradient : float | None, optional
        Gradient of an unbounded ray extending from the last point to the
        right.

    Examples
    --------
    >>> from gamspy.formulations import PWLCurve
    >>> curve = PWLCurve([(0, 0), (2, 4), None, (4, 8), (6, 12)])

    """

    points: Sequence[Sequence[float] | None]
    left_gradient: float | None = None
    right_gradient: float | None = None
    _normalized_row: _PwlRow = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        points = _normalize_points(self.points)
        left_gradient = _normalize_gradient(self.left_gradient, "left_gradient", 0)
        right_gradient = _normalize_gradient(
            self.right_gradient, "right_gradient", len(points) - 1
        )

        x_points = [point[0] if point is not None else None for point in points]
        y_points = [point[1] if point is not None else None for point in points]
        if left_gradient is not None:
            x_points.insert(0, -math.inf)
            y_points.insert(0, left_gradient)
        if right_gradient is not None:
            x_points.append(math.inf)
            y_points.append(right_gradient)

        row = _normalize_sequence_data(
            x_points,
            y_points,
            bound_left=True,
            bound_right=True,
        )[0]

        object.__setattr__(self, "points", points)
        object.__setattr__(self, "left_gradient", left_gradient)
        object.__setattr__(self, "right_gradient", right_gradient)
        object.__setattr__(self, "_normalized_row", row)


def _normalize_curve_data(
    input_x: object,
    curve: object,
    curve_domain: object = None,
) -> _PwlData:
    if not isinstance(input_x, gp.Variable):
        raise ValidationError("input_x is expected to be a Variable")

    if input_x.dimension == 0:
        if curve_domain is not None:
            raise ValidationError(
                "curve_domain can only be used with a curve dictionary and an "
                "indexed input_x"
            )
        if isinstance(curve, Mapping):
            raise ValidationError(
                "a curve dictionary requires an indexed input_x; pass one PWLCurve "
                "for scalar input_x"
            )
        elif not isinstance(curve, PWLCurve):
            raise ValidationError("curve must be a PWLCurve")
        else:
            return _PwlData(rows=(curve._normalized_row,), curve_domain=())

    # A shared curve does not require expansion of the input row keys.
    input_domain = _get_explicit_domain_symbols(
        input_x, context="piecewise linear formulations"
    )
    if isinstance(curve, PWLCurve):
        if curve_domain is not None:
            raise ValidationError(
                "curve_domain can only be used with a curve dictionary"
            )
        return _PwlData(rows=(curve._normalized_row,), curve_domain=())

    if not isinstance(curve, Mapping):
        raise ValidationError(
            "curve must be a PWLCurve or a dictionary whose keys are domain "
            "labels and values are PWLCurve objects"
        )

    if curve_domain is None:
        normalized_curve_domain = input_domain
    else:
        if isinstance(curve_domain, (str, bytes)) or not isinstance(
            curve_domain, Sequence
        ):
            raise ValidationError(
                "curve_domain must be a list or tuple of input_x domains"
            )
        if not curve_domain:
            raise ValidationError(
                "curve_domain must contain at least one input_x domain"
            )
        normalized_curve_domain = _match_curve_domain(input_domain, curve_domain)

    row_keys = tuple(product(*(domain.toList() for domain in normalized_curve_domain)))
    curve_mapping = cast("Mapping[object, object]", curve)
    mapping_keys = tuple(
        key[0] if len(normalized_curve_domain) == 1 else key for key in row_keys
    )
    expected_keys = set(mapping_keys)
    provided_keys = set(curve_mapping)
    missing_keys = expected_keys - provided_keys
    extra_keys = provided_keys - expected_keys
    if missing_keys or extra_keys:
        details = []
        if missing_keys:
            details.append(f"missing keys: {_format_keys(missing_keys)}")
        if extra_keys:
            details.append(f"extra keys: {_format_keys(extra_keys)}")
        raise ValidationError(
            "curve dictionary keys must exactly match curve_domain; "
            + "; ".join(details)
        )

    rows: list[_PwlRow] = []
    for row_key, mapping_key in zip(row_keys, mapping_keys, strict=True):
        row_curve = curve_mapping[mapping_key]
        if not isinstance(row_curve, PWLCurve):
            raise ValidationError(
                f"curve dictionary value for key {mapping_key!r} must be a PWLCurve"
            )
        rows.append(_copy_row_with_key(row_curve._normalized_row, row_key))

    return _PwlData(rows=tuple(rows), curve_domain=normalized_curve_domain)


def _format_keys(keys: Iterable[object]) -> str:
    return ", ".join(sorted(repr(key) for key in keys))


def _normalize_points(
    points: Sequence[Sequence[float] | None],
) -> tuple[tuple[float, float] | None, ...]:
    if isinstance(points, (str, bytes)) or not isinstance(points, Sequence):
        raise ValidationError("points must be a list or tuple")

    normalized: list[tuple[float, float] | None] = []
    for position, point in enumerate(points):
        if point is None:
            normalized.append(None)
            continue

        if (
            isinstance(point, (str, bytes))
            or not isinstance(point, Sequence)
            or len(point) != 2
        ):
            raise ValidationError(
                f"point at position {position} must be an (x, y) pair or None"
            )

        normalized.append(
            (
                _to_finite_number(point[0], "x-coordinate", position),
                _to_finite_number(point[1], "y-coordinate", position),
            )
        )

    return tuple(normalized)


def _normalize_gradient(
    gradient: float | None, name: str, position: int
) -> float | None:
    if gradient is None:
        return None

    return _to_finite_number(gradient, name, position)
