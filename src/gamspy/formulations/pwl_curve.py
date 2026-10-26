from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

import gamspy as gp
from gamspy.exceptions import ValidationError
from gamspy.formulations._piecewise_data import (
    _copy_row_with_key,
    _get_curve_row_keys,
    _get_explicit_domain_symbols,
    _match_curve_domain,
    _normalize_label_key,
    _normalize_restriction,
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
        """
        Validate and freeze curve data when a PWLCurve is created.
        """
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
    *,
    restrict: gp.Set | None = None,
) -> _PwlData:
    """
    Prepare a shared PWLCurve or an indexed curve dictionary for pwlinear.
    Dictionary keys follow curve_domain; omitted input domains are broadcast.
    """
    if not isinstance(input_x, gp.Variable):
        raise ValidationError("input_x is expected to be a Variable")

    input_keys = _normalize_restriction(input_x, restrict)
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
            return _PwlData(
                rows=(curve._normalized_row,), curve_domain=(), input_keys=input_keys
            )

    # A shared curve does not require expansion of the input row keys.
    input_domain = _get_explicit_domain_symbols(
        input_x, context="piecewise linear formulations"
    )
    if isinstance(curve, PWLCurve):
        if curve_domain is not None:
            raise ValidationError(
                "curve_domain can only be used with a curve dictionary"
            )
        return _PwlData(
            rows=(curve._normalized_row,), curve_domain=(), input_keys=input_keys
        )

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

    row_keys = _get_curve_row_keys(input_domain, normalized_curve_domain, input_keys)
    curve_mapping = cast("Mapping[object, object]", curve)
    mapping_keys = tuple(
        key[0] if len(normalized_curve_domain) == 1 else key for key in row_keys
    )
    expected_keys = {_normalize_label_key(key) for key in mapping_keys}
    provided_keys: dict[object, object] = {}
    for key in curve_mapping:
        normalized_key = _normalize_label_key(key)
        if normalized_key in provided_keys:
            raise ValidationError(
                "curve dictionary keys are case-insensitive. "
                f"Supplied {provided_keys[normalized_key]!r} and {key!r}."
            )
        provided_keys[normalized_key] = key
    missing_keys = {
        key for key in mapping_keys if _normalize_label_key(key) not in provided_keys
    }
    extra_keys = provided_keys.keys() - expected_keys
    if input_keys is not None:
        # Unselected curves may remain in the dictionary, but their keys must
        # still belong to the declared curve domains.
        domain_labels = [
            {label.casefold() for label in domain.toList()}
            for domain in normalized_curve_domain
        ]
        extra_keys = set()
        for mapping_key in provided_keys.keys() - expected_keys:
            key = (mapping_key,) if len(domain_labels) == 1 else mapping_key
            if (
                not isinstance(key, tuple)
                or len(key) != len(domain_labels)
                or any(
                    label not in labels
                    for label, labels in zip(key, domain_labels, strict=True)
                )
            ):
                extra_keys.add(mapping_key)

    if missing_keys:
        raise ValidationError(f"curve is missing keys: {_format_keys(missing_keys)}")
    if extra_keys:
        raise ValidationError(
            "curve contains invalid keys: "
            f"{_format_keys(provided_keys[key] for key in extra_keys)}"
        )

    for mapping_key, row_curve in curve_mapping.items():
        if not isinstance(row_curve, PWLCurve):
            raise ValidationError(f"curve[{mapping_key!r}] must be a PWLCurve")

    curve_mapping = cast("Mapping[object, PWLCurve]", curve_mapping)
    rows = tuple(
        _copy_row_with_key(
            curve_mapping[
                provided_keys[_normalize_label_key(mapping_key)]
            ]._normalized_row,
            row_key,
        )
        for row_key, mapping_key in zip(row_keys, mapping_keys, strict=True)
    )
    return _PwlData(
        rows=rows, curve_domain=normalized_curve_domain, input_keys=input_keys
    )


def _format_keys(keys: Iterable[object]) -> str:
    return ", ".join(sorted(repr(key) for key in keys))


def _normalize_points(
    points: Sequence[Sequence[float] | None],
) -> tuple[tuple[float, float] | None, ...]:
    """
    Copy PWLCurve points into immutable pairs with finite coordinates.
    Rays are specified separately through the gradient arguments.
    """
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
