from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise, product
from numbers import Real
from typing import Literal

import gamspy as gp
from gamspy.exceptions import ValidationError


@dataclass(frozen=True)
class _PwlPoint:
    x: float
    y: float


@dataclass(frozen=True)
class _PwlSegment:
    left: _PwlPoint
    right: _PwlPoint


@dataclass(frozen=True)
class _PwlRay:
    base: _PwlPoint
    gradient: float
    direction: Literal[-1, 1]


@dataclass(frozen=True)
class _PwlRow:
    key: tuple[str, ...]
    ordered_points: tuple[_PwlPoint, ...]
    separator_indices: tuple[int, ...]
    segments: tuple[_PwlSegment, ...]
    rays: tuple[_PwlRay, ...]


@dataclass(frozen=True)
class _PwlData:
    rows: tuple[_PwlRow, ...]
    curve_domain: tuple[gp.Set | gp.Alias, ...]

    @property
    def is_broadcast(self) -> bool:
        return not self.curve_domain


@dataclass(frozen=True)
class _PwlDiscontinuity:
    row_key: tuple[str, ...]
    x: float
    y_values: tuple[float, ...]


@dataclass(frozen=True)
class _PwlOverlap:
    row_key: tuple[str, ...]
    x_lower: float
    x_upper: float


@dataclass(frozen=True)
class _PwlMultivaluedInfo:
    discontinuities: tuple[_PwlDiscontinuity, ...]
    overlaps: tuple[_PwlOverlap, ...]


def _copy_row_with_key(row: _PwlRow, key: tuple[str, ...]) -> _PwlRow:
    return _PwlRow(
        key=key,
        ordered_points=row.ordered_points,
        separator_indices=row.separator_indices,
        segments=row.segments,
        rays=row.rays,
    )


def _classify_multivalued_rows(
    rows: tuple[_PwlRow, ...],
) -> _PwlMultivaluedInfo:
    discontinuities: list[_PwlDiscontinuity] = []
    overlaps: list[_PwlOverlap] = []
    for row in rows:
        row_info = _classify_multivalued_row(row)
        discontinuities.extend(row_info.discontinuities)
        overlaps.extend(row_info.overlaps)

    return _PwlMultivaluedInfo(tuple(discontinuities), tuple(overlaps))


def _classify_multivalued_row(row: _PwlRow) -> _PwlMultivaluedInfo:
    chains = _split_point_chains(row)
    chain_ranges = [_get_chain_x_range(chain, row.rays) for chain in chains]
    point_values: dict[float, set[float]] = {}
    overlap_ranges: list[tuple[float, float]] = []

    for left_index, left_chain in enumerate(chains):
        left_lower, left_upper = chain_ranges[left_index]
        for right_index in range(left_index + 1, len(chains)):
            right_chain = chains[right_index]
            right_lower, right_upper = chain_ranges[right_index]
            overlap_lower = max(left_lower, right_lower)
            overlap_upper = min(left_upper, right_upper)

            if overlap_lower > overlap_upper:
                continue

            # A positive-width intersection is an overlapping range. A single
            # shared x-coordinate is multivalued only when its y-values differ.
            if overlap_lower < overlap_upper:
                overlap_ranges.append((overlap_lower, overlap_upper))
                continue

            left_y = _get_chain_y_value(left_chain, row.rays, overlap_lower)
            right_y = _get_chain_y_value(right_chain, row.rays, overlap_lower)
            if left_y != right_y:
                point_values.setdefault(overlap_lower, set()).update((left_y, right_y))

    merged_overlaps = _merge_overlap_ranges(overlap_ranges)
    discontinuities = tuple(
        _PwlDiscontinuity(row.key, x, tuple(sorted(y_values)))
        for x, y_values in sorted(point_values.items())
        if not any(x_lower <= x <= x_upper for x_lower, x_upper in merged_overlaps)
    )
    overlaps = tuple(
        _PwlOverlap(row.key, x_lower, x_upper) for x_lower, x_upper in merged_overlaps
    )

    return _PwlMultivaluedInfo(discontinuities, overlaps)


def _split_point_chains(row: _PwlRow) -> tuple[tuple[_PwlPoint, ...], ...]:
    chains: list[tuple[_PwlPoint, ...]] = []
    first = 0
    for separator_index in row.separator_indices:
        chains.append(row.ordered_points[first : separator_index + 1])
        first = separator_index + 1
    chains.append(row.ordered_points[first:])
    return tuple(chains)


def _get_chain_x_range(
    chain: tuple[_PwlPoint, ...], rays: tuple[_PwlRay, ...]
) -> tuple[float, float]:
    x_lower = chain[0].x
    x_upper = chain[-1].x
    for ray in rays:
        if ray.direction == -1 and ray.base is chain[0]:
            x_lower = -math.inf
        elif ray.direction == 1 and ray.base is chain[-1]:
            x_upper = math.inf

    return x_lower, x_upper


def _get_chain_y_value(
    chain: tuple[_PwlPoint, ...], rays: tuple[_PwlRay, ...], x: float
) -> float:
    for point in chain:
        if point.x == x:
            return point.y

    for left, right in pairwise(chain):
        if left.x < x < right.x:
            gradient = (right.y - left.y) / (right.x - left.x)
            return left.y + gradient * (x - left.x)

    for ray in rays:
        if ray.base is chain[0] and ray.direction == -1 and x < ray.base.x:
            return ray.base.y + ray.gradient * (x - ray.base.x)
        if ray.base is chain[-1] and ray.direction == 1 and x > ray.base.x:
            return ray.base.y + ray.gradient * (x - ray.base.x)

    raise AssertionError("x must belong to the chain domain")


def _merge_overlap_ranges(
    ranges: list[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    merged: list[tuple[float, float]] = []
    for x_lower, x_upper in sorted(ranges):
        if not merged or x_lower > merged[-1][1]:
            merged.append((x_lower, x_upper))
        else:
            previous_lower, previous_upper = merged[-1]
            merged[-1] = (previous_lower, max(previous_upper, x_upper))

    return tuple(merged)


def _get_explicit_domain_symbols(
    input_x: gp.Variable, *, context: str
) -> tuple[gp.Set | gp.Alias, ...]:
    domains: list[gp.Set | gp.Alias] = []
    for domain in input_x.domain:
        if not isinstance(domain, (gp.Set, gp.Alias)):
            raise ValidationError(
                f"input_x domains must be explicit Sets or Aliases for {context}"
            )
        if len(domain) == 0:
            raise ValidationError(f"input_x domain cannot be empty for {context}")
        domains.append(domain)

    return tuple(domains)


def _normalize_paired_data(
    input_x: object,
    x_points: object,
    y_points: object,
    *,
    bound_left: bool,
    bound_right: bool,
) -> _PwlData:
    if not isinstance(input_x, gp.Variable):
        raise ValidationError("input_x is expected to be a Variable")

    if input_x.dimension:
        _get_explicit_domain_symbols(input_x, context="piecewise linear formulations")

    x_is_parameter = isinstance(x_points, gp.Parameter)
    y_is_parameter = isinstance(y_points, gp.Parameter)

    if input_x.dimension == 0:
        if x_is_parameter or y_is_parameter:
            return _normalize_scalar_parameter_data(
                input_x,
                x_points,
                y_points,
                bound_left=bound_left,
                bound_right=bound_right,
            )
        else:
            return _PwlData(
                rows=_normalize_sequence_data(
                    x_points,
                    y_points,
                    bound_left=bound_left,
                    bound_right=bound_right,
                ),
                curve_domain=(),
            )
    # One-dimensional Parameters describe one graph and broadcast like sequences.
    elif (
        x_is_parameter
        and y_is_parameter
        and x_points.dimension == 1
        and y_points.dimension == 1
    ):
        return _normalize_broadcast_parameter_data(
            input_x,
            x_points,
            y_points,
            bound_left=bound_left,
            bound_right=bound_right,
        )
    elif x_is_parameter or y_is_parameter:
        return _normalize_indexed_parameter_data(
            input_x,
            x_points,
            y_points,
            bound_left=bound_left,
            bound_right=bound_right,
        )
    else:
        return _PwlData(
            rows=_normalize_sequence_data(
                x_points,
                y_points,
                bound_left=bound_left,
                bound_right=bound_right,
            ),
            curve_domain=(),
        )


def _normalize_sequence_data(
    x_points: object,
    y_points: object,
    *,
    bound_left: bool,
    bound_right: bool,
    row_keys: Sequence[tuple[str, ...]] = ((),),
) -> tuple[_PwlRow, ...]:
    if not isinstance(bound_left, bool):
        raise ValidationError("bound_left is expected to be a boolean")

    if not isinstance(bound_right, bool):
        raise ValidationError("bound_right is expected to be a boolean")

    x_values = _to_point_sequence(x_points, "x_points")
    y_values = _to_point_sequence(y_points, "y_points")
    row = _normalize_sequence_row(
        x_values,
        y_values,
        bound_left=bound_left,
        bound_right=bound_right,
    )

    # Shared sequence data is normalized once and copied to every input row.
    return tuple(_copy_row_with_key(row, key) for key in row_keys)


def _normalize_scalar_parameter_data(
    input_x: object,
    x_points: object,
    y_points: object,
    *,
    bound_left: bool,
    bound_right: bool,
) -> _PwlData:
    if not isinstance(input_x, gp.Variable) or input_x.dimension != 0:
        raise ValidationError("input_x is expected to be a scalar Variable")

    return _normalize_one_dimensional_parameter_data(
        input_x,
        x_points,
        y_points,
        context="scalar input_x",
        bound_left=bound_left,
        bound_right=bound_right,
    )


def _normalize_broadcast_parameter_data(
    input_x: object,
    x_points: object,
    y_points: object,
    *,
    bound_left: bool,
    bound_right: bool,
) -> _PwlData:
    if not isinstance(input_x, gp.Variable) or input_x.dimension == 0:
        raise ValidationError("input_x is expected to be an indexed Variable")

    return _normalize_one_dimensional_parameter_data(
        input_x,
        x_points,
        y_points,
        context="indexed input_x broadcasting",
        bound_left=bound_left,
        bound_right=bound_right,
    )


def _normalize_one_dimensional_parameter_data(
    input_x: gp.Variable,
    x_points: object,
    y_points: object,
    *,
    context: str,
    bound_left: bool,
    bound_right: bool,
) -> _PwlData:
    x_points, y_points = _validate_parameter_pair(input_x, x_points, y_points)

    if x_points.dimension != 1 or y_points.dimension != 1:
        raise ValidationError(
            f"x_points and y_points must be one-dimensional Parameters for {context}"
        )

    if x_points.domain[0] is not y_points.domain[0]:
        raise ValidationError("y_points domain must exactly match x_points domain")

    if x_points.shape != y_points.shape:
        raise ValidationError("x_points and y_points must have the same shape")

    return _PwlData(
        rows=(
            _normalize_parameter_row(
                x_points.toDense(),
                y_points.toDense(),
                key=(),
                trim_trailing_padding=False,
                bound_left=bound_left,
                bound_right=bound_right,
            ),
        ),
        curve_domain=(),
    )


def _normalize_indexed_parameter_data(
    input_x: object,
    x_points: object,
    y_points: object,
    *,
    bound_left: bool,
    bound_right: bool,
) -> _PwlData:
    if not isinstance(input_x, gp.Variable) or input_x.dimension == 0:
        raise ValidationError("input_x is expected to be an indexed Variable")

    x_points, y_points = _validate_parameter_pair(input_x, x_points, y_points)

    if x_points.dimension != y_points.dimension:
        raise ValidationError("x_points and y_points must have matching dimensions")

    if x_points.dimension < 2:
        raise ValidationError(
            "indexed x_points and y_points must have at least one curve domain "
            "followed by one breakpoint domain"
        )

    if any(
        x_domain is not y_domain
        for x_domain, y_domain in zip(x_points.domain, y_points.domain, strict=True)
    ):
        raise ValidationError("y_points domain must exactly match x_points domain")

    if x_points.shape != y_points.shape:
        raise ValidationError("x_points and y_points must have the same shape")

    input_domain = _get_explicit_domain_symbols(input_x, context="indexed Parameters")
    curve_domain = _match_curve_domain(input_domain, x_points.domain[:-1])

    # Parameter data follows domain order. Flatten the curve dimensions into
    # row keys and keep the final breakpoint dimension intact.
    row_keys = tuple(product(*(domain.toList() for domain in curve_domain)))
    x_values = x_points.toDense().reshape((len(row_keys), x_points.shape[-1]))
    y_values = y_points.toDense().reshape((len(row_keys), y_points.shape[-1]))

    return _PwlData(
        rows=tuple(
            _normalize_parameter_row(
                x_row,
                y_row,
                key=key,
                trim_trailing_padding=True,
                bound_left=bound_left,
                bound_right=bound_right,
            )
            for key, x_row, y_row in zip(row_keys, x_values, y_values, strict=True)
        ),
        curve_domain=curve_domain,
    )


def _match_curve_domain(
    input_domain: tuple[gp.Set | gp.Alias, ...],
    parameter_domain: Sequence[object],
) -> tuple[gp.Set | gp.Alias, ...]:
    curve_domain: list[gp.Set | gp.Alias] = []
    input_position = 0
    for domain in parameter_domain:
        while (
            input_position < len(input_domain)
            and input_domain[input_position] is not domain
        ):
            input_position += 1

        if input_position == len(input_domain):
            raise ValidationError(
                "curve domains must be an ordered subset of input_x.domain"
            )

        curve_domain.append(input_domain[input_position])
        input_position += 1

    return tuple(curve_domain)


def _validate_parameter_pair(
    input_x: gp.Variable,
    x_points: object,
    y_points: object,
) -> tuple[gp.Parameter, gp.Parameter]:
    if not isinstance(x_points, gp.Parameter) or not isinstance(y_points, gp.Parameter):
        raise ValidationError(
            "x_points and y_points must both be Parameters, or both be lists or tuples"
        )

    for name, points in (("x_points", x_points), ("y_points", y_points)):
        if points.container is not input_x.container:
            raise ValidationError(f"{name} must belong to input_x.container")

    return x_points, y_points


def _normalize_parameter_row(
    x_values: Iterable[int | float | str],
    y_values: Iterable[int | float | str],
    *,
    key: tuple[str, ...],
    trim_trailing_padding: bool,
    bound_left: bool,
    bound_right: bool,
) -> _PwlRow:
    # Convert paired NA values to the sequence representation. Interior pairs
    # remain holes; only an indexed row's trailing suffix is trimmed below.
    normalized_x: list[object] = []
    normalized_y: list[object] = []
    for position, (x_value, y_value) in enumerate(zip(x_values, y_values, strict=True)):
        location = f"row {key}, position {position}" if key else f"position {position}"
        x_is_na = bool(gp.SpecialValues.isNA(x_value))
        y_is_na = bool(gp.SpecialValues.isNA(y_value))
        if x_is_na != y_is_na:
            raise ValidationError(f"{location}: x_points and y_points must both be NA")

        for name, value in (
            ("x_points", x_value),
            ("y_points", y_value),
        ):
            if bool(gp.SpecialValues.isUndef(value)):
                raise ValidationError(f"{name} at {location} must not be UNDF or NaN")

        if x_is_na:
            normalized_x.append(None)
            normalized_y.append(None)
        else:
            normalized_x.append(x_value)
            normalized_y.append(y_value)

    if trim_trailing_padding:
        while normalized_x and normalized_x[-1] is None:
            normalized_x.pop()
            normalized_y.pop()

        if not normalized_x:
            raise ValidationError(f"row {key} cannot contain only NA padding")

        if normalized_x[0] is None:
            raise ValidationError(f"row {key} cannot start with NA")

    try:
        return _normalize_sequence_data(
            normalized_x,
            normalized_y,
            bound_left=bound_left,
            bound_right=bound_right,
            row_keys=(key,),
        )[0]
    except ValidationError as error:
        if key:
            raise ValidationError(f"row {key}: {error}") from error
        raise


def _normalize_sequence_row(
    x_values: tuple[object, ...],
    y_values: tuple[object, ...],
    *,
    bound_left: bool,
    bound_right: bool,
) -> _PwlRow:
    if len(x_values) < 2:
        raise ValidationError("piecewise linear functions require at least 2 points")

    if len(x_values) != len(y_values):
        raise ValidationError("x_points and y_points have different lengths")

    for x_value, y_value in zip(x_values, y_values, strict=True):
        if (x_value is None) != (y_value is None):
            raise ValidationError(
                "x_points and y_points must both use None at the same position"
            )

    first = 0
    last = len(x_values)
    left_gradient = None
    right_gradient = None

    # Extreme infinity markers describe rays and do not enter the finite point
    # sequence.
    if _is_infinity(x_values[0], -1):
        if not bound_left:
            raise ValidationError(
                "an explicit left ray cannot be combined with bound_left=False"
            )

        left_gradient = _to_finite_number(
            y_values[0], "the explicit left ray gradient", 0
        )
        first += 1

    if _is_infinity(x_values[-1], 1):
        if not bound_right:
            raise ValidationError(
                "an explicit right ray cannot be combined with bound_right=False"
            )

        right_gradient = _to_finite_number(
            y_values[-1],
            "the explicit right ray gradient",
            len(y_values) - 1,
        )
        last -= 1

    if first == last:
        raise ValidationError("a ray requires a finite base point")

    finite_x = x_values[first:last]
    finite_y = y_values[first:last]
    if finite_x[0] is None or finite_x[-1] is None:
        raise ValidationError("point data cannot start or end with a None value")

    points: list[_PwlPoint] = []
    separator_indices: list[int] = []
    follows_hole = False

    for source_position, (x_value, y_value) in enumerate(
        zip(finite_x, finite_y, strict=True), start=first
    ):
        if x_value is None:
            if follows_hole:
                raise ValidationError(
                    "point data cannot contain two consecutive None values"
                )

            follows_hole = True
            continue

        point = _PwlPoint(
            x=_to_finite_number(x_value, "x_points", source_position),
            y=_to_finite_number(y_value, "y_points", source_position),
        )

        if points:
            previous = points[-1]
            # Holes, repeated x values, and x decreases split consecutive point
            # chains.
            if follows_hole:
                if point.x <= previous.x:
                    raise ValidationError(
                        "a point following None must have a greater x value than "
                        "the point preceding None"
                    )
                separator_indices.append(len(points) - 1)
            elif point.x <= previous.x:
                separator_indices.append(len(points) - 1)

        points.append(point)
        follows_hole = False

    if len(points) == 1 and left_gradient is None and right_gradient is None:
        raise ValidationError(
            "piecewise linear functions require at least 2 finite points"
        )

    segments = _build_segments(points, separator_indices)
    rays: list[_PwlRay] = []

    # Explicit markers supply their gradient. False bound flags retain the
    # historical endpoint-slope extension.
    if left_gradient is not None:
        rays.append(_PwlRay(points[0], left_gradient, -1))
    elif not bound_left:
        if len(points) < 2:
            raise ValidationError("bound_left=False requires at least 2 finite points")
        rays.append(
            _PwlRay(
                points[0],
                _get_legacy_gradient(points[0], points[1]),
                -1,
            )
        )

    if right_gradient is not None:
        rays.append(
            _PwlRay(
                points[-1],
                right_gradient,
                1,
            )
        )
    elif not bound_right:
        if len(points) < 2:
            raise ValidationError("bound_right=False requires at least 2 finite points")
        rays.append(
            _PwlRay(
                points[-1],
                _get_legacy_gradient(points[-2], points[-1]),
                1,
            )
        )

    return _PwlRow(
        key=(),
        ordered_points=tuple(points),
        separator_indices=tuple(separator_indices),
        segments=segments,
        rays=tuple(rays),
    )


def _to_point_sequence(points: object, name: str) -> tuple[object, ...]:
    if isinstance(points, (str, bytes)) or not isinstance(points, Sequence):
        raise ValidationError(f"{name} must be a list or tuple")

    return tuple(points)


def _to_finite_number(value: object, name: str, position: int) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValidationError(
            f"{name} at position {position} must be a finite real number"
        )

    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise ValidationError(
            f"{name} at position {position} must be a finite real number"
        ) from error

    if not math.isfinite(number):
        raise ValidationError(
            f"{name} at position {position} must be a finite real number"
        )

    return number


def _is_infinity(value: object, direction: Literal[-1, 1]) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False

    try:
        number = float(value)
    except (OverflowError, ValueError):
        return False

    return math.isinf(number) and (-1 if number < 0 else 1) == direction


def _build_segments(
    points: Sequence[_PwlPoint], separator_indices: Sequence[int]
) -> tuple[_PwlSegment, ...]:
    if len(points) < 2:
        return ()

    separators = set(separator_indices)
    segments: list[_PwlSegment] = []
    finished_at_separator = True
    for index in range(len(points) - 1):
        if index in separators:
            # A point isolated by separators still needs a selectable
            # zero-length segment.
            if finished_at_separator:
                segments.append(_PwlSegment(points[index], points[index]))
            finished_at_separator = True
        else:
            segments.append(_PwlSegment(points[index], points[index + 1]))
            finished_at_separator = False

    if finished_at_separator:
        segments.append(_PwlSegment(points[-1], points[-1]))

    return tuple(segments)


def _get_legacy_gradient(left: _PwlPoint, right: _PwlPoint) -> float:
    # Repeated extreme x values historically produce a zero-gradient ray.
    if left.x == right.x:
        return 0.0

    return (right.y - left.y) / (right.x - left.x)
