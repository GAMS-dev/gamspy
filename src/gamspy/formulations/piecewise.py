from __future__ import annotations

import math
import typing
import warnings

import numpy as np

import gamspy as gp
from gamspy._symbols.implicits import (
    ImplicitVariable,
)
from gamspy.exceptions import ValidationError
from gamspy.formulations._piecewise_data import (
    _classify_multivalued_rows,
    _normalize_paired_data,
    _PwlData,
    _PwlRow,
)
from gamspy.formulations.pwl_curve import PWLCurve, _normalize_curve_data

if typing.TYPE_CHECKING:
    from collections.abc import Mapping


def _validate_multivalued_data(
    data: _PwlData,
    allow_multivalued: bool,
    formulation: typing.Literal["interval", "convexity", "dlog"],
) -> None:
    multivalued = _classify_multivalued_rows(data.rows)

    if multivalued.overlaps:
        overlap = multivalued.overlaps[0]
        row_location = _format_curve_location(data, overlap.row_key)
        x_lower = f"{overlap.x_lower:g}"
        x_upper = f"{overlap.x_upper:g}"
        if formulation == "convexity":
            raise ValidationError(
                "Convexity formulation cannot represent overlapping independent "
                f"chains{row_location} from x={x_lower} to x={x_upper}. Use the "
                "Interval or DLog formulation with allow_multivalued=True instead."
            )
        if not allow_multivalued:
            raise ValidationError(
                f"The curve{row_location} contains independent chains whose x-ranges "
                f"overlap from {x_lower} to {x_upper}. This allows multiple y values "
                "for the same x. Set allow_multivalued=True to allow overlapping "
                "chains."
            )

    if multivalued.discontinuities and not allow_multivalued:
        discontinuity = multivalued.discontinuities[0]
        row_location = _format_curve_location(data, discontinuity.row_key)
        y_values = " or ".join(f"{value:g}" for value in discontinuity.y_values)
        message = (
            f"The curve{row_location} is multivalued at x={discontinuity.x:g}: y can "
            f"be {y_values}. Set allow_multivalued=True to acknowledge this behavior "
            "and suppress this warning."
        )
        additional = len(multivalued.discontinuities) - 1
        if additional:
            noun = "discontinuity" if additional == 1 else "discontinuities"
            message += f" The curve has {additional} additional {noun}."
        warnings.warn(message, UserWarning, stacklevel=3)


def _format_curve_location(data: _PwlData, row_key: tuple[str, ...]) -> str:
    if not row_key:
        return ""

    labels = ", ".join(
        f"{domain.name}={label!r}"
        for domain, label in zip(data.curve_domain, row_key, strict=True)
    )
    return f" for {labels}"


def _generate_gray_code(n: int, n_bits: int) -> np.ndarray:
    """
    Returns an n x n_bits NumPy array containing gray codes.
    The bit difference between two consecutive rows is exactly
    1 bits. Required for the log piecewise linear formulation.
    """
    a = np.arange(n)
    b = a >> 1
    numbers = a ^ b
    numbers_in_bit_array = ((numbers[:, None] & (1 << np.arange(n_bits))) > 0).astype(
        int
    )
    return numbers_in_bit_array


def _enforce_sos2_with_binary(lambda_var: gp.Variable) -> list[gp.Equation]:
    """
    Enforces SOS2 constraints using binary variables. This function is not suitable
    for generic SOS2 implementation since it restricts the lambda_var values to be
    between 0 and 1. However, it is usually faster than using SOS2 variables.

    Based on paper:
    `Modeling disjunctive constraints with a logarithmic number of binary variables and constraints
    <https://link.springer.com/article/10.1007/s10107-009-0295-4>`_
    """
    equations: list[gp.Equation] = []
    m = lambda_var.container
    count_x = len(lambda_var.domain[-1])
    # edge case
    lambda_var.lo[...] = 0
    lambda_var.up[...] = 1
    if count_x == 2:
        # if there are only 2 elements, it is already sos2
        return equations

    J = lambda_var.domain[-1]
    previous_domains = lambda_var.domain[:-1]

    l_len = math.ceil(math.log2(count_x - 1))
    I, L = gp.math._generate_dims(
        m,
        [
            count_x - 1,
            l_len,
        ],
    )

    J, I, L = gp.formulations.utils._next_domains([J, I, L], previous_domains)
    bin_var = m.addVariable(domain=[*previous_domains, L], type="binary")
    gray_code = _generate_gray_code(count_x - 1, l_len)

    B = m.addParameter(domain=[I, L], records=gray_code)

    JI = m.addSet(domain=[J, I])
    JI[J, I].where[(gp.Ord(J) == gp.Ord(I)) | (gp.Ord(J) - 1 == gp.Ord(I))] = 1

    use_set_1 = m.addSet(domain=[L, J])
    use_set_1[L, J].where[gp.Smin(JI[J, I], B[I, L]) == 1] = 1

    use_set_2 = m.addSet(domain=[L, J])
    use_set_2[L, J].where[gp.Smax(JI[J, I], B[I, L]) == 0] = 1

    sos2_eq_1 = m.addEquation(domain=[*previous_domains, L])
    sos2_eq_1[[*previous_domains, L]] = (
        gp.Sum(use_set_1[L, J], lambda_var[[*previous_domains, J]])
        <= bin_var[[*previous_domains, L]]
    )
    equations.append(sos2_eq_1)

    sos2_eq_2 = m.addEquation(domain=[*previous_domains, L])
    sos2_eq_2[[*previous_domains, L]] = (
        gp.Sum(use_set_2[L, J], lambda_var[[*previous_domains, J]])
        <= 1 - bin_var[[*previous_domains, L]]
    )
    equations.append(sos2_eq_2)

    return equations


def _enforce_discontinuity(
    lambda_var: gp.Variable,
    combined_indices: typing.Sequence[int],
) -> list[gp.Equation]:
    equations: list[gp.Equation] = []

    len_x_points = len(lambda_var.domain[-1])
    previous_domains = lambda_var.domain[:-1]

    m = lambda_var.container
    J, J2, SB = gp.math._generate_dims(
        m, [len_x_points, len_x_points, len(combined_indices)]
    )

    J, J2, SB = gp.formulations.utils._next_domains([J, J2, SB], previous_domains)

    # Each selector prevents the lambda values on both sides of a separator
    # from being positive together.
    di_param = [(str(i), str(j), str(j + 1)) for i, j in enumerate(combined_indices)]

    select_set = m.addSet(domain=[SB, J, J2], records=di_param)
    select_var = m.addVariable(domain=[*previous_domains, SB], type="binary")

    select_equation = m.addEquation(domain=[*previous_domains, SB, J, J2])
    select_equation[[*previous_domains, select_set[SB, J, J2]]] = (
        lambda_var[[*previous_domains, J]] <= select_var[[*previous_domains, SB]]
    )
    equations.append(select_equation)

    select_equation_2 = m.addEquation(domain=[*previous_domains, SB, J, J2])
    select_equation_2[[*previous_domains, select_set[SB, J, J2]]] = (
        lambda_var[[*previous_domains, J2]] <= 1 - select_var[[*previous_domains, SB]]
    )
    equations.append(select_equation_2)

    return equations


def _enforce_indexed_discontinuity(
    lambda_var: gp.Variable,
    rows: tuple[_PwlRow, ...],
    curve_domain: tuple[gp.Set | gp.Alias, ...],
) -> list[gp.Equation]:
    max_separators = max(len(row.separator_indices) for row in rows)
    if not max_separators:
        return []

    m = lambda_var.container
    previous_domains = typing.cast("list[gp.Set]", list(lambda_var.domain[:-1]))
    curve_index = [*curve_domain]
    J = typing.cast("gp.Set", lambda_var.domain[-1])
    # Separator counts differ by row, so use rectangular slots and mask the
    # unused tail.
    J2, SB = gp.math._generate_dims(m, [len(J), max_separators])
    J2, SB = gp.formulations.utils._next_domains([J2, SB], [*previous_domains, J])

    # Map each active (row, separator) pair to its two boundary lambda slots.
    active_records = []
    separator_records = []
    for row in rows:
        for slot, separator_index in enumerate(row.separator_indices):
            active_records.append((*row.key, str(slot)))
            separator_records.append(
                (
                    *row.key,
                    str(slot),
                    str(separator_index),
                    str(separator_index + 1),
                )
            )

    active_separators = m.addSet(domain=[*curve_index, SB], records=active_records)
    separator_map = m.addSet(
        domain=[*curve_index, SB, J, J2],
        records=separator_records,
    )
    select_var = m.addVariable(domain=[*previous_domains, SB], type="binary")
    active_index = [*previous_domains, SB]
    active_data_index = [*curve_index, SB]
    separator_index = [*previous_domains, SB, J, J2]
    separator_data_index = [*curve_index, SB, J, J2]
    # Inactive selectors must not introduce free binary decisions.
    select_var.fx[active_index].where[~active_separators[active_data_index]] = 0

    select_equation = m.addEquation(domain=[*previous_domains, SB, J, J2])
    select_equation[separator_index].where[separator_map[separator_data_index]] = (
        lambda_var[[*previous_domains, J]] <= select_var[active_index]
    )

    select_equation_2 = m.addEquation(domain=[*previous_domains, SB, J, J2])
    select_equation_2[separator_index].where[separator_map[separator_data_index]] = (
        lambda_var[[*previous_domains, J2]] <= 1 - select_var[active_index]
    )

    return [select_equation, select_equation_2]


def _indicator(
    indicator_var: gp.Variable,
    indicator_val: typing.Literal[0, 1],
    expr: gp.Expression,
) -> list[gp.Equation]:
    # We will make this generic and public
    if not isinstance(indicator_var, (gp.Variable, ImplicitVariable)):
        raise ValidationError("indicator_var needs to be a variable")

    if indicator_var.type != "binary":
        raise ValidationError("indicator_var needs to be a binary variable")

    if indicator_val not in (0, 1):
        raise ValidationError("indicator_val needs to be 1 or 0")

    if not isinstance(expr, gp.Expression):
        raise ValidationError("expr needs to be an expression")

    if expr.operator not in {"=l=", "=e=", "=g="}:
        raise ValidationError("expr needs to be inequality or equality")

    if len(expr.domain) != len(indicator_var.domain):
        raise ValidationError("indicator_var and expr must have the same domain")

    for i in range(len(expr.domain)):
        if expr.domain[i].name != indicator_var.domain[i].name:
            raise ValidationError("indicator_var and expr must have the same domain")

    if expr.operator == "=e=":
        # sos1(bin_var, lhs - rhs) might be better
        eqs1 = _indicator(
            indicator_var,
            indicator_val,
            expr.left <= expr.right,  # ty: ignore[unsupported-operator]
        )
        eqs2 = _indicator(
            indicator_var,
            indicator_val,
            -expr.left <= -expr.right,  # ty: ignore[invalid-argument-type, unsupported-operator]
        )
        return [*eqs1, *eqs2]

    if expr.operator == "=g=":
        return _indicator(
            indicator_var,
            indicator_val,
            -expr.left <= -expr.right,  # ty: ignore[invalid-argument-type, unsupported-operator]
        )

    equations = []
    m = indicator_var.container

    slack_var = m.addVariable(domain=expr.domain, type="positive")
    slack_eq = m.addEquation(
        domain=expr.domain, definition=(expr.left - slack_var <= expr.right)
    )
    equations.append(slack_eq)

    expr_domain = ... if len(expr.domain) == 0 else [*expr.domain]

    sos_dim = gp.math._generate_dims(m, [2])[0]
    sos1_var = m.addVariable(domain=[*expr.domain, sos_dim], type="sos1")
    sos1_eq_1 = m.addEquation(domain=expr.domain)
    if indicator_val == 1:
        sos1_eq_1[...] = sos1_var[[*expr.domain, "0"]] == indicator_var[expr_domain]
    else:
        sos1_eq_1[...] = sos1_var[[*expr.domain, "0"]] == 1 - indicator_var[expr_domain]
    equations.append(sos1_eq_1)

    sos1_eq_2 = m.addEquation(domain=expr.domain)
    sos1_eq_2[...] = sos1_var[[*expr.domain, "1"]] == slack_var[expr_domain]
    equations.append(sos1_eq_2)

    return equations


def _generate_ray(
    container: gp.Container, domain: typing.Sequence[gp.Set | gp.Alias]
) -> tuple[gp.Variable, gp.Variable, list[gp.Equation]]:
    # if b_var == 0 => x_var = 0 o.w x_var >= 0
    # effectively x_var <= bigM * b_var without bigM
    x_var = container.addVariable(domain=domain, type="positive")
    b_var = container.addVariable(domain=domain, type="binary")
    eqs = _indicator(b_var, 0, x_var <= 0)
    return x_var, b_var, eqs


def _get_interval_segment_data(
    row: _PwlRow,
) -> list[tuple[str, float, float, float, float]]:
    segment_data = []
    for index, segment in enumerate(row.segments):
        if segment.left.x == segment.right.x:
            slope = 0.0
            offset = segment.left.y
        else:
            slope = (segment.right.y - segment.left.y) / (
                segment.right.x - segment.left.x
            )
            offset = segment.left.y - slope * segment.left.x

        segment_data.append(
            (
                str(index),
                segment.left.x,
                segment.right.x,
                slope,
                offset,
            )
        )

    return segment_data


def _get_dlog_segment_data(
    row: _PwlRow,
) -> list[tuple[str, float, float, float, float]]:
    return [
        (
            str(index),
            segment.left.x,
            segment.right.x,
            segment.left.y,
            segment.right.y,
        )
        for index, segment in enumerate(row.segments)
    ]


def _set_pwl_output_bounds(
    out_y: gp.Variable,
    rows: tuple[_PwlRow, ...],
    *,
    curve_domain: tuple[gp.Set | gp.Alias, ...],
) -> None:
    row_bounds: list[tuple[_PwlRow, float | None, float | None]] = []
    for row in rows:
        lower_bound: float | None = min(point.y for point in row.ordered_points)
        upper_bound: float | None = max(point.y for point in row.ordered_points)
        # A ray removes the y bound approached by its signed gradient.
        for ray in row.rays:
            y_direction = ray.direction * ray.gradient
            if y_direction < 0:
                lower_bound = None
            elif y_direction > 0:
                upper_bound = None
        row_bounds.append((row, lower_bound, upper_bound))

    if not curve_domain:
        _, lower_bound, upper_bound = row_bounds[0]
        if lower_bound is not None:
            out_y.lo[...] = lower_bound
        if upper_bound is not None:
            out_y.up[...] = upper_bound
        return

    input_domain = typing.cast("list[gp.Set]", list(out_y.domain))
    curve_index = [*curve_domain]
    # Rays can remove only one side, so lower and upper bounds need separate
    # row masks.
    lower_bounds = [
        (row, lower_bound)
        for row, lower_bound, _ in row_bounds
        if lower_bound is not None
    ]
    if lower_bounds:
        lower_bounded_rows = out_y.container.addSet(
            domain=curve_index,
            records=[row.key for row, _ in lower_bounds],
        )
        lower_y = out_y.container.addParameter(
            domain=curve_index,
            records=[(*row.key, lower_bound) for row, lower_bound in lower_bounds],
        )
        out_y.lo[input_domain].where[lower_bounded_rows[curve_index]] = lower_y[
            curve_index
        ]

    upper_bounds = [
        (row, upper_bound)
        for row, _, upper_bound in row_bounds
        if upper_bound is not None
    ]
    if upper_bounds:
        upper_bounded_rows = out_y.container.addSet(
            domain=curve_index,
            records=[row.key for row, _ in upper_bounds],
        )
        upper_y = out_y.container.addParameter(
            domain=curve_index,
            records=[(*row.key, upper_bound) for row, upper_bound in upper_bounds],
        )
        out_y.up[input_domain].where[upper_bounded_rows[curve_index]] = upper_y[
            curve_index
        ]


def _build_scalar_interval_formulation(
    input_x: gp.Variable,
    row: _PwlRow,
) -> tuple[gp.Variable, list[gp.Equation]]:
    m = input_x.container
    equations: list[gp.Equation] = []
    segment_selector_term = gp.Number(0)
    x_term = gp.Number(0)
    y_term = gp.Number(0)

    if row.segments:
        segment_data = _get_interval_segment_data(row)

        J = gp.math._generate_dims(m, [len(row.segments)])[0]
        lowerbounds = m.addParameter(
            domain=J, records=[(key, lower) for key, lower, _, _, _ in segment_data]
        )
        upperbounds = m.addParameter(
            domain=J, records=[(key, upper) for key, _, upper, _, _ in segment_data]
        )
        slopes = m.addParameter(
            domain=J, records=[(key, slope) for key, _, _, slope, _ in segment_data]
        )
        offsets = m.addParameter(
            domain=J, records=[(key, offset) for key, _, _, _, offset in segment_data]
        )
        bin_var = m.addVariable(domain=J, type="binary")
        segment_x = m.addVariable(domain=J)

        set_segment_lowerbound = m.addEquation(domain=J)
        set_segment_lowerbound[J] = lowerbounds * bin_var <= segment_x
        equations.append(set_segment_lowerbound)

        set_segment_upperbound = m.addEquation(domain=J)
        set_segment_upperbound[J] = upperbounds * bin_var >= segment_x
        equations.append(set_segment_upperbound)

        segment_selector_term = gp.Sum(J, bin_var)
        x_term = gp.Sum(J, segment_x)
        y_term = gp.Sum(J, segment_x * slopes) + gp.Sum(J, bin_var * offsets)

    out_y = m.addVariable()

    _set_pwl_output_bounds(
        out_y,
        (row,),
        curve_domain=(),
    )

    ray_selector_term = gp.Number(0)
    for ray in row.rays:
        movement, active, ray_equations = _generate_ray(m, [])
        equations.extend(ray_equations)
        ray_selector_term += active
        x_term += active * ray.base.x + ray.direction * movement
        y_term += active * ray.base.y + ray.direction * ray.gradient * movement

    pick_one = m.addEquation()
    pick_one[...] = segment_selector_term + ray_selector_term == 1
    equations.append(pick_one)

    set_x = m.addEquation()
    set_x[...] = input_x == x_term
    equations.append(set_x)

    set_y = m.addEquation()
    set_y[...] = out_y == y_term
    equations.append(set_y)

    return out_y, equations


def _build_indexed_interval_formulation(
    input_x: gp.Variable,
    data: _PwlData,
) -> tuple[gp.Variable, list[gp.Equation]]:
    m = input_x.container
    rows = data.rows
    input_domain = typing.cast("list[gp.Set]", list(input_x.domain))
    input_index = [*input_domain]
    curve_domain = [*data.curve_domain]
    equations: list[gp.Equation] = []
    segment_selector_term = gp.Number(0)
    x_term = gp.Number(0)
    y_term = gp.Number(0)

    # Formulation symbols are rectangular; active_segments removes padded slots
    # for rows with fewer segments.
    max_segments = max(len(row.segments) for row in rows)
    if max_segments:
        J = gp.math._generate_dims(m, [max_segments])[0]
        J = gp.formulations.utils._next_domains([J], input_domain)[0]
        segment_domain = [*input_domain, J]
        # Curve data keeps only the domains that distinguish its rows and is
        # broadcast over omitted input domains.
        segment_data_domain = [*curve_domain, J]
        segment_index = [*input_domain, J]
        segment_data_index = [*curve_domain, J]

        active_records = []
        lowerbound_records = []
        upperbound_records = []
        slope_records = []
        offset_records = []
        for row in rows:
            prefix = row.key
            for slot, lower, upper, slope, offset in _get_interval_segment_data(row):
                active_records.append((*prefix, slot))
                lowerbound_records.append((*prefix, slot, lower))
                upperbound_records.append((*prefix, slot, upper))
                slope_records.append((*prefix, slot, slope))
                offset_records.append((*prefix, slot, offset))

        active_segments = m.addSet(domain=segment_data_domain, records=active_records)
        lowerbounds = m.addParameter(
            domain=segment_data_domain, records=lowerbound_records
        )
        upperbounds = m.addParameter(
            domain=segment_data_domain, records=upperbound_records
        )
        slopes = m.addParameter(domain=segment_data_domain, records=slope_records)
        offsets = m.addParameter(domain=segment_data_domain, records=offset_records)
        bin_var = m.addVariable(domain=segment_domain, type="binary")
        segment_x = m.addVariable(domain=segment_domain)

        # Fix unused rectangular slots so they cannot affect the model.
        bin_var.fx[segment_index].where[~active_segments[segment_data_index]] = 0
        segment_x.fx[segment_index].where[~active_segments[segment_data_index]] = 0

        set_segment_lowerbound = m.addEquation(domain=segment_domain)
        set_segment_lowerbound[segment_index].where[
            active_segments[segment_data_index]
        ] = (
            lowerbounds[segment_data_index] * bin_var[segment_index]
            <= segment_x[segment_index]
        )
        equations.append(set_segment_lowerbound)

        set_segment_upperbound = m.addEquation(domain=segment_domain)
        set_segment_upperbound[segment_index].where[
            active_segments[segment_data_index]
        ] = (
            upperbounds[segment_data_index] * bin_var[segment_index]
            >= segment_x[segment_index]
        )
        equations.append(set_segment_upperbound)

        segment_selector_term = gp.Sum(
            J,
            active_segments[segment_data_index] * bin_var[segment_index],
        )
        x_term = gp.Sum(
            J,
            active_segments[segment_data_index] * segment_x[segment_index],
        )
        y_term = gp.Sum(
            J,
            active_segments[segment_data_index]
            * (
                segment_x[segment_index] * slopes[segment_data_index]
                + bin_var[segment_index] * offsets[segment_data_index]
            ),
        )

    out_y = m.addVariable(domain=input_domain)
    _set_pwl_output_bounds(
        out_y,
        rows,
        curve_domain=data.curve_domain,
    )

    ray_selector_term = gp.Number(0)
    # Rays use the same rectangular-and-mask layout as finite segments.
    max_rays = max(len(row.rays) for row in rows)
    if max_rays:
        R = gp.math._generate_dims(m, [max_rays])[0]
        R = gp.formulations.utils._next_domains([R], input_domain)[0]
        ray_domain = [*input_domain, R]
        ray_data_domain = [*curve_domain, R]
        ray_index = [*input_domain, R]
        ray_data_index = [*curve_domain, R]

        active_records = []
        base_x_records = []
        base_y_records = []
        direction_records = []
        gradient_records = []
        for row in rows:
            prefix = row.key
            for slot, ray in enumerate(row.rays):
                slot_key = str(slot)
                active_records.append((*prefix, slot_key))
                base_x_records.append((*prefix, slot_key, ray.base.x))
                base_y_records.append((*prefix, slot_key, ray.base.y))
                direction_records.append((*prefix, slot_key, ray.direction))
                gradient_records.append((*prefix, slot_key, ray.gradient))

        active_rays = m.addSet(domain=ray_data_domain, records=active_records)
        base_x = m.addParameter(domain=ray_data_domain, records=base_x_records)
        base_y = m.addParameter(domain=ray_data_domain, records=base_y_records)
        direction = m.addParameter(domain=ray_data_domain, records=direction_records)
        gradient = m.addParameter(domain=ray_data_domain, records=gradient_records)
        movement, ray_selector, ray_equations = _generate_ray(m, ray_domain)
        equations.extend(ray_equations)

        movement.fx[ray_index].where[~active_rays[ray_data_index]] = 0
        ray_selector.fx[ray_index].where[~active_rays[ray_data_index]] = 0

        ray_selector_term = gp.Sum(
            R,
            active_rays[ray_data_index] * ray_selector[ray_index],
        )
        x_term += gp.Sum(
            R,
            active_rays[ray_data_index]
            * (
                ray_selector[ray_index] * base_x[ray_data_index]
                + direction[ray_data_index] * movement[ray_index]
            ),
        )
        y_term += gp.Sum(
            R,
            active_rays[ray_data_index]
            * (
                ray_selector[ray_index] * base_y[ray_data_index]
                + direction[ray_data_index]
                * gradient[ray_data_index]
                * movement[ray_index]
            ),
        )

    pick_one = m.addEquation(domain=input_domain)
    pick_one[input_index] = segment_selector_term + ray_selector_term == 1
    equations.append(pick_one)

    set_x = m.addEquation(domain=input_domain)
    set_x[input_index] = input_x[input_index] == x_term
    equations.append(set_x)

    set_y = m.addEquation(domain=input_domain)
    set_y[input_index] = out_y[input_index] == y_term
    equations.append(set_y)

    return out_y, equations


def _build_interval_formulation(
    input_x: gp.Variable,
    data: _PwlData,
) -> tuple[gp.Variable, list[gp.Equation]]:
    if input_x.dimension == 0:
        return _build_scalar_interval_formulation(input_x, data.rows[0])
    else:
        return _build_indexed_interval_formulation(input_x, data)


def pwl_interval_formulation(
    input_x: gp.Variable,
    x_points: typing.Sequence[int | float | None] | gp.Parameter,
    y_points: typing.Sequence[int | float | None] | gp.Parameter,
    *,
    bound_left: bool = True,
    bound_right: bool = True,
    allow_multivalued: bool = False,
) -> tuple[gp.Variable, list[gp.Equation]]:
    """
    Create a piecewise-linear relationship using the Interval formulation.

    This function implements a piecewise linear function using the intervals formulation.
    Given an input (independent) variable `input_x`, along with the defining `x_points`
    and corresponding `y_points` of the piecewise function, it constructs the dependent
    variable `y` and formulates the equations necessary to define the function.

    Here is the interval formulation:

    .. math::
        \\lambda_i \\geq b_i * LB_i \\quad \\forall{i}

        \\lambda_i \\leq b_i * UB_i \\quad \\forall{i}

        \\sum_{i}{b_i} = 1

        x = \\sum_{i}{\\lambda_i}

        y = \\sum_{i}{(\\lambda_i * slope_i) + (b_i * offset_i) }

        b_i \\in \\{0, 1\\} \\quad \\forall{i}

    Repeated x-coordinates describe discontinuities, matching `None` values
    exclude the range between two chains, and a decrease in x starts an
    independent chain that overlaps an earlier chain. Interval supports all
    three structures.

    Extreme `-math.inf` and `math.inf` values in `x_points` define explicit
    rays; the matching `y_points` values are their gradients. Otherwise,
    setting a bound argument to False extends the corresponding outer segment
    with its existing gradient.

    Parameters
    ----------
    input_x : gp.Variable
        Independent variable of the piecewise linear function
    x_points : typing.Sequence[int | float | None] | gp.Parameter
        Breakpoints of the piecewise linear function in the x-axis.
        An indexed Parameter may use an ordered subset of the input domains
        followed by one breakpoint domain. It is broadcast over omitted input
        domains.
    y_points : typing.Sequence[int | float | None] | gp.Parameter
        Breakpoints of the piecewise linear function in the y-axis. Parameter
        input must exactly match the container and domains of `x_points`.
    bound_left : bool = True
        Whether `input_x` is limited to start at the first finite point. If
        False, the first finite segment is extended to the left.
    bound_right : bool = True
        Whether `input_x` is limited to end at the last finite point. If False,
        the last finite segment is extended to the right.
    allow_multivalued : bool = False
        Whether to explicitly allow multivalued curves. If False, point
        discontinuities issue a warning and overlapping chains are rejected.

    Returns
    -------
    tuple[gp.Variable, list[Equation]]

    Examples
    --------
    >>> from gamspy import Container, Variable
    >>> from gamspy.formulations import pwl_interval_formulation
    >>> m = Container()
    >>> x = Variable(m, "x")
    >>> y, eqs = pwl_interval_formulation(
    ...     x,
    ...     [-1, 4, 10, 10, 20],
    ...     [-2, 8, 15, 17, 37],
    ...     allow_multivalued=True,
    ... )

    """

    if not isinstance(input_x, gp.Variable):
        raise ValidationError("input_x is expected to be a Variable")

    if not isinstance(bound_left, bool):
        raise ValidationError("bound_left is expected to be a boolean")

    if not isinstance(bound_right, bool):
        raise ValidationError("bound_right is expected to be a boolean")

    if not isinstance(allow_multivalued, bool):
        raise ValidationError("allow_multivalued is expected to be a boolean")

    data = _normalize_paired_data(
        input_x,
        x_points,
        y_points,
        bound_left=bound_left,
        bound_right=bound_right,
    )
    _validate_multivalued_data(data, allow_multivalued, "interval")

    return _build_interval_formulation(input_x, data)


def _build_scalar_dlog_formulation(
    input_x: gp.Variable,
    row: _PwlRow,
) -> tuple[gp.Variable, list[gp.Equation]]:
    m = input_x.container
    equations: list[gp.Equation] = []
    segment_count = len(row.segments)
    ray_count = len(row.rays)
    unit_count = segment_count + ray_count
    bit_count = math.ceil(math.log2(unit_count)) if unit_count > 1 else 0
    dimension_sizes = []
    if segment_count:
        dimension_sizes.append(segment_count)
    if ray_count:
        dimension_sizes.append(ray_count)
    if bit_count:
        dimension_sizes.append(bit_count)
    dimensions = gp.math._generate_dims(m, dimension_sizes)
    dimension_position = 0
    segment_weight_term = gp.Number(0)
    ray_weight_term = gp.Number(0)
    x_term = gp.Number(0)
    y_term = gp.Number(0)

    if segment_count:
        J = dimensions[dimension_position]
        dimension_position += 1
        segment_data = _get_dlog_segment_data(row)
        left_x = m.addParameter(
            domain=J,
            records=[(slot, value) for slot, value, _, _, _ in segment_data],
        )
        right_x = m.addParameter(
            domain=J,
            records=[(slot, value) for slot, _, value, _, _ in segment_data],
        )
        left_y = m.addParameter(
            domain=J,
            records=[(slot, value) for slot, _, _, value, _ in segment_data],
        )
        right_y = m.addParameter(
            domain=J,
            records=[(slot, value) for slot, _, _, _, value in segment_data],
        )
        gamma_left = m.addVariable(domain=J, type="positive")
        gamma_right = m.addVariable(domain=J, type="positive")

        segment_weight_term = gp.Sum(J, gamma_left[J] + gamma_right[J])
        x_term = gp.Sum(
            J,
            gamma_left[J] * left_x[J] + gamma_right[J] * right_x[J],
        )
        y_term = gp.Sum(
            J,
            gamma_left[J] * left_y[J] + gamma_right[J] * right_y[J],
        )

    if ray_count:
        R = dimensions[dimension_position]
        dimension_position += 1
        base_x = m.addParameter(
            domain=R,
            records=[(str(slot), ray.base.x) for slot, ray in enumerate(row.rays)],
        )
        base_y = m.addParameter(
            domain=R,
            records=[(str(slot), ray.base.y) for slot, ray in enumerate(row.rays)],
        )
        direction = m.addParameter(
            domain=R,
            records=[(str(slot), ray.direction) for slot, ray in enumerate(row.rays)],
        )
        gradient = m.addParameter(
            domain=R,
            records=[(str(slot), ray.gradient) for slot, ray in enumerate(row.rays)],
        )
        movement, ray_weight, ray_equations = _generate_ray(m, [R])
        equations.extend(ray_equations)

        ray_weight_term = gp.Sum(R, ray_weight[R])
        x_term += gp.Sum(
            R,
            ray_weight[R] * base_x[R] + direction[R] * movement[R],
        )
        y_term += gp.Sum(
            R,
            ray_weight[R] * base_y[R] + direction[R] * gradient[R] * movement[R],
        )

    out_y = m.addVariable()
    _set_pwl_output_bounds(
        out_y,
        (row,),
        curve_domain=(),
    )

    pick_one = m.addEquation()
    pick_one[...] = segment_weight_term + ray_weight_term == 1
    equations.append(pick_one)

    set_x = m.addEquation()
    set_x[...] = input_x == x_term
    equations.append(set_x)

    set_y = m.addEquation()
    set_y[...] = out_y == y_term
    equations.append(set_y)

    if bit_count:
        L = dimensions[dimension_position]
        gray_code = _generate_gray_code(unit_count, bit_count)
        address_term = gp.Number(0)
        if segment_count:
            segment_addresses = m.addParameter(
                domain=[J, L],
                records=gray_code[:segment_count],
            )
            address_term += gp.Sum(
                J,
                (gamma_left[J] + gamma_right[J]) * segment_addresses[J, L],
            )
        if ray_count:
            ray_addresses = m.addParameter(
                domain=[R, L],
                records=gray_code[segment_count:],
            )
            address_term += gp.Sum(R, ray_weight[R] * ray_addresses[R, L])

        select_bit = m.addVariable(domain=L, type="binary")
        # A binary weighted address prevents endpoint weights from different
        # selectable units from being positive together.
        select_address = m.addEquation(domain=L)
        select_address[L] = address_term == select_bit[L]
        equations.append(select_address)

    return out_y, equations


def _build_indexed_dlog_formulation(
    input_x: gp.Variable,
    data: _PwlData,
) -> tuple[gp.Variable, list[gp.Equation]]:
    m = input_x.container
    rows = data.rows
    input_domain = typing.cast("list[gp.Set]", list(input_x.domain))
    input_index = [*input_domain]
    curve_domain = [*data.curve_domain]
    equations: list[gp.Equation] = []

    max_segments = max(len(row.segments) for row in rows)
    max_rays = max(len(row.rays) for row in rows)
    max_units = max(len(row.segments) + len(row.rays) for row in rows)
    bit_count = math.ceil(math.log2(max_units)) if max_units > 1 else 0
    dimension_sizes = []
    if max_segments:
        dimension_sizes.append(max_segments)
    if max_rays:
        dimension_sizes.append(max_rays)
    if bit_count:
        dimension_sizes.append(bit_count)
    dimensions = gp.formulations.utils._next_domains(
        gp.math._generate_dims(m, dimension_sizes),
        input_domain,
    )
    dimension_position = 0
    segment_weight_term = gp.Number(0)
    ray_weight_term = gp.Number(0)
    x_term = gp.Number(0)
    y_term = gp.Number(0)

    if max_segments:
        J = dimensions[dimension_position]
        dimension_position += 1
        segment_domain = [*input_domain, J]
        # Data symbols omit broadcast dimensions, while formulation variables
        # keep the complete input domain.
        segment_data_domain = [*curve_domain, J]
        segment_index = [*input_domain, J]
        segment_data_index = [*curve_domain, J]

        active_records = []
        left_x_records = []
        right_x_records = []
        left_y_records = []
        right_y_records = []
        for row in rows:
            prefix = row.key
            for slot, left_x, right_x, left_y, right_y in _get_dlog_segment_data(row):
                active_records.append((*prefix, slot))
                left_x_records.append((*prefix, slot, left_x))
                right_x_records.append((*prefix, slot, right_x))
                left_y_records.append((*prefix, slot, left_y))
                right_y_records.append((*prefix, slot, right_y))

        active_segments = m.addSet(domain=segment_data_domain, records=active_records)
        left_x = m.addParameter(domain=segment_data_domain, records=left_x_records)
        right_x = m.addParameter(domain=segment_data_domain, records=right_x_records)
        left_y = m.addParameter(domain=segment_data_domain, records=left_y_records)
        right_y = m.addParameter(domain=segment_data_domain, records=right_y_records)
        gamma_left = m.addVariable(domain=segment_domain, type="positive")
        gamma_right = m.addVariable(domain=segment_domain, type="positive")

        # Shorter rows occupy a prefix of the rectangular segment dimension.
        gamma_left.fx[segment_index].where[~active_segments[segment_data_index]] = 0
        gamma_right.fx[segment_index].where[~active_segments[segment_data_index]] = 0

        segment_weight_term = gp.Sum(
            J,
            active_segments[segment_data_index]
            * (gamma_left[segment_index] + gamma_right[segment_index]),
        )
        x_term = gp.Sum(
            J,
            active_segments[segment_data_index]
            * (
                gamma_left[segment_index] * left_x[segment_data_index]
                + gamma_right[segment_index] * right_x[segment_data_index]
            ),
        )
        y_term = gp.Sum(
            J,
            active_segments[segment_data_index]
            * (
                gamma_left[segment_index] * left_y[segment_data_index]
                + gamma_right[segment_index] * right_y[segment_data_index]
            ),
        )

    if max_rays:
        R = dimensions[dimension_position]
        dimension_position += 1
        ray_domain = [*input_domain, R]
        ray_data_domain = [*curve_domain, R]
        ray_index = [*input_domain, R]
        ray_data_index = [*curve_domain, R]

        active_records = []
        base_x_records = []
        base_y_records = []
        direction_records = []
        gradient_records = []
        for row in rows:
            prefix = row.key
            for slot, ray in enumerate(row.rays):
                slot_key = str(slot)
                active_records.append((*prefix, slot_key))
                base_x_records.append((*prefix, slot_key, ray.base.x))
                base_y_records.append((*prefix, slot_key, ray.base.y))
                direction_records.append((*prefix, slot_key, ray.direction))
                gradient_records.append((*prefix, slot_key, ray.gradient))

        active_rays = m.addSet(domain=ray_data_domain, records=active_records)
        base_x = m.addParameter(domain=ray_data_domain, records=base_x_records)
        base_y = m.addParameter(domain=ray_data_domain, records=base_y_records)
        direction = m.addParameter(domain=ray_data_domain, records=direction_records)
        gradient = m.addParameter(domain=ray_data_domain, records=gradient_records)
        movement, ray_weight, ray_equations = _generate_ray(m, ray_domain)
        equations.extend(ray_equations)

        movement.fx[ray_index].where[~active_rays[ray_data_index]] = 0
        ray_weight.fx[ray_index].where[~active_rays[ray_data_index]] = 0

        ray_weight_term = gp.Sum(
            R,
            active_rays[ray_data_index] * ray_weight[ray_index],
        )
        x_term += gp.Sum(
            R,
            active_rays[ray_data_index]
            * (
                ray_weight[ray_index] * base_x[ray_data_index]
                + direction[ray_data_index] * movement[ray_index]
            ),
        )
        y_term += gp.Sum(
            R,
            active_rays[ray_data_index]
            * (
                ray_weight[ray_index] * base_y[ray_data_index]
                + direction[ray_data_index]
                * gradient[ray_data_index]
                * movement[ray_index]
            ),
        )

    out_y = m.addVariable(domain=input_domain)
    _set_pwl_output_bounds(
        out_y,
        rows,
        curve_domain=data.curve_domain,
    )

    pick_one = m.addEquation(domain=input_domain)
    pick_one[input_index] = segment_weight_term + ray_weight_term == 1
    equations.append(pick_one)

    set_x = m.addEquation(domain=input_domain)
    set_x[input_index] = input_x[input_index] == x_term
    equations.append(set_x)

    set_y = m.addEquation(domain=input_domain)
    set_y[input_index] = out_y[input_index] == y_term
    equations.append(set_y)

    if bit_count:
        L = dimensions[dimension_position]
        address_term = gp.Number(0)
        segment_address_records = []
        ray_address_records = []
        # Segments receive the first row-local addresses and rays continue the
        # same sequence. Padded rectangular slots keep their default zero data.
        for row in rows:
            prefix = row.key
            gray_code = _generate_gray_code(
                len(row.segments) + len(row.rays), bit_count
            )
            for segment_slot, address in enumerate(gray_code[: len(row.segments)]):
                for bit_slot, value in enumerate(address):
                    segment_address_records.append(
                        (*prefix, str(segment_slot), str(bit_slot), int(value))
                    )
            for ray_slot, address in enumerate(gray_code[len(row.segments) :]):
                for bit_slot, value in enumerate(address):
                    ray_address_records.append(
                        (*prefix, str(ray_slot), str(bit_slot), int(value))
                    )

        if max_segments:
            segment_address_domain = [*curve_domain, J, L]
            segment_address_index = [*curve_domain, J, L]
            segment_addresses = m.addParameter(
                domain=segment_address_domain,
                records=segment_address_records,
            )
            address_term += gp.Sum(
                J,
                active_segments[segment_data_index]
                * (gamma_left[segment_index] + gamma_right[segment_index])
                * segment_addresses[segment_address_index],
            )

        if max_rays:
            ray_address_domain = [*curve_domain, R, L]
            ray_address_index = [*curve_domain, R, L]
            ray_addresses = m.addParameter(
                domain=ray_address_domain,
                records=ray_address_records,
            )
            address_term += gp.Sum(
                R,
                active_rays[ray_data_index]
                * ray_weight[ray_index]
                * ray_addresses[ray_address_index],
            )

        bit_domain = [*input_domain, L]
        bit_index = [*input_domain, L]
        select_bit = m.addVariable(domain=bit_domain, type="binary")
        select_address = m.addEquation(domain=bit_domain)
        select_address[bit_index] = address_term == select_bit[bit_index]
        equations.append(select_address)

    return out_y, equations


def _build_dlog_formulation(
    input_x: gp.Variable,
    data: _PwlData,
) -> tuple[gp.Variable, list[gp.Equation]]:
    if input_x.dimension == 0:
        return _build_scalar_dlog_formulation(input_x, data.rows[0])
    else:
        return _build_indexed_dlog_formulation(input_x, data)


def pwl_dlog_formulation(
    input_x: gp.Variable,
    x_points: typing.Sequence[int | float | None] | gp.Parameter,
    y_points: typing.Sequence[int | float | None] | gp.Parameter,
    *,
    bound_left: bool = True,
    bound_right: bool = True,
    allow_multivalued: bool = False,
) -> tuple[gp.Variable, list[gp.Equation]]:
    """
    This function implements a piecewise linear function using the
    disaggregated logarithmic formulation. Every finite segment receives two
    endpoint weights, while a logarithmic number of binary variables selects
    the active segment. Sequence data and one-dimensional Parameters are
    broadcast over an indexed input. Indexed Parameters may select an ordered
    subset of the input domains and are broadcast over the omitted domains.
    Infinity markers define explicit rays, and false bound arguments extend the
    corresponding outer segment.

    A repeated x coordinate represents a discontinuity. Matching `None` values
    in the point sequences create an excluded range, and a decrease in x starts
    an independent chain that overlaps the earlier chain. DLog selects these
    finite segments independently, so it supports all three cases.

    For a scalar or shared graph with `k` selectable units, including finite
    segments and rays, the formulation creates `ceil(log2(k))` binary address
    variables. Indexed row-specific graphs use the largest unit count among
    their rows. A graph with one unit does not require address variables. Rays
    additionally use a binary unit weight to gate their unbounded movement.

    Returns the dependent variable `y` and the equations required to model the
    piecewise linear relationship.

    Parameters
    ----------
    input_x : gp.Variable
        Independent variable of the piecewise linear function. Sequence data
        is broadcast when this Variable is indexed. Indexed inputs must use
        non-empty explicit domain Sets.
    x_points : typing.Sequence[int | float | None] | gp.Parameter
        Breakpoints of the piecewise linear function on the x-axis. An indexed
        Parameter may use an ordered subset of the input domains followed by
        one breakpoint domain. It is broadcast over omitted input domains.
    y_points : typing.Sequence[int | float | None] | gp.Parameter
        Breakpoints of the piecewise linear function on the y-axis. Parameter
        input must exactly match the container and domains of `x_points`.
    bound_left : bool = True
        Whether input_x should be limited to start from the first finite point.
        If False, the first finite segment is extended to the left.
    bound_right : bool = True
        Whether input_x should be limited to end at the last finite point. If
        False, the last finite segment is extended to the right.
    allow_multivalued : bool = False
        Whether to explicitly allow multivalued curves. If False, point
        discontinuities issue a warning and overlapping chains are rejected.

    Returns
    -------
    tuple[gp.Variable, list[Equation]]

    Examples
    --------
    >>> from gamspy import Container, Variable
    >>> from gamspy.formulations import pwl_dlog_formulation
    >>> m = Container()
    >>> x = Variable(m, "x")
    >>> y, eqs = pwl_dlog_formulation(x, [0, 1, 2], [0, 3, 1])

    """
    if not isinstance(input_x, gp.Variable):
        raise ValidationError("input_x is expected to be a Variable")

    if not isinstance(bound_left, bool):
        raise ValidationError("bound_left is expected to be a boolean")

    if not isinstance(bound_right, bool):
        raise ValidationError("bound_right is expected to be a boolean")

    if not isinstance(allow_multivalued, bool):
        raise ValidationError("allow_multivalued is expected to be a boolean")

    data = _normalize_paired_data(
        input_x,
        x_points,
        y_points,
        bound_left=bound_left,
        bound_right=bound_right,
    )
    _validate_multivalued_data(data, allow_multivalued, "dlog")

    return _build_dlog_formulation(input_x, data)


def _validate_convexity_rows(rows: tuple[_PwlRow, ...]) -> None:
    for row in rows:
        row_location = f" in row {row.key}" if row.key else ""
        if len(row.ordered_points) == 1 and len(row.rays) == 2:
            raise ValidationError(
                "Convexity formulation cannot represent left and right rays "
                f"sharing one finite point{row_location}. Add another finite point "
                "or use pwl_interval_formulation or pwl_dlog_formulation instead."
            )

        for separator_index in row.separator_indices:
            left = row.ordered_points[separator_index]
            right = row.ordered_points[separator_index + 1]
            # A decrease across a separator means independent chains overlap
            # on the x-axis.
            if right.x < left.x:
                raise ValidationError(
                    "Convexity formulation cannot represent independent chains with "
                    f"decreasing x-order{row_location} because they have no single "
                    "non-decreasing SOS2 ordering. Use "
                    "pwl_interval_formulation or pwl_dlog_formulation instead."
                )


def _build_convexity_formulation(
    input_x: gp.Variable,
    data: _PwlData,
    using: typing.Literal["binary", "sos2"],
) -> tuple[gp.Variable, list[gp.Equation]]:
    m = input_x.container
    rows = data.rows
    input_domain = typing.cast("list[gp.Set]", list(input_x.domain))
    curve_domain = [*data.curve_domain]
    broadcast = data.is_broadcast
    equations: list[gp.Equation] = []

    # Lambda variables are rectangular; active_points trims the unused tail of
    # shorter rows.
    max_points = max(len(row.ordered_points) for row in rows)
    J = gp.math._generate_dims(m, [max_points])[0]
    J = gp.formulations.utils._next_domains([J], input_domain)[0]
    point_domain = [*input_domain, J]
    # Lambda variables use the complete input domain, while their point data
    # broadcasts over dimensions omitted from the curve domain.
    point_data_domain = [*curve_domain, J]
    point_index = [*input_domain, J]
    point_data_index = [*curve_domain, J]

    active_records = []
    x_records = []
    y_records = []
    for row in rows:
        prefix = row.key
        for index, point in enumerate(row.ordered_points):
            slot = str(index)
            active_records.append((*prefix, slot))
            x_records.append((*prefix, slot, point.x))
            y_records.append((*prefix, slot, point.y))

    active_points = m.addSet(domain=point_data_domain, records=active_records)
    x_par = m.addParameter(
        domain=point_data_domain,
        records=x_records,
    )
    y_par = m.addParameter(
        domain=point_data_domain,
        records=y_records,
    )
    lambda_var = m.addVariable(
        domain=point_domain,
        type="free" if using == "binary" else "sos2",
    )
    lambda_var.lo[...] = 0
    lambda_var.up[...] = 1
    # The binary helper resets lambda bounds, so apply the row mask afterwards.
    if using == "binary" and max_points > 1:
        equations.extend(_enforce_sos2_with_binary(lambda_var))
    lambda_var.fx[point_index].where[~active_points[point_data_index]] = 0

    out_y = m.addVariable(domain=input_domain)
    _set_pwl_output_bounds(
        out_y,
        rows,
        curve_domain=data.curve_domain,
    )

    x_term = gp.Sum(
        J,
        active_points[point_data_index]
        * x_par[point_data_index]
        * lambda_var[point_index],
    )
    y_term = gp.Sum(
        J,
        active_points[point_data_index]
        * y_par[point_data_index]
        * lambda_var[point_index],
    )

    # Rows may have different ray counts, so rays also use rectangular slots.
    max_rays = max(len(row.rays) for row in rows)
    if max_rays:
        R = gp.math._generate_dims(m, [max_rays])[0]
        R = gp.formulations.utils._next_domains([R], input_domain)[0]
        ray_domain = [*input_domain, R]
        ray_data_domain = [*curve_domain, R]
        ray_index = [*input_domain, R]
        ray_data_index = [*curve_domain, R]
        ray_endpoint_domain = [*curve_domain, R, J]
        ray_endpoint_index = [*curve_domain, R, J]

        active_records = []
        direction_records = []
        gradient_records = []
        endpoint_records = []
        for row in rows:
            prefix = row.key
            for index, ray in enumerate(row.rays):
                slot = str(index)
                base_index = next(
                    point_index
                    for point_index, point in enumerate(row.ordered_points)
                    if point is ray.base
                )
                active_records.append((*prefix, slot))
                direction_records.append((*prefix, slot, ray.direction))
                gradient_records.append((*prefix, slot, ray.gradient))
                endpoint_records.append((*prefix, slot, str(base_index)))

        active_rays = m.addSet(domain=ray_data_domain, records=active_records)
        direction = m.addParameter(domain=ray_data_domain, records=direction_records)
        gradient = m.addParameter(domain=ray_data_domain, records=gradient_records)
        ray_endpoints = m.addSet(
            domain=ray_endpoint_domain,
            records=endpoint_records,
        )
        movement, ray_selector, ray_equations = _generate_ray(m, ray_domain)
        equations.extend(ray_equations)

        movement.fx[ray_index].where[~active_rays[ray_data_index]] = 0
        ray_selector.fx[ray_index].where[~active_rays[ray_data_index]] = 0

        # A ray can activate only with lambda weight at its finite base point.
        limit_ray = m.addEquation(domain=[*input_domain, R, J])
        limit_index = [*input_domain, R, J]
        limit_ray[limit_index].where[ray_endpoints[ray_endpoint_index]] = (
            ray_selector[ray_index] <= lambda_var[point_index]
        )
        equations.append(limit_ray)

        x_term += gp.Sum(
            R,
            active_rays[ray_data_index]
            * direction[ray_data_index]
            * movement[ray_index],
        )
        y_term += gp.Sum(
            R,
            active_rays[ray_data_index]
            * direction[ray_data_index]
            * gradient[ray_data_index]
            * movement[ray_index],
        )

    lambda_sum = m.addEquation(domain=input_domain)
    lambda_sum[...] = (
        gp.Sum(
            J,
            active_points[point_data_index] * lambda_var[point_index],
        )
        == 1
    )
    equations.append(lambda_sum)

    set_x = m.addEquation(domain=input_domain)
    set_x[...] = input_x == x_term
    equations.append(set_x)

    set_y = m.addEquation(domain=input_domain)
    set_y[...] = out_y == y_term
    equations.append(set_y)

    # Broadcast data has one separator layout; Parameter rows need their own
    # separator mappings.
    if broadcast:
        if rows[0].separator_indices:
            equations.extend(
                _enforce_discontinuity(lambda_var, rows[0].separator_indices)
            )
    else:
        equations.extend(
            _enforce_indexed_discontinuity(
                lambda_var,
                rows,
                data.curve_domain,
            )
        )

    return out_y, equations


def pwl_convexity_formulation(
    input_x: gp.Variable,
    x_points: typing.Sequence[int | float | None] | gp.Parameter,
    y_points: typing.Sequence[int | float | None] | gp.Parameter,
    using: typing.Literal["binary", "sos2"] = "binary",
    *,
    bound_left: bool = True,
    bound_right: bool = True,
    allow_multivalued: bool = False,
) -> tuple[gp.Variable, list[gp.Equation]]:
    """
    Create a piecewise-linear relationship using the Convexity formulation.

    This function implements a piecewise linear function using the convexity formulation.
    Given an input (independent) variable `input_x`, along with the defining `x_points`
    and corresponding `y_points` of the piecewise function, it constructs the dependent
    variable `y` and formulates the equations necessary to define the function.

    Here is the convexity formulation:

    .. math::
        x = \\sum_{i}{x\\_points_i * \\lambda_i}

        y = \\sum_{i}{y\\_points_i * \\lambda_i}

        \\sum_{i}{\\lambda_i} = 1

        \\lambda_i \\in SOS2


    By default, SOS2 variables are implemented using binary variables, following
    `Modeling disjunctive constraints with a logarithmic number of binary variables and constraints
    <https://link.springer.com/article/10.1007/s10107-009-0295-4>`_. Native SOS2
    variables can instead be selected with `using="sos2"`.

    Repeated x-coordinates describe discontinuities, and matching `None` values
    exclude the range between two chains. The finite points must retain one
    non-decreasing SOS2 ordering, so Convexity does not support overlapping
    independent chains.

    Extreme `-math.inf` and `math.inf` values in `x_points` define explicit
    rays; the matching `y_points` values are their gradients. Otherwise,
    setting a bound argument to False extends the corresponding outer segment.
    Convexity cannot represent both rays when they share a single finite point;
    add another finite point or use Interval or DLog for that graph.

    Parameters
    ----------
    input_x : gp.Variable
        Independent variable of the piecewise linear function
    x_points : typing.Sequence[int | float | None] | gp.Parameter
        Breakpoints of the piecewise linear function in the x-axis. An indexed
        Parameter may use an ordered subset of the input domains followed by
        one breakpoint domain. It is broadcast over omitted input domains.
    y_points : typing.Sequence[int | float | None] | gp.Parameter
        Breakpoints of the piecewise linear function in the y-axis.
        Parameter input must exactly match the container and domains
        of `x_points`.
    using : typing.Literal["binary", "sos2"] = "binary"
        Whether to implement SOS2 adjacency with binary variables or native
        SOS2 variables.
    bound_left : bool = True
        Whether `input_x` is limited to start at the first finite point. If
        False, the first finite segment is extended to the left.
    bound_right : bool = True
        Whether `input_x` is limited to end at the last finite point. If False,
        the last finite segment is extended to the right.
    allow_multivalued : bool = False
        Whether to acknowledge point discontinuities and suppress their warning.
        Overlapping chains are unsupported by the Convexity formulation and are
        rejected regardless of this argument.

    Returns
    -------
    tuple[gp.Variable, list[Equation]]

    Examples
    --------
    >>> from gamspy import Container, Variable, Set
    >>> from gamspy.formulations import pwl_convexity_formulation
    >>> m = Container()
    >>> x = Variable(m, "x")
    >>> y, eqs = pwl_convexity_formulation(
    ...     x,
    ...     [-1, 4, 10, 10, 20],
    ...     [-2, 8, 15, 17, 37],
    ...     allow_multivalued=True,
    ... )

    """
    if using not in {"binary", "sos2"}:
        raise ValidationError(
            "Invalid value for the using argument."
            "Possible values are 'binary' and 'sos2'"
        )

    if not isinstance(input_x, gp.Variable):
        raise ValidationError("input_x is expected to be a Variable")

    if not isinstance(bound_left, bool):
        raise ValidationError("bound_left is expected to be a boolean")

    if not isinstance(bound_right, bool):
        raise ValidationError("bound_right is expected to be a boolean")

    if not isinstance(allow_multivalued, bool):
        raise ValidationError("allow_multivalued is expected to be a boolean")

    data = _normalize_paired_data(
        input_x,
        x_points,
        y_points,
        bound_left=bound_left,
        bound_right=bound_right,
    )

    _validate_multivalued_data(data, allow_multivalued, "convexity")
    _validate_convexity_rows(data.rows)

    return _build_convexity_formulation(
        input_x,
        data,
        using,
    )


def pwlinear(
    input_x: gp.Variable,
    curve: PWLCurve | Mapping[str | tuple[str, ...], PWLCurve],
    *,
    curve_domain: typing.Sequence[gp.Set | gp.Alias] | None = None,
    method: typing.Literal["interval", "convexity", "dlog"] = "interval",
    using: typing.Literal["binary", "sos2"] | None = None,
    allow_multivalued: bool = False,
) -> tuple[gp.Variable, list[gp.Equation]]:
    """Create a piecewise-linear relationship from a PWLCurve.

    Parameters
    ----------
    input_x : gp.Variable
        Independent variable of the piecewise-linear function.
    curve : PWLCurve | Mapping[str | tuple[str, ...], PWLCurve]
        One shared curve, or a dictionary of domain keys and curves.
    curve_domain : Sequence[gp.Set | gp.Alias] | None, optional
        Input domains represented by the dictionary keys. They must be an ordered
        subset of `input_x.domain`; the curves are broadcast over omitted
        domains. If omitted, a dictionary uses every input domain.
    method : str, optional
        Formulation method. Possible values are ``"interval"``,
        ``"convexity"``, and ``"dlog"``.
    using : str | None, optional
        SOS2 implementation used by the Convexity method. ``None`` selects
        ``"binary"``. This argument is ignored by Interval and DLog.
    allow_multivalued : bool, optional
        Whether to explicitly allow multivalued curves. If False, point
        discontinuities issue a warning and overlapping chains are rejected.

    Returns
    -------
    tuple[gp.Variable, list[gp.Equation]]
        Dependent variable and equations defining the relationship.
    """
    if method not in {"interval", "convexity", "dlog"}:
        raise ValidationError(
            "Invalid value for the method argument. Possible values are "
            "'interval', 'convexity', and 'dlog'"
        )

    if not isinstance(allow_multivalued, bool):
        raise ValidationError("allow_multivalued is expected to be a boolean")

    if method == "convexity":
        if using is None:
            using = "binary"
        elif using not in {"binary", "sos2"}:
            raise ValidationError(
                "Invalid value for the using argument. "
                "Possible values are 'binary' and 'sos2'"
            )
    elif using is not None:
        warnings.warn(
            f"The using argument is ignored when method='{method}'.",
            UserWarning,
            stacklevel=2,
        )
        using = None

    data = _normalize_curve_data(input_x, curve, curve_domain)
    _validate_multivalued_data(data, allow_multivalued, method)

    if method == "convexity":
        _validate_convexity_rows(data.rows)
        return _build_convexity_formulation(
            input_x,
            data,
            using,
        )
    elif method == "dlog":
        return _build_dlog_formulation(input_x, data)
    else:
        return _build_interval_formulation(input_x, data)
