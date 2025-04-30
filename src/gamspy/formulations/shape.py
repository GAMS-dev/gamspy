from __future__ import annotations

import math

import gamspy as gp
import gamspy.formulations.nn.utils as utils
from gamspy.exceptions import ValidationError


def _get_new_domain(
    x: gp.Variable | gp.Parameter, dims: list[int]
) -> tuple[list[gp.Set | gp.Alias], gp.Set]:
    lens = [len(x.domain[d]) for d in dims]
    new_card = math.prod(lens)

    # get new domain
    flattened = gp.math._generate_dims(x.container, [new_card])
    # flattened should be unique, this is only for declaration
    flattened = utils._next_domains(
        flattened, [*x.domain[: min(dims)], *x.domain[max(dims) + 1 :]]
    )[0]
    new_domain = []
    new_domain.extend(x.domain[: min(dims)])
    new_domain.append(flattened)
    new_domain.extend(x.domain[max(dims) + 1 :])

    return new_domain, flattened


def _flatten_dims_par(
    x: gp.Parameter, dims: list[int]
) -> tuple[gp.Parameter, list[gp.Equation]]:
    data = x.toDense()

    m = x.container
    new_domain, _ = _get_new_domain(x, dims)

    new_shape = [len(d) for d in new_domain]

    if data is not None:
        data = data.reshape(new_shape)

    out = m.addParameter(domain=new_domain, records=data)
    return out, []


def _generate_index_matching_statement(
    domains: list[gp.Set], flattened: gp.Set, matching_set: gp.Set
) -> str:
    base_txt = "option {}({}:{})"
    domains_str = ",".join([x.name for x in domains])
    return base_txt.format(matching_set.name, domains_str, flattened.name)


def _propagate_bounds(x, out):
    """Propagate bounds from the input to the output variable"""

    m = x.container

    # set domain for variable
    x_domain = x.domain
    x_domain = utils._next_domains(x_domain, [])

    bounds_set = m.addSet(records=["lb", "ub"])
    bounds = m.addParameter(domain=[bounds_set, *x_domain])

    # capture original bounds
    bounds[("lb",) + tuple(x_domain)] = x.lo[x_domain]
    bounds[("ub",) + tuple(x_domain)] = x.up[x_domain]

    # reshape bounds based on the output variable's shape
    # when bounds.records is None, it means the bounds are zeros
    nb_data = (
        None
        if bounds.records is None
        else bounds.toDense().reshape((2,) + out.shape)
    )

    # set new domain for bounds
    nb_dom = [bounds_set, *out.domain]

    new_bounds = m.addParameter(domain=nb_dom, records=nb_data)

    # assign new bounds to the output variable
    out.lo[...] = new_bounds[("lb",) + tuple(out.domain)]
    out.up[...] = new_bounds[("ub",) + tuple(out.domain)]


def _flatten_dims_var(
    x: gp.Variable, dims: list[int], propagate_bounds: bool = True
) -> tuple[gp.Variable, list[gp.Equation]]:
    m = x.container
    new_domain, flattened = _get_new_domain(x, dims)

    out = m.addVariable(
        domain=new_domain
    )  # outputs domain nearly matches the input domain

    if propagate_bounds and x.records is not None:
        _propagate_bounds(x, out)

    # match the flattened set to correct dims
    forwarded_domain = utils._next_domains([flattened, *x.domain], [])
    doms_to_flatten = [forwarded_domain[d + 1] for d in dims]

    name = "ds_" + gp.utils._get_unique_name()
    subset = m.addSet(name, domain=[*doms_to_flatten, flattened])
    m.addGamsCode(
        _generate_index_matching_statement(doms_to_flatten, flattened, subset)
    )

    set_out = m.addEquation(domain=forwarded_domain)
    set_out[forwarded_domain].where[subset[[*doms_to_flatten, flattened]]] = (
        out[
            [
                *forwarded_domain[1 : min(dims) + 1],
                flattened,
                *forwarded_domain[max(dims) + 2 :],
            ]
        ]
        == x[forwarded_domain[1:]]
    )
    return out, [set_out]


def flatten_dims(
    x: gp.Variable | gp.Parameter,
    dims: list[int],
    propagate_bounds: bool = True,
) -> tuple[gp.Parameter | gp.Variable, list[gp.Equation]]:
    """
    Flatten domains indicated by `dims` into a single domain.
    If `propagate_bounds` is True, and `x` is of type variable,
    the bounds of the input variable are propagated to the output.

    Parameters
    ----------
    x : gp.Variable | gp.Parameter
        Input to be flattened
    dims: list[int]
        List of integers indicating indices of the domains
        to be flattened. Must be consecutive indices.
    propagate_bounds: bool, optional
        Propagate bounds from the input to the output variable. Default is True.

    Examples
    --------
    >>> import gamspy as gp
    >>> from gamspy.math import dim
    >>> m = gp.Container()
    >>> inp = gp.Variable(m, domain=dim((10, 1, 24, 24)))
    >>> out, eqs = gp.formulations.flatten_dims(inp, [2, 3])
    >>> type(out)
    <class 'gamspy._symbols.variable.Variable'>
    >>> [len(x) for x in out.domain]
    [10, 1, 576]

    """
    if not isinstance(x, (gp.Parameter, gp.Variable)):
        raise ValidationError("Expected a parameter or a variable input")

    if not isinstance(propagate_bounds, bool):
        raise ValidationError("Expected a boolean for propagate_bounds")

    if len(dims) < 2:
        raise ValidationError("Expected at least 2 items in the dim array")

    x_len = len(x.domain)
    for i, d in enumerate(dims):
        if not isinstance(d, int):
            raise ValidationError("Expected integers in the dim array")

        if d < 0 or d >= x_len:
            raise ValidationError(
                "dims must contain numbers between 0 to len(domain) - 1"
            )

        if i > 0 and dims[i - 1] != d - 1:
            raise ValidationError(
                "Expected consecutive integers in the dim array"
            )

    for domain in x.domain:
        if len(domain) == 0:
            raise ValidationError(
                f"domain {domain} had 0 cardinality, please populate the domain first"
            )

    if isinstance(x, gp.Parameter):
        return _flatten_dims_par(x, dims)

    return _flatten_dims_var(x, dims, propagate_bounds)
