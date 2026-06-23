from __future__ import annotations

import numpy as np
import pytest

import gamspy as gp
from gamspy.exceptions import ValidationError
from gamspy.formulations import flatten_dims
from gamspy.math import dim


@pytest.mark.unit
def test_flatten_bad(data):
    m, w1, _b1, _inp, par_input, _ii, *_ = data
    # should only work for parameter or variable
    with pytest.raises(ValidationError):
        flatten_dims(w1, [2, 3])
    with pytest.raises(ValidationError):
        flatten_dims(par_input, [0])  # single dim
    with pytest.raises(ValidationError):
        flatten_dims(par_input, [])  # no dim
    with pytest.raises(ValidationError):
        flatten_dims(par_input, ["a", "b"])
    with pytest.raises(ValidationError):
        flatten_dims(par_input, [-1, 0])
    with pytest.raises(ValidationError):
        flatten_dims(par_input, [5, 6])
    with pytest.raises(ValidationError):
        flatten_dims(par_input, [1, 3])  # non consecutive
    with pytest.raises(ValidationError):
        flatten_dims(
            par_input, [0, 1], propagate_bounds="True"
        )  # propagate_bounds not bool

    i = gp.Set(m, "i")
    j = gp.Set(m, "j")
    k = gp.Set(m, "k")
    var1 = gp.Variable(m, "var1", domain=[i, j, k])  # j, k not populated yet
    with pytest.raises(ValidationError):
        flatten_dims(var1, [1, 2])


@pytest.mark.unit
def test_flatten_par(data):
    m, _w1, _b1, inp, par_input, ii, *_ = data
    # 3x1x5x5 -> 3x25
    par_flattened, eqs = flatten_dims(par_input, [1, 2, 3])
    out_data = par_flattened.toDense()
    assert np.allclose(out_data, inp.reshape(3, 25))
    assert eqs == []  # for parameters no equation needed

    # 3x1x5x5 -> 75
    par_flattened, eqs = flatten_dims(par_input, [0, 1, 2, 3])
    out_data = par_flattened.toDense()
    assert np.allclose(out_data, inp.reshape(75))
    assert eqs == []  # for parameters no equation needed

    # 3x1x5x5 -> 3x1x25
    par_flattened, eqs = flatten_dims(par_input, [2, 3])
    out_data = par_flattened.toDense()
    assert np.allclose(out_data, inp.reshape(3, 1, 25))
    assert eqs == []  # for parameters no equation needed

    # test flatten par with copied domain as well
    data = np.random.rand(20, 20, 20, 20)
    par = gp.Parameter(m, "par_iii", domain=[ii, ii, ii, ii], records=data)

    par_flattened, eqs = flatten_dims(par, [2, 3])
    out_data = par_flattened.toDense()
    assert np.allclose(out_data, data.reshape(20, 20, 400))
    assert eqs == []  # for parameters no equation needed


@pytest.mark.unit
def test_flatten_par_with_no_records(data):
    m, *_ = data

    par = gp.Parameter(m, "par", domain=dim([10, 5]))

    # 10x5 -> 50
    par_flattened, eqs = flatten_dims(par, [0, 1])

    assert np.allclose(
        par_flattened.toDense(), np.zeros(par_flattened.shape, dtype=float)
    )
    assert par_flattened.shape == (50,)
    assert eqs == []  # for parameters no equation needed


@pytest.mark.requires_license
def test_flatten_var_copied_domain(data):
    m, _w1, _b1, _inp, _par_input, ii, *_ = data

    a1 = gp.Alias(m, "ii2", alias_with=ii)
    a2 = gp.Alias(m, "ii3", alias_with=ii)
    a3 = gp.Alias(m, "ii4", alias_with=ii)

    var = gp.Variable(
        m,
        "var_ii",
        domain=[ii, ii, ii, ii],
    )
    data = np.random.rand(20, 20, 20, 20)

    fix_var = gp.Parameter(m, "var_ii_data", domain=var.domain, records=data)
    var.fx[ii, a1, a2, a3] = fix_var[ii, a1, a2, a3]
    var_2, eqs = flatten_dims(var, [2, 3])
    var_3, eqs_2 = flatten_dims(var_2, [0, 1])
    var_4, eqs_3 = flatten_dims(var_3, [0, 1])

    model = gp.Model(
        m,
        "flatten_everything",
        equations=[*eqs, *eqs_2, *eqs_3],
        problem="lp",
        objective=var_4["240"] + 1,
        sense="min",
    )

    model.solve()

    out_data = var_3.toDense()
    assert np.allclose(out_data, data.reshape(400, 400))

    out_data_2 = var_4.toDense()
    assert np.allclose(out_data_2, data.reshape(400 * 400))


@pytest.mark.requires_license
def test_flatten_2d_propagate_bounds(data):
    m, *_ = data
    i = gp.Set(m, name="i", records=[f"i{i}" for i in range(1, 41)])
    j = gp.Set(m, name="j", records=[f"j{j}" for j in range(1, 51)])
    var = gp.Variable(m, name="var", domain=[i, j])

    # If the variable is unbounded, the bounds are not propagated even if propagate_bounds is True
    var_1, eqs_1 = flatten_dims(var, [0, 1], propagate_bounds=True)
    var_2, eqs_2 = flatten_dims(var, [0, 1], propagate_bounds=False)
    assert var_1.records == var_2.records

    # If the variable is bounded, the bounds are propagated
    bound_up = np.random.rand(40, 50) * 5
    bound_lo = np.random.rand(40, 50) * -5
    upper = gp.Parameter(m, name="upper", domain=[i, j], records=bound_up)
    lower = gp.Parameter(m, name="lower", domain=[i, j], records=bound_lo)
    var.up[...] = upper[...]
    var.lo[...] = lower[...]

    var_3, eqs_3 = flatten_dims(var, [0, 1])

    model = gp.Model(
        m,
        "flatten_everything",
        equations=[*eqs_1, *eqs_2, *eqs_3],
        problem="lp",
        objective=var_3["240"] + 1,
        sense="min",
    )

    model.solve()

    assert np.allclose(np.array(var_3.records.lower.tolist()), bound_lo.reshape(2000))
    assert np.allclose(np.array(var_3.records.upper.tolist()), bound_up.reshape(2000))


@pytest.mark.requires_license
def test_flatten_3d_propagate_bounds(data):
    m, *_ = data
    i = gp.Set(m, name="i", records=[f"i{i}" for i in range(1, 41)])
    j = gp.Set(m, name="j", records=[f"j{j}" for j in range(1, 51)])
    k = gp.Set(m, name="k", records=[f"k{k}" for k in range(1, 21)])
    var = gp.Variable(m, name="var", domain=[i, j, k])
    bounds_set = gp.Set(m, name="bounds_set", records=["lb", "ub"])

    # If the variable is unbounded, the bounds are not propagated even if propagate_bounds is True
    var_1, eqs_1 = flatten_dims(var, [0, 1], propagate_bounds=True)
    var_2, eqs_2 = flatten_dims(var, [1, 2], propagate_bounds=True)
    assert (var_1.records is None) and (var_2.records is None)
    assert var_1.shape == (2000, 20) and var_2.shape == (40, 1000)

    # If the variable is bounded, the bounds are propagated
    bound_up = np.random.rand(40, 50, 20) * 5
    bound_lo = np.random.rand(40, 50, 20) * -5
    all_bounds = np.stack([bound_lo, bound_up], axis=0)

    bounds = gp.Parameter(
        m, name="bounds", domain=[bounds_set, i, j, k], records=all_bounds
    )

    var.up[...] = bounds[("ub", i, j, k)]
    var.lo[...] = bounds[("lb", i, j, k)]

    var_3, eqs_3 = flatten_dims(var, [0, 1])
    var_4, eqs_4 = flatten_dims(var, [1, 2])
    var_5, eqs_5 = flatten_dims(var, [0, 1, 2])

    model = gp.Model(
        m,
        "flatten_everything",
        equations=[*eqs_1, *eqs_2, *eqs_3, *eqs_4, *eqs_5],
        problem="lp",
        objective=var_3["140", "k4"] + 1,
        sense="min",
    )

    model.solve()

    var_3_bounds = gp.Parameter(
        m, name="var_3_bounds", domain=[bounds_set, *var_3.domain]
    )
    var_3_bounds[("lb",) + tuple(var_3.domain)] = var_3.lo[...]
    var_3_bounds[("ub",) + tuple(var_3.domain)] = var_3.up[...]

    var_4_bounds = gp.Parameter(
        m, name="var_4_bounds", domain=[bounds_set, *var_4.domain]
    )
    var_4_bounds[("lb",) + tuple(var_4.domain)] = var_4.lo[...]
    var_4_bounds[("ub",) + tuple(var_4.domain)] = var_4.up[...]

    var_5_bounds = gp.Parameter(
        m, name="var_5_bounds", domain=[bounds_set, *var_5.domain]
    )
    var_5_bounds[("lb",) + tuple(var_5.domain)] = var_5.lo[...]
    var_5_bounds[("ub",) + tuple(var_5.domain)] = var_5.up[...]

    assert np.allclose(var_3_bounds.toDense(), all_bounds.reshape(2, 2000, 20))
    assert np.allclose(var_4_bounds.toDense(), all_bounds.reshape(2, 40, 1000))
    assert np.allclose(var_5_bounds.toDense(), all_bounds.reshape(2, 40000))


@pytest.mark.requires_license
def test_flatten_propagate_zero_bounds(data):
    m, *_ = data
    var = gp.Variable(m, name="var1", domain=dim([30, 40, 10, 5]))

    var.up[...] = 0
    var.lo[...] = 0

    var_1, eqs_1 = flatten_dims(var, [0, 1])
    var_2, eqs_2 = flatten_dims(var, [0, 1, 2, 3])
    var_3, eqs_3 = flatten_dims(var_1, [1, 2])

    model = gp.Model(
        m,
        "flatten_everything",
        equations=[*eqs_1, *eqs_2, *eqs_3],
        problem="lp",
        objective=5 * var_1["0", "0", "0"] + 1,
        sense="min",
    )

    model.solve()

    expected_bounds = np.zeros([30, 40, 10, 5])

    # Because the bounds are zero, there are no way, currently, to represent them as an array
    assert np.allclose(
        np.array(var_1.records.upper).reshape(var_1.shape),
        expected_bounds.reshape(var_1.shape),
    )
    assert np.allclose(
        np.array(var_1.records.lower).reshape(var_1.shape),
        expected_bounds.reshape(var_1.shape),
    )

    assert np.allclose(
        np.array(var_2.records.upper).reshape(var_2.shape),
        expected_bounds.reshape(var_2.shape),
    )
    assert np.allclose(
        np.array(var_2.records.lower).reshape(var_2.shape),
        expected_bounds.reshape(var_2.shape),
    )

    assert np.allclose(
        np.array(var_3.records.upper).reshape(var_3.shape),
        expected_bounds.reshape(var_3.shape),
    )
    assert np.allclose(
        np.array(var_3.records.lower).reshape(var_3.shape),
        expected_bounds.reshape(var_3.shape),
    )


@pytest.mark.unit
def test_flatten_more_complex_propagate_bounds(data):
    m, *_ = data
    var = gp.Variable(m, name="var", domain=dim([2, 4, 5]))
    bounds_set = gp.Set(m, name="bounds_set", records=["lb", "ub"])

    bound_up = np.array(
        [
            [
                [1.6873254, np.inf, 4.64399079, np.inf, 0.85146007],
                [4.31392932, 1.99165668, 4.19013802, 3.77449253, 0],
                [np.inf, 4.13450595, 4.25880061, 1.529363, 2.54171194],
                [1.79348688, 2.04002383, 0.19198094, 4.14445882, 4.72650868],
            ],
            [
                [1.54070398, np.inf, 0, 3.55077501, 2.12700496],
                [0.13939228, 1.10668786, 0.23710837, 3.61857607, 1.64761417],
                [1.80097419, 0.89434166, 1.46039526, 1.31960681, np.inf],
                [2.50636193, 1.3920737, np.inf, 3.35616509, 4.98534911],
            ],
        ]
    )

    bound_lo = -bound_up
    all_bounds = np.stack([bound_lo, bound_up], axis=0)

    bounds = gp.Parameter(
        m, name="bounds", domain=[bounds_set, *var.domain], records=all_bounds
    )

    var.up[...] = bounds[("ub",) + tuple(var.domain)]
    var.lo[...] = bounds[("lb",) + tuple(var.domain)]

    var_1, eqs_1 = flatten_dims(var, [0, 1])

    model = gp.Model(
        m,
        "flatten_everything",
        equations=[*eqs_1],
        problem="lp",
        objective=5 * var_1["5", "3"] + 1,
        sense="min",
    )

    model.solve()

    var_1_bounds = gp.Parameter(
        m, name="var_1_bounds", domain=[bounds_set, *var_1.domain]
    )
    var_1_bounds[("lb",) + tuple(var_1.domain)] = var_1.lo[...]
    var_1_bounds[("ub",) + tuple(var_1.domain)] = var_1.up[...]

    assert np.allclose(var_1_bounds.toDense(), all_bounds.reshape(2, 8, 5))
