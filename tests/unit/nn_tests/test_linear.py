from __future__ import annotations

import numpy as np
import pytest

import gamspy as gp
from gamspy.exceptions import ValidationError
from gamspy.formulations.nn import Linear
from gamspy.formulations.result import FormulationResult
from gamspy.math import dim


@pytest.mark.unit
def test_linear_bad_init(data):
    m, *_ = data
    # in feature must be integer
    with pytest.raises(ValidationError):
        Linear(m, 2.5, 4, False)
    with pytest.raises(ValidationError):
        Linear(m, "2", 4, True)
    # out feature must be integer
    with pytest.raises(ValidationError):
        Linear(m, 2, 4.1, True)
    with pytest.raises(ValidationError):
        Linear(m, 2, "4.1", False)
    # in feature must be positive
    with pytest.raises(ValidationError):
        Linear(m, -4, 4, False)
    # out feature must be positive
    with pytest.raises(ValidationError):
        Linear(m, 4, -4, False)
    # bias must be a bool
    with pytest.raises(ValidationError):
        Linear(m, 4, 4, bias=10)


@pytest.mark.unit
def test_linear_load_weights(data):
    m, *_ = data
    lin1 = Linear(m, 1, 2, bias=True)
    lin2 = Linear(m, 1, 2, bias=False)

    w1 = np.random.rand(2, 1)
    b1 = np.random.rand(2)

    # needs bias as well
    with pytest.raises(ValidationError):
        lin1.load_weights(w1)

    # conv2 does not have bias
    with pytest.raises(ValidationError):
        lin2.load_weights(w1, b1)

    # test bad shape
    bad1 = np.random.rand(1)
    with pytest.raises(ValidationError):
        lin1.load_weights(bad1, b1)
    with pytest.raises(ValidationError):
        lin1.load_weights(w1, bad1)

    bad2 = np.random.rand(2, 2)
    with pytest.raises(ValidationError):
        lin1.load_weights(bad2, b1)

    bad3 = np.random.rand(6)
    with pytest.raises(ValidationError):
        lin1.load_weights(w1, bad3)

    bad4 = np.random.rand(6, 2)
    with pytest.raises(ValidationError):
        lin1.load_weights(w1, bad4)


@pytest.mark.unit
def test_linear_same_indices(data):
    m, *_ = data
    lin1 = Linear(m, 4, 4, bias=True)
    w1 = np.random.rand(4, 4)
    b1 = np.random.rand(4)
    lin1.load_weights(w1, b1)
    inp = gp.Variable(m, domain=dim([4, 4, 4, 4]))
    _out, _eqs = lin1(inp)
    lin2 = Linear(m, 4, 4, bias=True)
    lin2.load_weights(w1, b1)
    _out2, _eqs2 = lin2(inp)


@pytest.mark.unit
def test_linear_reloading_weights(data):
    m, *_ = data
    lin1 = Linear(m, 1, 2, bias=True)
    w1 = np.random.rand(2, 1)
    b1 = np.random.rand(2)
    lin1.load_weights(w1, b1)

    w1 = np.ones((2, 1))
    b1 = np.ones(2)
    lin1.load_weights(w1, b1)

    assert np.allclose(w1, lin1.weight.toDense())
    assert np.allclose(b1, lin1.bias.toDense())


@pytest.mark.unit
def test_linear_make_variable(data):
    m, *_ = data
    lin1 = Linear(m, 4, 2)
    lin1.make_variable()
    assert lin1.weight.records is None
    assert lin1.bias.records is None

    lin2 = Linear(m, 4, 2)
    lin2.make_variable(init_weights=True)
    assert lin2.weight.records is not None
    assert lin2.bias.records is not None
    w1 = np.random.rand(2, 4)
    b1 = np.random.rand(2)
    with pytest.raises(ValidationError):
        lin1.load_weights(w1, b1)
    assert isinstance(lin1.weight, gp.Variable)
    assert isinstance(lin1.bias, gp.Variable)
    inp = gp.Variable(m, domain=dim([4, 1, 2, 4]))
    out, _eqs = lin1(inp)
    assert len(out.domain) == 4
    assert len({x.name for x in out.domain}) == 4


@pytest.mark.unit
def test_linear_load_weight_make_var(data):
    m, *_ = data
    lin1 = Linear(m, 1, 2, bias=True)
    w1 = np.random.rand(2, 1)
    b1 = np.random.rand(2)
    lin1.load_weights(w1, b1)
    assert isinstance(lin1.weight, gp.Parameter)
    assert isinstance(lin1.bias, gp.Parameter)
    with pytest.raises(ValidationError):
        lin1.make_variable()


@pytest.mark.unit
def test_linear_call_bad(data):
    m, *_ = data
    lin1 = Linear(m, 4, 4, bias=True)
    inp = gp.Variable(m, domain=dim([4, 4, 4, 4]))
    # requires initialization before
    with pytest.raises(ValidationError):
        lin1(inp)

    w1 = np.random.rand(4, 4)
    b1 = np.random.rand(4)
    lin1.load_weights(w1, b1)

    # needs at least 1 dim
    bad_inp = gp.Variable(m, domain=[])
    with pytest.raises(ValidationError):
        lin1(bad_inp)

    # in channel must match 4
    bad_inp_2 = gp.Variable(m, domain=dim([10, 3, 4, 5]))
    with pytest.raises(ValidationError):
        lin1(bad_inp_2)


@pytest.mark.requires_license
def test_linear_simple_correctness(data):
    m, _, _, _, par_input, *_ = data
    lin1 = Linear(m, 5, 128, bias=True)
    lin2 = Linear(m, 128, 64, bias=False)
    w1 = np.random.rand(128, 5)
    b1 = np.random.rand(128)
    w2 = np.random.rand(64, 128)

    lin1.load_weights(w1, b1)
    lin2.load_weights(w2)

    out1, eqs1 = lin1(par_input)
    out2, eqs2 = lin2(out1)

    obj = gp.Sum(out2.domain, out2)

    model = gp.Model(
        m,
        "affine",
        equations=[*eqs1, *eqs2],
        objective=obj,
        sense="min",
        problem="LP",
    )
    model.solve()

    expected_out = (par_input.toDense() @ w1.T + b1) @ w2.T
    assert np.allclose(out2.toDense(), expected_out)


@pytest.mark.requires_license
def test_linear_bias_domain_conflict(data):
    m, *_ = data
    lin1 = Linear(m, 20, 30, bias=True)
    w1 = np.random.rand(30, 20)
    b1 = np.random.rand(30)
    lin1.load_weights(w1, b1)

    input_data = np.random.rand(30, 20, 30, 20)
    par_input = gp.Parameter(m, domain=dim([30, 20, 30, 20]), records=input_data)
    out1, eqs1 = lin1(par_input)

    last_domain = out1.domain[-1].name
    # get rhs of equality
    definition = eqs1[0].getDefinition().split("=e=")[1]
    # 1 from weight 1 from bias
    assert definition.count(last_domain) == 2

    obj = gp.Sum(out1.domain, out1)

    model = gp.Model(
        m,
        "affine2",
        equations=eqs1,
        objective=obj,
        sense="min",
        problem="LP",
    )
    model.solve()
    expected_out = input_data @ w1.T + b1
    assert np.allclose(out1.toDense(), expected_out)


@pytest.mark.unit
def test_linear_propagate_bounds_non_boolean(data):
    m, *_ = data
    lin1 = Linear(m, 20, 30, bias=True)
    w1 = np.random.rand(30, 20)
    b1 = np.random.rand(30)
    lin1.load_weights(w1, b1)

    par_input = gp.Parameter(m, domain=dim([30, 20, 30, 20]))
    with pytest.raises(ValidationError):
        lin1(par_input, "True")


@pytest.mark.unit
def test_linear_propagate_bounded_input(data):
    m, *_ = data
    lin1 = Linear(m, 4, 3, name_prefix="lin1")
    w1 = np.random.rand(3, 4)
    b1 = np.random.rand(3)
    lin1.load_weights(w1, b1)

    # bounds for the input
    xlb = np.random.randint(-5, 1, (2, 4))
    xub = np.random.randint(1, 5, (2, 4))

    x_lb = gp.Parameter(m, "x_lb", domain=dim([2, 4]), records=xlb)
    x_ub = gp.Parameter(m, "x_ub", domain=dim([2, 4]), records=xub)

    x = gp.Variable(m, "x", domain=dim([2, 4]))
    x.up[...] = x_ub[...]
    x.lo[...] = x_lb[...]

    out1, _ = lin1(x)

    wpos = np.maximum(w1, 0)
    wneg = np.minimum(w1, 0)

    expected_lb = np.dot(xlb, wpos.T) + np.dot(xub, wneg.T) + b1
    expected_ub = np.dot(xub, wpos.T) + np.dot(xlb, wneg.T) + b1

    # check if the bounds are propagated correctly
    assert np.allclose(np.array(out1.lo.records.lower).reshape(2, 3), expected_lb)
    assert np.allclose(np.array(out1.up.records.upper).reshape(2, 3), expected_ub)

    output_var_found = False
    weight_par_found = False
    bias_par_found = False
    for sym_name in m.data:
        if sym_name.startswith("v_lin1_output"):
            output_var_found = True
        elif sym_name.startswith("p_lin1_weight"):
            weight_par_found = True
        elif sym_name.startswith("p_lin1_bias"):
            bias_par_found = True

    assert output_var_found
    assert weight_par_found
    assert bias_par_found


@pytest.mark.unit
def test_linear_propagate_unbounded_input(data):
    m, *_ = data
    lin1 = Linear(m, 20, 30, bias=True)
    w1 = np.random.rand(30, 20)
    b1 = np.random.rand(30)
    lin1.load_weights(w1, b1)

    x = gp.Variable(m, "x", domain=dim([30, 20, 30, 20]))

    out1, _ = lin1(x)

    # Data to bounds is not added; which means its unbounded
    assert out1.lo.records is None
    assert out1.up.records is None


@pytest.mark.unit
def test_linear_propagate_partially_bounded_input(data):
    m, *_ = data
    lin1 = Linear(m, 4, 3, bias=False)
    w1 = np.array([[-3, 2, 1, 0], [1, -1, 0, 1], [0, 1, -1, 3]])
    lin1.load_weights(w1)

    x = gp.Variable(m, "x", domain=dim([2, 4]))

    xlb = np.array([[-4, -np.inf, -5, 0], [-1, -2, -1, -4]])
    xub = np.array([[4, 5, np.inf, 0], [np.inf, 1, 2, 1]])

    x_lb = gp.Parameter(m, "x_lb", domain=dim([2, 4]), records=xlb)
    x_ub = gp.Parameter(m, "x_ub", domain=dim([2, 4]), records=xub)

    x.up[...] = x_ub[...]
    x.lo[...] = x_lb[...]

    out1, _ = lin1(x)

    out1_lb = gp.Parameter(m, "out1_lb", domain=dim([2, 3]))
    out1_ub = gp.Parameter(m, "out1_ub", domain=dim([2, 3]))

    out1_lb[...] = out1.lo[...]
    out1_ub[...] = out1.up[...]

    expected_lb = np.array([[-np.inf, -9, -np.inf], [-np.inf, -6, -16]])

    expected_ub = np.array([[np.inf, np.inf, 10], [7, np.inf, 5]])

    # check if the bounds are propagated correctly
    assert np.allclose(out1_lb.toDense(), expected_lb)
    assert np.allclose(out1_ub.toDense(), expected_ub)


@pytest.mark.unit
def test_linear_propagate_unbounded_input_with_zero_weight(data):
    m, *_ = data
    lin1 = Linear(m, 20, 30, bias=False)
    w1 = np.zeros((30, 20))
    lin1.load_weights(w1)

    x = gp.Variable(m, "x", domain=dim([30, 20, 30, 20]))

    out1, _ = lin1(x)
    out1_ub = np.array(out1.records.upper).reshape(30, 20, 30, 30)
    out1_lb = np.array(out1.records.lower).reshape(30, 20, 30, 30)

    expected_bounds = np.zeros((30, 20, 30, 30))

    # check if the bounds are zeros, since the weights are all zeros
    assert np.allclose(out1_ub, expected_bounds)
    assert np.allclose(out1_lb, expected_bounds)


@pytest.mark.unit
def test_linear_propagate_zero_bounds(data):
    m, *_ = data
    lin1 = Linear(m, 4, 3, bias=False)
    w1 = np.random.rand(3, 4)
    lin1.load_weights(w1)

    x = gp.Variable(m, "x", domain=dim([2, 4]))

    x.up[...] = 0
    x.lo[...] = 0

    out1, _ = lin1(x)

    expected_bounds = np.zeros((2, 3))

    assert np.allclose(
        np.array(out1.records.upper).reshape(out1.shape), expected_bounds
    )
    assert np.allclose(
        np.array(out1.records.lower).reshape(out1.shape), expected_bounds
    )

    lin2 = Linear(m, 4, 3, bias=True)
    b1 = np.random.rand(3)
    lin2.load_weights(w1, b1)

    out2, _ = lin2(x)

    expected_bounds = np.stack((b1, b1))

    assert np.allclose(
        np.array(out2.records.upper).reshape(out1.shape), expected_bounds
    )
    assert np.allclose(
        np.array(out2.records.lower).reshape(out1.shape), expected_bounds
    )


@pytest.mark.unit
def test_linear_return_formulation_result(data):
    m, *_ = data
    lin1 = gp.formulations.Linear(m, 4, 3, bias=False)
    w1 = np.random.rand(3, 4)
    lin1.load_weights(w1)

    x = gp.Variable(m, "x", domain=dim([2, 4]))

    x.up[...] = 0
    x.lo[...] = 0

    result = lin1(x)
    expected_bounds = np.zeros((2, 3))

    assert isinstance(result, FormulationResult), "Expected a FormulationResult object"
    assert isinstance(result.result, gp.Variable), "Expected the output variable"
    assert isinstance(result.variables_created["output"], gp.Variable), (
        "Expected the output variable"
    )
    out1 = result.result
    assert np.allclose(out1.toDense(), expected_bounds)


@pytest.mark.unit
def test_linear_check_str_method(data):
    m, *_ = data
    lin1 = gp.formulations.Linear(m, 4, 3, bias=False)
    w1 = np.random.rand(3, 4)
    lin1.load_weights(w1)

    expected = "Linear(\n  in_features=4\n  out_features=3\n  bias=False\n  weights_loaded=True\n)"
    assert str(lin1) == expected, "Unexpected string"
