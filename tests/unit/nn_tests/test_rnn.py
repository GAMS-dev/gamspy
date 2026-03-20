from __future__ import annotations

import numpy as np
import pytest

import gamspy as gp
from gamspy.exceptions import ValidationError
from gamspy.formulations.nn import RNN
from gamspy.formulations.result import FormulationResult
from gamspy.math import dim


@pytest.mark.unit
def test_rnn_bad_init(data):
    m, *_ = data
    # input_size must be an integer
    with pytest.raises(ValidationError):
        RNN(m, 2.5, 4, "tanh")
    with pytest.raises(ValidationError):
        RNN(m, "2", 4, "tanh")
    # hidden_size must be an integer
    with pytest.raises(ValidationError):
        RNN(m, 2, 4.1, "tanh")
    with pytest.raises(ValidationError):
        RNN(m, 2, "4.1", "tanh")
    # input_size must be positive
    with pytest.raises(ValidationError):
        RNN(m, -4, 4, "tanh")
    # hidden_size must be positive
    with pytest.raises(ValidationError):
        RNN(m, 4, -4, "tanh")
    # activation must be one of ["tanh", "relu", "linear"]
    with pytest.raises(ValidationError):
        RNN(m, 4, 4, activation=False)


@pytest.mark.unit
def test_rnn_load_weights(rnn_data):
    m, w_ih, w_hh, b_ih, b_hh = rnn_data

    input_size, hidden_size = 4, 3
    rnn1 = RNN(m, input_size, hidden_size, activation="tanh")

    bad_w_ih = np.random.rand(hidden_size, 2)
    bad_w_hh = np.random.rand(hidden_size, 4)
    bad_b_ih = np.random.rand(4)
    bad_b_hh = np.random.rand(4)

    # missing weight_ih
    with pytest.raises(TypeError):
        rnn1.load_weights(weight_hh=w_hh)

    # missing weight_hh
    with pytest.raises(TypeError):
        rnn1.load_weights(weight_ih=w_ih)

    # test bad shape
    with pytest.raises(ValidationError):
        rnn1.load_weights(bad_w_ih, w_hh, b_ih, b_hh)

    with pytest.raises(ValidationError):
        rnn1.load_weights(w_ih, bad_w_hh, b_ih, b_hh)

    with pytest.raises(ValidationError):
        rnn1.load_weights(w_ih, w_hh, bad_b_ih, b_hh)

    with pytest.raises(ValidationError):
        rnn1.load_weights(w_ih, w_hh, b_ih, bad_b_hh)


@pytest.mark.unit
def test_rnn_same_indices(rnn_data):
    m, w_ih, w_hh, *_ = rnn_data
    rnn1 = RNN(m, 4, 3, activation="tanh")

    rnn1.load_weights(w_ih, w_hh)
    inp = gp.Variable(m, domain=dim([1, 3, 4]))
    _out, _eqs = rnn1(inp)
    rnn2 = RNN(m, 4, 3, activation="tanh")
    rnn2.load_weights(w_ih, w_hh)
    _out2, _eqs2 = rnn2(inp)


@pytest.mark.unit
def test_rnn_reloading_weights(rnn_data):
    m, w_ih, w_hh, b_ih, b_hh = rnn_data
    rnn1 = RNN(m, 4, 3, activation="tanh")

    rnn1.load_weights(w_ih, w_hh, b_ih, b_hh)

    w_ih = np.ones((3, 4))
    w_hh = np.ones((3, 3))
    b_ih = np.ones(3)
    rnn1.load_weights(w_ih, w_hh, b_ih)

    assert np.allclose(w_ih, rnn1.weight_ih.toDense())
    assert np.allclose(w_hh, rnn1.weight_hh.toDense())
    assert np.allclose(b_ih, rnn1.bias_ih.toDense())
    assert np.allclose(np.zeros(3), rnn1.bias_hh.toDense())


@pytest.mark.unit
def test_rnn_call_bad(rnn_data):
    m, w_ih, w_hh, b_ih, b_hh = rnn_data
    rnn1 = RNN(m, 4, 3, activation="tanh")
    inp = gp.Variable(m, domain=dim([1, 3, 4]))
    # requires initialization before
    with pytest.raises(ValidationError):
        rnn1(inp)

    rnn1.load_weights(w_ih, w_hh, b_ih, b_hh)

    # needs 3 dimensions, (batch, time_step, in_feature)
    bad_inp = gp.Variable(m, domain=dim([1, 3]))
    with pytest.raises(ValidationError):
        rnn1(bad_inp)

    # input_feature must match to 4
    bad_inp_2 = gp.Variable(m, domain=dim([1, 3, 3]))
    with pytest.raises(ValidationError):
        rnn1(bad_inp_2)

    # shape must be (1, 3)
    bad_h0 = gp.Parameter(m, domain=dim([1, 4]))
    with pytest.raises(ValidationError):
        rnn1(inp, bad_h0)

    with pytest.raises(ValidationError):
        rnn1(inp, propagate_bounds="True")


@pytest.mark.unit
def test_rnn_return_formulation_result(rnn_data):
    m, w_ih, w_hh, b_ih, b_hh = rnn_data
    rnn1 = RNN(m, 4, 3, activation="tanh")

    rnn1.load_weights(w_ih, w_hh, b_ih, b_hh)

    x = gp.Variable(m, "x", domain=dim([1, 3, 4]))

    x.up[...] = 0
    x.lo[...] = 0

    result = rnn1(x)

    assert isinstance(result, FormulationResult), "Expected a FormulationResult object"
    assert isinstance(result.result, gp.Variable), "Expected the output variable"
    assert isinstance(result.variables_created["output"], gp.Variable), (
        "Expected the output variable"
    )


@pytest.mark.unit
def test_rnn_check_relu_intermediate_symbols(rnn_data):
    m, w_ih, w_hh, b_ih, b_hh = rnn_data
    rnn1 = RNN(m, 4, 3, activation="relu")

    rnn1.load_weights(w_ih, w_hh, b_ih, b_hh)

    x = gp.Variable(m, "x", domain=dim([1, 3, 4]))

    x.up[...] = -5
    x.lo[...] = 5

    result = rnn1(x)

    assert isinstance(result, FormulationResult), "Expected a FormulationResult object"
    assert isinstance(result.result, gp.Variable), "Expected the output variable"
    assert isinstance(result.variables_created["output"], gp.Variable), (
        "Expected the output variable"
    )

    list_of_parameters = ["input_bounds", "out_bounds", "relu_bounds"]
    for par in list_of_parameters:
        assert isinstance(result.parameters_created[par], gp.Parameter), (
            f"Expected `{par}` parameter."
        )
    list_of_equations = ["set_pre_act", "y_gte_x", "y_lte_x_1", "y_lte_x_2"]
    for eqn in list_of_equations:
        assert isinstance(result.equations_created[eqn], gp.Equation), (
            f"Expected `{eqn}` equation."
        )
    list_of_vars = ["pre_act", "binary"]
    for var in list_of_vars:
        assert isinstance(result.variables_created[var], gp.Variable), (
            f"Expected `{var}` Variable."
        )


@pytest.mark.unit
def test_rnn_one_time_step(rnn_data):
    m, w_ih, w_hh, b_ih, b_hh = rnn_data
    rnn1 = RNN(m, 4, 3, activation="tanh")

    rnn1.load_weights(w_ih, w_hh, b_ih, b_hh)

    x = gp.Variable(m, "x", domain=dim([1, 1, 4]))

    result = rnn1(x)
    expected = (1, 1, 3)

    assert expected == tuple(len(i) for i in result.result.domain), (
        "Unexpected shape of out variable"
    )


@pytest.mark.unit
def test_rnn_n_time_step(rnn_data):
    m, w_ih, w_hh, b_ih, b_hh = rnn_data
    rnn1 = RNN(m, 4, 3, activation="relu")

    rnn1.load_weights(w_ih, w_hh, b_ih, b_hh)

    x = gp.Variable(m, "x", domain=dim([2, 3, 4]))

    result = rnn1(x)
    expected = (2, 3, 3)

    assert expected == tuple(len(i) for i in result.result.domain), (
        "Unexpected shape of out variable"
    )


@pytest.mark.unit
def test_rnn_check_lag_correctness(rnn_data):
    m, *_ = rnn_data

    # input_seq: [1,2,3,4,5], prediction: 6
    rnn1 = RNN(m, 1, 1, activation="linear")

    w_ih = np.array([[0.4]])
    w_hh = np.array([[1.0]])

    rnn1.load_weights(w_ih, w_hh)

    x = gp.Parameter(
        m, "x", domain=dim([1, 5, 1]), records=np.array([[[1], [2], [3], [4], [5]]])
    )
    h_out = rnn1(x).result

    model = gp.Model(
        m,
        name="sample_rnn",
        equations=m.getEquations(),
        problem="LP",
        sense="min",
        objective=gp.Sum(h_out.domain, h_out),
    )

    model.solve()

    assert h_out.l.records.iloc[-1]["level"] == 6.0, "Expected 6"


@pytest.mark.unit
def test_rnn_check_tanh_correctness(rnn_data):
    """
    tanh should swing between -1 and 1
    """
    m, *_ = rnn_data

    rnn1 = RNN(m, 1, 1, activation="tanh")

    w_ih = np.array([[1.0]])
    w_hh = np.array([[1.0]])

    rnn1.load_weights(w_ih, w_hh)

    x = gp.Parameter(
        m, "x", domain=dim([1, 4, 1]), records=np.array([[[1], [-2], [3], [-1]]])
    )
    h_out = rnn1(x).result

    model = gp.Model(
        m,
        name="sample_rnn_tanh",
        equations=m.getEquations(),
        problem="NLP",
        sense="min",
        objective=gp.Sum(h_out.domain, h_out),
    )

    model.solve()

    assert h_out.l.records["level"].max() <= 1, "This should not be more than 1"
    assert h_out.l.records["level"].min() >= -1, "This should not be less than -1"


@pytest.mark.unit
def test_rnn_check_relu_correctness(rnn_data):
    """
    relu should always be >= 0
    """
    m, *_ = rnn_data

    rnn1 = RNN(m, 1, 1, activation="relu")

    w_ih = np.array([[1.0]])
    w_hh = np.array([[1.0]])

    rnn1.load_weights(w_ih, w_hh)

    x = gp.Parameter(
        m, "x", domain=dim([1, 4, 1]), records=np.array([[[1], [-2], [3], [-1]]])
    )
    h_out = rnn1(x).result

    model = gp.Model(
        m,
        name="sample_rnn_relu",
        equations=m.getEquations(),
        problem="MIP",
        sense="min",
        objective=gp.Sum(h_out.domain, h_out),
    )

    model.solve()

    assert h_out.l.records["level"].min() > 0, "Expected only positive level"


@pytest.mark.unit
def test_rnn_check_bound_propagation_set_0(rnn_data):
    """
    no bias and lo/up is set to 0
    """
    m, *_ = rnn_data

    rnn1 = RNN(m, 1, 1, activation="linear")

    w_ih = np.array([[1.0]])
    w_hh = np.array([[1.0]])

    rnn1.load_weights(w_ih, w_hh)

    x = gp.Variable(m, "x", domain=dim([1, 4, 1]))
    x.lo[...] = 0
    x.up[...] = 0

    h_out = rnn1(x).result

    h_lo = h_out.toDense("lower")
    h_up = h_out.toDense("upper")

    assert np.allclose(h_lo, np.zeros((1, 4))), "Expecting an array of 0s"
    assert np.allclose(h_up, np.zeros((1, 4))), "Expecting an array of 0s"

    """
    with bias and lo/up is set to 0
    """

    b_ih = np.random.randint(10, size=1)
    b_hh = np.random.randint(10, size=1)

    rnn1.load_weights(w_ih, w_hh, b_ih, b_hh)

    h_out = rnn1(x).result

    h_lo = h_out.toDense("lower")
    h_up = h_out.toDense("upper")

    expected = (b_ih + b_hh) * np.array([[1], [2], [3], [4]])

    assert np.allclose(h_lo, expected), "Wrong Lower bounds"
    assert np.allclose(h_up, expected), "Wrong Upper bounds"


@pytest.mark.unit
def test_rnn_check_bound_propagation_set_default(rnn_data):
    """
    No bias and lo=-inf, up=inf. With bias the bounds will be the same.
    """
    m, *_ = rnn_data
    act_rnn = {
        "linear": {"lo": np.full((1, 4, 1), -np.inf), "up": np.full((1, 4, 1), np.inf)},
        "tanh": {"lo": -1 * np.ones((1, 4, 1)), "up": np.ones((1, 4, 1))},
        "relu": {"lo": np.zeros((1, 4, 1)), "up": np.full((1, 4, 1), np.inf)},
    }

    w_ih = np.random.random(size=(1, 1))
    w_hh = np.random.random(size=(1, 1))
    x = gp.Variable(m, domain=dim([1, 4, 1]))
    p = gp.Parameter(m, domain=dim([2, 1, 4, 1]))

    for act, expected in act_rnn.items():
        rnn = RNN(m, 1, 1, activation=act)
        rnn.load_weights(w_ih, w_hh)
        h_out = rnn(x).result
        p[("0",) + tuple(h_out.domain)] = h_out.lo[...]
        p[("1",) + tuple(h_out.domain)] = h_out.up[...]

        p_lo, p_up = p.toDense()
        assert np.allclose(p_lo, expected["lo"]), f"Wrong lower bound for {act}"
        assert np.allclose(p_up, expected["up"]), f"Wrong upper bound for {act}"


@pytest.mark.unit
def test_rnn_check_bound_propagation_set_unique_bounds(
    rnn_data, calculate_expected_rnn_bounds
):
    m, *_ = rnn_data
    act_rnn = ["linear", "tanh", "relu"]

    w_ih = np.random.random(size=(1, 1))
    w_hh = np.random.random(size=(1, 1))
    b_ih = np.random.random(size=(1))
    b_hh = np.random.random(size=(1))

    x = gp.Variable(m, domain=dim([1, 4, 1]))
    p = gp.Parameter(m, domain=dim([2, 1, 4, 1]))

    xlb = np.random.randint(-5, 1, (1, 4, 1))
    xub = np.random.randint(1, 5, (1, 4, 1))

    x_lb = gp.Parameter(m, "x_lb", domain=dim([1, 4, 1]), records=xlb)
    x_ub = gp.Parameter(m, "x_ub", domain=dim([1, 4, 1]), records=xub)

    x.lo[...] = x_lb[...]
    x.up[...] = x_ub[...]

    for act in act_rnn:
        rnn = RNN(m, 1, 1, activation=act)
        rnn.load_weights(w_ih, w_hh, b_ih, b_hh)
        result = rnn(x)
        h_out = result.result
        p[("0",) + tuple(h_out.domain)] = h_out.lo[...]
        p[("1",) + tuple(h_out.domain)] = h_out.up[...]

        actual = np.stack([*p.toDense()], axis=0)
        expected_pre, expected_out = calculate_expected_rnn_bounds(
            xlb, xub, w_ih, w_hh, b_ih, b_hh, activation=act
        )
        np.testing.assert_allclose(
            actual, expected_out, err_msg=f"Wrong bounds for out variable in {act}."
        )
        if act == "relu":
            relu_bounds = result.parameters_created["relu_bounds"]
            np.testing.assert_allclose(
                np.stack([*relu_bounds.toDense()], axis=0),
                expected_pre,
                err_msg="Wrong bounds for `pre_act` variable in relu.",
            )


@pytest.mark.unit
def test_rnn_check_bound_propagation_with_h0(rnn_data, calculate_expected_rnn_bounds):
    m, *_ = rnn_data
    act_rnn = ["linear", "tanh", "relu"]

    w_ih = np.random.random(size=(1, 1))
    w_hh = np.random.random(size=(1, 1))
    b_ih = np.random.random(size=(1))
    b_hh = np.random.random(size=(1))

    time_step = 1
    x = gp.Variable(m, domain=dim([1, time_step, 1]))
    p = gp.Parameter(m, domain=dim([2, 1, time_step, 1]))

    xlb = np.random.randint(-5, 1, (1, time_step, 1))
    xub = np.random.randint(1, 5, (1, time_step, 1))

    x_lb = gp.Parameter(m, "x_lb", domain=dim([1, time_step, 1]), records=xlb)
    x_ub = gp.Parameter(m, "x_ub", domain=dim([1, time_step, 1]), records=xub)

    x.lo[...] = x_lb[...]
    x.up[...] = x_ub[...]

    h0_array = np.random.random(size=(1, 1))
    h0 = gp.Parameter(m, "h0", domain=dim([1, 1]), records=h0_array)

    for act in act_rnn:
        rnn = RNN(m, 1, 1, activation=act)
        rnn.load_weights(w_ih, w_hh, b_ih, b_hh)
        result = rnn(x, h0=h0)
        h_out = result.result
        p[("0",) + tuple(h_out.domain)] = h_out.lo[...]
        p[("1",) + tuple(h_out.domain)] = h_out.up[...]

        actual = np.stack([*p.toDense()], axis=0)
        expected_pre, expected_out = calculate_expected_rnn_bounds(
            xlb, xub, w_ih, w_hh, b_ih, b_hh, activation=act, h0=h0_array
        )
        np.testing.assert_allclose(
            actual, expected_out, err_msg=f"Wrong bounds for out variable in {act}."
        )
        if act == "relu":
            relu_bounds = result.parameters_created["relu_bounds"]
            np.testing.assert_allclose(
                np.stack([*relu_bounds.toDense()], axis=0),
                expected_pre,
                err_msg="Wrong bounds for `pre_act` variable in relu.",
            )
