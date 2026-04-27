from __future__ import annotations

import numpy as np
import pytest

import gamspy as gp
from gamspy.exceptions import ValidationError
from gamspy.formulations import FormulationResult
from gamspy.formulations.nn import GRU
from gamspy.math import dim

try:
    import torch

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


@pytest.mark.unit
def test_gru_bad_init(data):
    m, *_ = data
    # input_size must be an integer
    with pytest.raises(ValidationError):
        GRU(m, 2.5, 4)
    with pytest.raises(ValidationError):
        GRU(m, "2", 4)
    # hidden_size must be an integer
    with pytest.raises(ValidationError):
        GRU(m, 2, 4.1)
    with pytest.raises(ValidationError):
        GRU(m, 2, "4.1")
    # input_size must be positive
    with pytest.raises(ValidationError):
        GRU(m, -4, 4)
    # hidden_size must be positive
    with pytest.raises(ValidationError):
        GRU(m, 4, -4)


@pytest.mark.unit
def test_gru_load_weights(gru_data):
    m, w_ih, w_hh, b_ih, b_hh = gru_data

    input_size, hidden_size = 4, 3
    gru = GRU(m, input_size, hidden_size)

    bad_w_ih = np.random.rand(3 * hidden_size, 2)
    bad_w_hh = np.random.rand(3 * hidden_size, 4)
    bad_b_ih = np.random.rand(3 * 4)
    bad_b_hh = np.random.rand(3 * 4)

    # missing weight_ih
    with pytest.raises(TypeError):
        gru.load_weights(weight_hh=w_hh)

    # missing weight_hh
    with pytest.raises(TypeError):
        gru.load_weights(weight_ih=w_ih)

    # test bad shape
    with pytest.raises(ValidationError):
        gru.load_weights(bad_w_ih, w_hh, b_ih, b_hh)

    with pytest.raises(ValidationError):
        gru.load_weights(w_ih, bad_w_hh, b_ih, b_hh)

    with pytest.raises(ValidationError):
        gru.load_weights(w_ih, w_hh, bad_b_ih, b_hh)

    with pytest.raises(ValidationError):
        gru.load_weights(w_ih, w_hh, b_ih, bad_b_hh)


@pytest.mark.unit
def test_gru_same_indices(gru_data):
    m, w_ih, w_hh, *_ = gru_data
    gru1 = GRU(m, 4, 3)

    gru1.load_weights(w_ih, w_hh)
    inp = gp.Variable(m, domain=dim([1, 3, 4]))
    _out, _eqs = gru1(inp)
    gru2 = GRU(m, 4, 3)
    gru2.load_weights(w_ih, w_hh)
    _out2, _eqs2 = gru2(inp)


@pytest.mark.unit
def test_gru_check_weights(gru_data):
    m, w_ih, w_hh, b_ih, b_hh = gru_data
    gru1 = GRU(m, 4, 3)

    gru1.load_weights(w_ih, w_hh, b_ih, b_hh)

    out_w_ih = np.vstack(
        [gru1.w_ih["r"].toDense(), gru1.w_ih["z"].toDense(), gru1.w_ih["n"].toDense()]
    )
    out_w_hh = np.vstack(
        [gru1.w_hh["r"].toDense(), gru1.w_hh["z"].toDense(), gru1.w_hh["n"].toDense()]
    )
    out_b_ih = np.hstack(
        [gru1.b_ih["r"].toDense(), gru1.b_ih["z"].toDense(), gru1.b_ih["n"].toDense()]
    )
    out_b_hh = np.hstack(
        [gru1.b_hh["r"].toDense(), gru1.b_hh["z"].toDense(), gru1.b_hh["n"].toDense()]
    )

    assert np.allclose(w_ih, out_w_ih)
    assert np.allclose(w_hh, out_w_hh)
    assert np.allclose(b_ih, out_b_ih)
    assert np.allclose(b_hh, out_b_hh)


@pytest.mark.unit
def test_gru_reloading_weights(gru_data):
    m, w_ih, w_hh, b_ih, b_hh = gru_data
    gru1 = GRU(m, 4, 3)

    gru1.load_weights(w_ih, w_hh, b_ih, b_hh)

    w_ih = np.ones((3 * 3, 4))
    w_hh = np.ones((3 * 3, 3))
    b_ih = np.ones(3 * 3)

    gru1.load_weights(w_ih, w_hh, b_ih)

    out_w_ih = np.vstack(
        [gru1.w_ih["r"].toDense(), gru1.w_ih["z"].toDense(), gru1.w_ih["n"].toDense()]
    )
    out_w_hh = np.vstack(
        [gru1.w_hh["r"].toDense(), gru1.w_hh["z"].toDense(), gru1.w_hh["n"].toDense()]
    )
    out_b_ih = np.hstack(
        [gru1.b_ih["r"].toDense(), gru1.b_ih["z"].toDense(), gru1.b_ih["n"].toDense()]
    )
    out_b_hh = np.hstack(
        [gru1.b_hh["r"].toDense(), gru1.b_hh["z"].toDense(), gru1.b_hh["n"].toDense()]
    )

    assert np.allclose(w_ih, out_w_ih)
    assert np.allclose(w_hh, out_w_hh)
    assert np.allclose(b_ih, out_b_ih)
    assert np.allclose(np.zeros(3 * 3), out_b_hh)


@pytest.mark.unit
def test_gru_call_bad(gru_data):
    m, w_ih, w_hh, b_ih, b_hh = gru_data
    gru1 = GRU(m, 4, 3)
    inp = gp.Variable(m, domain=dim([1, 3, 4]))
    # requires initialization before
    with pytest.raises(ValidationError):
        gru1(inp)

    gru1.load_weights(w_ih, w_hh, b_ih, b_hh)

    # needs 3 dimensions, (batch, time_step, in_feature)
    bad_inp = gp.Variable(m, domain=dim([1, 3]))
    with pytest.raises(ValidationError):
        gru1(bad_inp)

    # input_feature must match to 4
    bad_inp_2 = gp.Variable(m, domain=dim([1, 3, 3]))
    with pytest.raises(ValidationError):
        gru1(bad_inp_2)

    # shape must be (1, 3)
    bad_h0 = gp.Parameter(m, domain=dim([1, 4]))
    with pytest.raises(ValidationError):
        gru1(inp, bad_h0)


@pytest.mark.unit
def test_gru_check_intermediate_symbols(gru_data):
    m, w_ih, w_hh, b_ih, b_hh = gru_data
    gru1 = GRU(m, 4, 3)

    gru1.load_weights(w_ih, w_hh, b_ih, b_hh)

    x = gp.Variable(m, "x", domain=dim([1, 3, 4]))

    x.up[...] = -5
    x.lo[...] = 5

    result = gru1(x)

    assert isinstance(result, FormulationResult), "Expected a FormulationResult object"
    assert isinstance(result.result, gp.Variable), "Expected the output variable"
    assert isinstance(result.variables_created["output"], gp.Variable), (
        "Expected the output variable"
    )

    list_of_parameters = [
        "w_ih_r",
        "w_ih_z",
        "w_ih_n",
        "w_hh_r",
        "w_hh_z",
        "w_hh_n",
        "b_ih_r",
        "b_ih_z",
        "b_ih_n",
        "b_hh_r",
        "b_hh_z",
        "b_hh_n",
    ]
    for par in list_of_parameters:
        assert isinstance(result.parameters_created[par], gp.Parameter), (
            f"Expected `{par}` parameter."
        )
    list_of_equations = ["reset_gate", "update_gate", "new_gate", "set_output"]
    for eqn in list_of_equations:
        assert isinstance(result.equations_created[eqn], gp.Equation), (
            f"Expected `{eqn}` equation."
        )
    list_of_vars = ["r_gate", "z_gate", "n_gate", "output"]
    for var in list_of_vars:
        assert isinstance(result.variables_created[var], gp.Variable), (
            f"Expected `{var}` Variable."
        )


@pytest.mark.unit
def test_gru_check_str_method(gru_data):
    m, w_ih, w_hh, b_ih, b_hh = gru_data
    gru1 = GRU(m, 4, 3)
    gru1.load_weights(w_ih, w_hh, b_ih, b_hh)

    expected = "GRU(\n  input_size=4\n  hidden_size=3\n  weights_loaded=True\n)"
    assert str(gru1) == expected, "Unexpected string"


@pytest.mark.unit
def test_gru_one_time_step(gru_data):
    m, w_ih, w_hh, b_ih, b_hh = gru_data
    gru1 = GRU(m, 4, 3)

    gru1.load_weights(w_ih, w_hh, b_ih, b_hh)

    x = gp.Variable(m, "x", domain=dim([1, 1, 4]))

    result = gru1(x)
    expected = (1, 1, 3)

    assert expected == tuple(len(i) for i in result.result.domain), (
        "Unexpected shape of out variable"
    )


@pytest.mark.unit
def test_gru_n_time_step(gru_data):
    m, w_ih, w_hh, b_ih, b_hh = gru_data
    gru1 = GRU(m, 4, 3)

    gru1.load_weights(w_ih, w_hh, b_ih, b_hh)

    x = gp.Variable(m, "x", domain=dim([2, 3, 4]))

    result = gru1(x)
    expected = (2, 3, 3)

    assert expected == tuple(len(i) for i in result.result.domain), (
        "Unexpected shape of out variable"
    )


@pytest.mark.skipif(TORCH_AVAILABLE is False, reason="Requires PyTorch installed")
@pytest.mark.unit
def test_gru_check_lag_correctness(gru_data):
    m, *_ = gru_data

    time_steps = np.linspace(0, np.pi, 5)
    sine_wave = np.sin(time_steps).reshape(1, 5, 1)

    torch_gru = torch.nn.GRU(input_size=1, hidden_size=2, batch_first=True)
    x_tensor = torch.tensor(sine_wave, dtype=torch.float32)

    with torch.no_grad():
        torch_out, _ = torch_gru(x_tensor)

    w_ih = torch_gru.weight_ih_l0.detach().numpy()
    w_hh = torch_gru.weight_hh_l0.detach().numpy()
    b_ih = torch_gru.bias_ih_l0.detach().numpy()
    b_hh = torch_gru.bias_hh_l0.detach().numpy()

    m = gp.Container()

    gams_gru = GRU(m, 1, 2)
    gams_gru.load_weights(w_ih, w_hh, b_ih, b_hh)

    x = gp.Parameter(m, name="x_in", domain=dim([1, 5, 1]), records=sine_wave)
    h_out = gams_gru(x).result

    model = gp.Model(
        m,
        name="gru_test",
        equations=m.getEquations(),
        problem="NLP",
        sense="min",
        objective=gp.Sum(h_out.domain, h_out),
    )

    model.solve()

    assert np.allclose(torch_out[0, -1, :].numpy(), h_out.toDense()[0, -1, :]), (
        "Unexpected output"
    )


@pytest.mark.unit
def test_gru_with_h0_time_step_one(gru_data):
    m, *_ = gru_data

    gru1 = GRU(m, 1, 1)

    w_ih = np.ones((3, 1))
    w_hh = np.ones((3, 1))

    gru1.load_weights(w_ih, w_hh)

    x = gp.Variable(m, "x", domain=dim([1, 1, 1]))
    x.lo[...] = 0
    x.up[...] = 5

    h0 = gp.Parameter(m, "h0", domain=dim([1, 1]), records=np.ones((1, 1)))
    h_out = gru1(x, h0=h0).result

    model = gp.Model(
        m,
        name="sample_gru_time_step_one",
        equations=m.getEquations(),
        problem="NLP",
        sense="min",
        objective=gp.Sum(h_out.domain, h_out),
    )

    model.solve()

    import math

    r = 1 / (1 + math.exp(-1))
    z = 1 / (1 + math.exp(-1))
    n = math.tanh(r * 1)
    expected = (1 - z) * n + z * h0.toDense()

    assert np.allclose(h_out.toDense(), expected), "Unexpected h_out value."


@pytest.mark.unit
def test_gru_with_h0_time_step_k(gru_data):
    m, *_ = gru_data

    gru1 = GRU(m, 1, 1)

    w_ih = np.ones((3, 1))
    w_hh = np.ones((3, 1))

    gru1.load_weights(w_ih, w_hh)

    x = gp.Parameter(m, "x", domain=dim([1, 3, 1]), records=np.ones((1, 3, 1)))

    h0 = gp.Parameter(m, "h0", domain=dim([1, 1]), records=np.ones((1, 1)))
    h_out = gru1(x, h0=h0).result

    model = gp.Model(
        m,
        name="sample_gru_time_step_k",
        equations=m.getEquations(),
        problem="NLP",
        sense="min",
        objective=gp.Sum(h_out.domain, h_out),
    )

    model.solve()

    import math

    def get_output(x_seq, h_prev):
        expected_outputs = []

        def sigmoid(par):
            return 1 / (1 + math.exp(-par))

        for _, x_t in enumerate(x_seq):
            r = sigmoid(x_t * 1.0 + h_prev * 1.0)
            z = sigmoid(x_t * 1.0 + h_prev * 1.0)

            n = math.tanh(x_t * 1.0 + r * (h_prev * 1.0))

            h_curr = (1 - z) * n + z * h_prev
            expected_outputs.append(h_curr)

            h_prev = h_curr

        return np.array(expected_outputs).reshape(1, len(x_seq), 1)

    expected = get_output(x.toDense().flatten(), h_prev=float(h0.toDense()[0][0]))

    assert np.allclose(h_out.toDense(), expected), "Unexpected h_out value."
