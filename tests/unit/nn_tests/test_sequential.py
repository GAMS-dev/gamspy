from __future__ import annotations

import pytest

import gamspy as gp
from gamspy.exceptions import ValidationError
from gamspy.formulations.nn import TorchSequential
from gamspy.math import dim

try:
    import torch

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


@pytest.mark.skipif(TORCH_AVAILABLE is False, reason="Requires PyTorch installed")
@pytest.mark.unit
def test_sequential_layer(data):
    m, *_ = data

    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, bias=True),
        torch.nn.ReLU(),
        torch.nn.Conv2d(4, 4, 3, bias=False, padding=1),
        torch.nn.LeakyReLU(negative_slope=0.02),
        torch.nn.MaxPool2d((2, 2)),
        torch.nn.Linear(15, 15, bias=False),
        torch.nn.ReLU(),
        torch.nn.Linear(15, 20, bias=True),
        torch.nn.AvgPool2d((3, 4)),
    )

    x = gp.Variable(m, domain=dim([10, 3, 32, 32]))

    seq_formulation = gp.formulations.TorchSequential(m, model)
    y, _eqs = seq_formulation(x)
    expected_shape = (10, 4, 5, 5)
    assert tuple(len(d) for d in y.domain) == expected_shape

    model2 = torch.nn.Sequential(
        torch.nn.Conv1d(1, 1, 3, padding=1, bias=True),
        torch.nn.ReLU(),
        torch.nn.Conv1d(1, 1, 3, bias=False),
        torch.nn.ReLU(),
    )
    x2 = gp.Variable(m, domain=dim([10, 1, 8]))

    seq_formulation_2 = gp.formulations.TorchSequential(m, model2)
    output = seq_formulation_2(x2)
    assert "0.output" in output.variables_created
    assert "1.y_gte_x" in output.equations_created
    assert "1.binary" in output.variables_created
    assert "2.set_output" in output.equations_created
    assert "3.y_lte_x_1" in output.equations_created

    y2, _eqs2 = output
    expected_shape = (10, 1, 6)
    assert tuple(len(d) for d in y2.domain) == expected_shape


@pytest.mark.skipif(TORCH_AVAILABLE is False, reason="Requires PyTorch installed")
@pytest.mark.unit
def test_sequential_layer_not_implemented(data):
    m, *_ = data

    bad_inputs = [
        {"dilation": 2},
        {"groups": 2},
        {"padding_mode": "reflect"},
    ]

    for bad_input in bad_inputs:
        model = torch.nn.Sequential(
            torch.nn.Conv2d(4, 4, 3, bias=True, **bad_input),
        )
        with pytest.raises(ValidationError):
            TorchSequential(m, model)

        model2 = torch.nn.Sequential(
            torch.nn.Conv1d(2, 2, 3, padding=1, bias=True, **bad_input),
        )
        with pytest.raises(ValidationError):
            TorchSequential(m, model2)

    model = torch.nn.Sequential(torch.nn.MaxPool2d((2, 2), dilation=2))
    with pytest.raises(ValidationError):
        TorchSequential(m, model)

    model = torch.nn.Sequential(torch.nn.MaxPool2d((2, 2), return_indices=True))
    with pytest.raises(ValidationError):
        TorchSequential(m, model)

    model = torch.nn.Sequential(torch.nn.AvgPool2d((2, 2), ceil_mode=True))
    with pytest.raises(ValidationError):
        TorchSequential(m, model)


@pytest.mark.skipif(TORCH_AVAILABLE is False, reason="Requires PyTorch installed")
@pytest.mark.unit
def test_sequential_layer_custom_layer(data):
    m, *_ = data
    model = torch.nn.Sequential(
        torch.nn.Hardswish(),
    )
    with pytest.raises(ValidationError):
        TorchSequential(m, model)

    def hardswish(x: gp.Variable):
        y = x.container.addVariable(domain=x.domain)
        set_y = x.container.addEquation(domain=y.domain)
        # Of course not the correct implementation!
        # Otherwise would be included in the library
        set_y[...] = y[...] = x[...] * gp.math.errorf(x[...])
        return y, [set_y]

    def hardswish_converter(m: gp.Container, layer):
        return hardswish

    TorchSequential(m, model, layer_converters={"Hardswish": hardswish_converter})

    # or you can pick a different implementation
    model = torch.nn.Sequential(
        torch.nn.ReLU(),
    )

    def relu_converter(m: gp.Container, layer):
        return gp.math.activation.relu_with_complementarity_var

    seq_form = TorchSequential(m, model, layer_converters={"ReLU": relu_converter})

    x = gp.Variable(m)
    _, eqs = seq_form(x)
    assert len(eqs) == 2

    seq_form = TorchSequential(m, model)
    x = gp.Variable(m)
    output = seq_form(x)
    assert len(output) == 2
    assert len(output.equations_created) == 3
    assert "0.output" in output.variables_created
    assert "0.binary" in output.variables_created
    assert "0.y_gte_x" in output.equations_created
    assert "0.y_lte_x_1" in output.equations_created
    assert "0.y_lte_x_2" in output.equations_created


@pytest.mark.skipif(TORCH_AVAILABLE is False, reason="Requires PyTorch installed")
@pytest.mark.unit
def test_sequential_layer_with_matches(data):
    m, *_ = data
    model = torch.nn.Sequential(
        torch.nn.ReLU(),
    )

    def relu_converter(m: gp.Container, layer):
        return gp.math.activation.relu_with_equilibrium

    seq_form = TorchSequential(m, model, layer_converters={"ReLU": relu_converter})

    x = gp.Variable(m)
    _, matches, eqs = seq_form(x)
    assert len(eqs) == 0
    assert len(matches) == 1


@pytest.mark.unit
def test_sequential_layer_bad_model(data):
    m, *_ = data

    # must be torch.nn.Sequence
    with pytest.raises(ValidationError):
        TorchSequential(m, "model")


@pytest.mark.skipif(TORCH_AVAILABLE is False, reason="Requires PyTorch installed")
@pytest.mark.unit
def test_sequential_layer_with_rnn(data):
    m, *_ = data
    model = torch.nn.Sequential(
        torch.nn.RNN(
            input_size=2, hidden_size=4, nonlinearity="relu", batch_first=True
        ),
        torch.nn.Linear(in_features=4, out_features=1),
    )
    seq_form = TorchSequential(m, model)

    x = gp.Variable(m, name="x_in", domain=dim([1, 5, 2]))
    output = seq_form(x)

    assert "0.set_output" in output.equations_created
    assert "0.set_pre_act" in output.equations_created
    assert "0.y_lte_x_2" in output.equations_created
    assert "0.output" in output.variables_created
    assert "1.output" in output.variables_created

    expected_shape = (1, 5, 1)
    assert tuple(len(d) for d in output.result.domain) == expected_shape
