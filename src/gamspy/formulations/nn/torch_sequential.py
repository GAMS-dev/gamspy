from __future__ import annotations

from typing import TYPE_CHECKING

import gamspy as gp
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    import torch


def convert_linear(
    m: gp.Container, layer: torch.nn.Linear
) -> gp.formulations.Linear:
    has_bias = layer.bias is not None
    l = gp.formulations.Linear(
        m,
        in_features=layer.in_features,
        out_features=layer.out_features,
        bias=has_bias,
    )
    l.load_weights(
        layer.weight.numpy(), layer.bias.numpy() if has_bias else None
    )
    return l


def convert_conv1d(
    m: gp.Container, layer: torch.nn.Conv1d
) -> gp.formulations.Conv1d:
    if layer.dilation[0] != 1:
        raise ValidationError("Conv1d is not supported when dilation is not 1")

    if layer.groups != 1:
        raise ValidationError("Conv1d is not supported when groups is not 1")

    if layer.padding_mode != "zeros":
        raise ValidationError(
            "Conv1d is only supported with padding_mode zeros"
        )

    has_bias = layer.bias is not None
    l = gp.formulations.Conv1d(
        m,
        in_channels=layer.in_channels,
        out_channels=layer.out_channels,
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        bias=has_bias,
    )

    l.load_weights(
        layer.weight.numpy(), layer.bias.numpy() if has_bias else None
    )
    return l


def convert_conv2d(
    m: gp.Container, layer: torch.nn.Conv2d
) -> gp.formulations.Conv2d:
    if layer.dilation[0] != 1 or layer.dilation[-1] != 1:
        raise ValidationError("Conv2d is not supported when dilation is not 1")

    if layer.groups != 1:
        raise ValidationError("Conv2d is not supported when groups is not 1")

    if layer.padding_mode != "zeros":
        raise ValidationError(
            "Conv1d is only supported with padding_mode zeros"
        )

    has_bias = layer.bias is not None
    l = gp.formulations.Conv2d(
        m,
        in_channels=layer.in_channels,
        out_channels=layer.out_channels,
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        bias=has_bias,
    )

    l.load_weights(
        layer.weight.numpy(), layer.bias.numpy() if has_bias else None
    )
    return l


def convert_relu(m: gp.Container, layer: torch.nn.ReLU):
    return gp.math.relu_with_binary_var


def convert_pool2d(
    m: gp.Container, layer: torch.nn.MaxPool2d | torch.nn.AvgPool2d
):
    clz = layer.__class__.__name__
    if clz == "MaxPool2d":
        dilation = layer.dilation
        if isinstance(dilation, int):
            dilation = (dilation,)

        if dilation[0] != 1 or dilation[-1] != 1:
            raise ValidationError(
                "Pool2d is not supported when dilation is not 1"
            )

        if layer.return_indices is True:
            raise ValidationError(
                "Pool2d is not supported when return_indices is True"
            )

    if layer.ceil_mode is True:
        raise ValidationError("Pool2d is not supported when ceil_mode is True")

    if clz == "MaxPool2d":
        return gp.formulations.MaxPool2d(
            m,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
        )
    else:
        return gp.formulations.AvgPool2d(
            m,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
        )


_DEFAULT_CONVERTERS = {
    "Linear": convert_linear,
    "Conv1d": convert_conv1d,
    "Conv2d": convert_conv2d,
    "ReLU": convert_relu,
    "MaxPool2d": convert_pool2d,
    "AvgPool2d": convert_pool2d,
}


class TorchSequential:
    """
    Formulation generator for Sequential Layer from PyTorch.
    This is a convenience formulation that builds upon other
    formulations.

    Parameters
    ----------
    container : Container
        Container that will contain the new variable and equations.
    network : torch.nn.Sequential
        Sequential network that will be translated to GAMSPy
    layer_converters : dict | None
        You can change default layer converters or add support for
        not implemented layers through this dictionary. Key is the
        class name as string, and value expects a function that returns
        GAMSPy formulation given container and the PyTorch layer.

    Examples
    --------
    >>> import gamspy as gp
    >>> from gamspy.math import dim
    >>> def embed():
    ...     try:
    ...         import torch
    ...     except ModuleNotFoundError as e:
    ...         print("[10, 4, 30, 30]")
    ...         return
    ...     m = gp.Container()
    ...     model = torch.nn.Sequential(
    ...         torch.nn.Conv2d(3, 4, 3, bias=True),
    ...         torch.nn.ReLU(),
    ...         torch.nn.Conv2d(4, 4, 3, bias=False, padding=1),
    ...     )
    ...     x = gp.Variable(m, domain=dim([10, 3, 32, 32]))
    ...     seq_formulation = gp.formulations.TorchSequential(m, model)
    ...     y, eqs = seq_formulation(x)
    ...     print([len(d) for d in y.domain])
    >>> embed()
    [10, 4, 30, 30]

    """

    def __init__(
        self,
        container: gp.Container,
        network: torch.nn.Sequential,
        layer_converters: dict | None = None,
    ):
        try:
            import torch
        except ModuleNotFoundError as e:
            raise ValidationError(
                "You must first install PyTorch to use this functionality."
            ) from e

        self._layer_converters = _DEFAULT_CONVERTERS.copy()
        if layer_converters is not None:
            self._layer_converters.update(layer_converters)

        with torch.no_grad():
            self.layers = [
                self._convert_layer(container, layer) for layer in network
            ]

    def _convert_layer(self, container: gp.Container, layer):
        clz = layer.__class__.__name__
        if clz not in self._layer_converters:
            raise ValidationError(f"Formulation for {clz} not implemented!")

        l = self._layer_converters[clz](container, layer)
        return l

    def __call__(
        self, input: gp.Variable
    ) -> tuple[gp.Variable, list[gp.Equation]]:
        out = input
        equations = []
        for layer in self.layers:
            out, layer_eqs = layer(out)
            equations.extend(layer_eqs)

        return out, equations
