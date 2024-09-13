from __future__ import annotations

import uuid

import numpy as np

import gamspy as gp
import gamspy.formulations.nn.utils as utils
from gamspy.exceptions import ValidationError
from gamspy.math import dim, next_alias


class Conv2d:
    """
    Formulation generator for 2D Convolution symbol in GAMS. It can
    be used to embed convolutional layers of trained neural networks
    in your problem. It can also be used to embed convolutional layers
    when you need weights as variables.

    Parameters
    ----------
    container : Container
        Container that will contain the new variable and equations.
    in_channel : int
        Number of channels in the input
    out_channel : int
        Number of channels in the output
    kernel_size : int | tuple[int, int]
        Filter size
    stride : int | tuple[int, int]
        Stride in the convolution, by default 1
    padding : int | tuple[int, int]
        Amount of padding to be added to input, by default 0
    bias : bool
        Is bias added after the convolution, by default True

    Examples
    --------
    >>> import gamspy as gp
    >>> import numpy as np
    >>> from gamspy.math import dim
    >>> w1 = np.random.rand(2, 1, 3, 3)
    >>> b1 = np.random.rand(2)
    >>> m = gp.Container()
    >>> # in_channels=1, out_channels=2, kernel_size=3x3
    >>> conv1 = gp.formulations.Conv2d(m, 1, 2, 3)
    >>> conv1.load_weights(w1, b1)
    >>> # 10 images, 1 channel, 24 by 24
    >>> inp = gp.Variable(m, domain=dim((10, 1, 24, 24)))
    >>> out, eqs = conv1(inp)
    >>> type(out)
    <class 'gamspy._symbols.variable.Variable'>
    >>> [len(x) for x in out.domain]
    [10, 2, 22, 22]
    """

    def __init__(
        self,
        container: gp.Container,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        bias: bool = True,
    ):
        if not (isinstance(in_channels, int) and in_channels > 0):
            raise ValidationError("in_channels must be a positive integer")

        if not (isinstance(out_channels, int) and out_channels > 0):
            raise ValidationError("out_channels must be a positive integer")

        _kernel_size = utils._check_tuple_int(kernel_size, "kernel_size")
        _stride = utils._check_tuple_int(stride, "stride")
        _padding = utils._check_tuple_int(padding, "padding", allow_zero=True)

        if not isinstance(bias, bool):
            raise ValidationError("bias must be a boolean")

        self.container = container
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _kernel_size
        self.stride = _stride
        self.padding = _padding
        self.use_bias = bias

        self._state = 0
        self.weight = None
        self.bias = None

    def make_variable(self) -> None:
        """
        Mark Conv2d as variable. After this is called `load_weights`
        cannot be called. Use this when you need to learn the weights
        of your convolutional layer in your optimization model.

        This does not initialize the weights, it is highly recommended
        that you set initial values to `weight` and `bias` variables.
        """
        if self._state == 1:
            raise ValidationError(
                "make_variable cannot be used after calling load_weights"
            )

        expected_shape = (
            self.out_channels,
            self.in_channels,
            self.kernel_size[0],
            self.kernel_size[1],
        )

        if self.weight is None:
            self.weight = gp.Variable(
                self.container, domain=dim(expected_shape)
            )

        if self.use_bias and self.bias is None:
            self.bias = gp.Variable(
                self.container,
                domain=dim([self.out_channels]),
            )

        self._state = 2

    def load_weights(
        self, weight: np.ndarray, bias: np.ndarray | None = None
    ) -> None:
        """
        Mark Conv2d as parameter and load weights from NumPy arrays.
        After this is called `make_variable` cannot be called. Use this
        when you already have the weights of your convolutional layer.

        Parameters
        ----------
        weight : np.ndarray
                 Conv2d layer weights in shape
                 (out_channels x in_channels x kernel_size[0] x kernel_size[1])
        bias : np.ndarray | None
               Conv2d layer bias in shape (out_channels, ), only required when
               bias=True during initialization

        """
        if self._state == 2:
            raise ValidationError(
                "load_weights cannot be used after calling make_variable"
            )

        if self.use_bias is False and bias is not None:
            raise ValidationError(
                "bias must be None since bias was set to False during initialization"
            )

        if self.use_bias is True and bias is None:
            raise ValidationError("bias must be provided")

        if len(weight.shape) != 4:
            raise ValidationError(
                f"expected 4D input for weight (got {len(weight.shape)}D input)"
            )

        expected_shape = (
            self.out_channels,
            self.in_channels,
            self.kernel_size[0],
            self.kernel_size[1],
        )
        if weight.shape != expected_shape:
            raise ValidationError(f"weight expected to be {expected_shape}")

        if bias is not None:
            if len(bias.shape) != 1:
                raise ValidationError(
                    f"expected 1D input for bias (got {len(bias.shape)}D input)"
                )

            if bias.shape != (self.out_channels,):
                raise ValidationError(
                    f"weight expected to be ({self.out_channels},)"
                )

        if self.weight is None:
            self.weight = gp.Parameter(
                self.container, domain=dim(expected_shape), records=weight
            )
        else:
            self.weight.setRecords(weight)

        if self.use_bias:
            if self.bias is None:
                self.bias = gp.Parameter(
                    self.container,
                    domain=dim([self.out_channels]),
                    records=bias,
                )
            else:
                self.bias.setRecords(bias)

        self._state = 1

    def __call__(
        self, input: gp.Parameter | gp.Variable
    ) -> tuple[gp.Variable, list[gp.Equation]]:
        """
        Forward pass your input, generate output and equations required for
        calculating the convolution.

        Parameters
        ----------
        input : gp.Parameter | gp.Variable
                input to the conv layer, must be in shape
                (batch x in_channels x height x width)


        """
        if self.weight is None:
            raise ValidationError(
                "You must call load_weights or make_variable first before using the Conv2d"
            )

        if len(input.domain) != 4:
            raise ValidationError(
                f"expected 4D input (got {len(input.domain)}D input)"
            )

        N, C_in, H_in, W_in = input.domain

        if len(C_in) != self.in_channels:
            raise ValidationError("in_channels does not match")

        h_in = len(H_in)
        w_in = len(W_in)

        h_out, w_out = utils._calc_hw(
            self.padding, self.kernel_size, self.stride, h_in, w_in
        )

        out = gp.Variable(
            self.container,
            domain=dim([len(N), self.out_channels, h_out, w_out]),
        )

        N, C_out, H_out, W_out = out.domain

        set_out = gp.Equation(self.container, domain=out.domain)

        # expr must have domain N, C_out, H_out, W_out
        top_index = (
            (self.stride[0] * (gp.Ord(H_out) - 1)) - self.padding[0] + 1
        )
        left_index = (
            (self.stride[1] * (gp.Ord(W_out) - 1)) - self.padding[1] + 1
        )

        _, _, Hf, Wf = self.weight.domain

        while C_in in out.domain:
            C_in = next_alias(C_in)

        while Hf in out.domain or Hf == C_in:
            Hf = next_alias(Hf)

        while Wf in out.domain or Wf in [C_in, Hf]:
            Wf = next_alias(Wf)

        while H_in in out.domain or H_in in [C_in, Hf, Wf]:
            H_in = next_alias(H_in)

        while W_in in out.domain or W_in in [C_in, Hf, Wf, H_in]:
            W_in = next_alias(W_in)

        name = "ds_" + str(uuid.uuid4()).split("-")[0]
        subset = gp.Set(
            self.container, name, domain=[H_out, W_out, Hf, Wf, H_in, W_in]
        )
        subset[
            H_out,
            W_out,
            Hf,
            Wf,
            H_in,
            W_in,
        ].where[
            (gp.Ord(H_in) == (top_index + gp.Ord(Hf) - 1))
            & (gp.Ord(W_in) == (left_index + gp.Ord(Wf) - 1))
        ] = True

        expr = gp.Sum(
            [C_in],
            gp.Sum(
                subset[H_out, W_out, Hf, Wf, H_in, W_in],
                input[N, C_in, H_in, W_in] * self.weight[C_out, C_in, Hf, Wf],
            ),
        )

        if self.use_bias:
            expr = expr + self.bias[C_out]

        set_out[N, C_out, H_out, W_out] = out[N, C_out, H_out, W_out] == expr

        return out, [set_out]
