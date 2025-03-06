from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import gamspy as gp
import gamspy.formulations.nn.utils as utils
from gamspy.exceptions import ValidationError
from gamspy.math import dim

if TYPE_CHECKING:
    from gamspy import Parameter, Variable


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
    padding : int | tuple[int, int] | Literal["same", "valid"]
        Specifies the amount of padding to apply to the input, by default 0.
        If an integer is provided, that padding is applied to both the height and width.
        If a tuple of two integers is given, the first value determines the padding for the
        top and bottom, while the second value sets the padding for the left and right.
        It is also possible to provide string literals "same" and "valid". "same" pads
        the input so the output has the shape as the input. "valid" is the same as no
        padding.
    bias : bool
        Is bias added after the convolution, by default True
    name_prefix : str | None
        Prefix for names of the GAMS symbols generated, by default None which means
        random prefix. Using same name_prefix in different formulations causes name
        conflicts. Do not use same name_prefix again.

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
        padding: int | tuple[int, int] | Literal["same", "valid"] = 0,
        bias: bool = True,
        name_prefix: str | None = None,
    ):
        if not (isinstance(in_channels, int) and in_channels > 0):
            raise ValidationError("in_channels must be a positive integer")

        if not (isinstance(out_channels, int) and out_channels > 0):
            raise ValidationError("out_channels must be a positive integer")

        _kernel_size = utils._check_tuple_int(kernel_size, "kernel_size")
        _stride = utils._check_tuple_int(stride, "stride")

        if isinstance(padding, str):
            if padding not in {"same", "valid"}:
                raise ValidationError(
                    "padding must be 'same' or 'valid' when it is a string"
                )

            if padding == "same" and _stride != (1, 1):
                raise ValidationError(
                    "'same' padding can only be used with stride=1"
                )

            _padding: tuple[int, int, int, int] | str = (
                (0, 0, 0, 0) if padding == "valid" else "same"
            )

        else:
            _padding = utils._check_padding(padding)

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
        self.weight: Parameter | Variable | None = None
        self.weight_array = None
        self.bias: Parameter | Variable | None = None
        self.bias_array = None

        if name_prefix is None:
            name_prefix = str(uuid.uuid4()).split("-")[0]

        self._name_prefix = name_prefix

    def _encode_infinity(self, x):
        """
        Encode infinity values as complex numbers.
        - Replace -np.inf with 0 - 1j.
        - Replace np.inf with 0 + 1j.
        """
        x = np.where(x == -np.inf, 0 - 1j, x)
        x = np.where(x == np.inf, 0 + 1j, x)
        return x

    def _decode_complex_array(self, z: np.ndarray) -> np.ndarray:
        real = z.real.copy()
        real[z.imag > 0] = np.inf
        real[z.imag < 0] = -np.inf
        return real

    def _propagate_bounds(self, input, output, weight, bias, stride, padding):
        # Extract input bounds
        input_bounds = gp.Parameter(
            self.container,
            domain=dim([2, *input.shape]),
            name=utils._generate_name("p", self._name_prefix, "input_bounds"),
        )
        # lower bounds
        input_bounds[("0",) + tuple(input.domain)] = input.lo[...]
        # upper bounds
        input_bounds[("1",) + tuple(input.domain)] = input.up[...]

        # If the bounds are all zeros (None in GAMSPy parameters);
        # we skip matrix multiplication as it will result in zero values
        if input_bounds.records is None:
            out_bounds_array = np.zeros(output.shape)

            if self.use_bias:
                b = self.bias_array[:, np.newaxis, np.newaxis]
                out_bounds_array = out_bounds_array + b

            out_bounds = gp.Parameter(
                self.container,
                domain=dim(output.shape),
                records=out_bounds_array,
                name=utils._generate_name(
                    "p", self._name_prefix, "output_bounds"
                ),
            )
            output.lo[...] = out_bounds
            output.up[...] = out_bounds

            return

        input_lower, input_upper = input_bounds.toDense()

        # Check if the input bounds contain (-)infinity values
        inf_exists = input_bounds.countNegInf() or input_bounds.countPosInf()
        out_arr_dtype = np.complex128 if inf_exists else np.float64

        if inf_exists:
            # Encode infinity values as complex numbers
            input_lower = self._encode_infinity(input_lower)
            input_upper = self._encode_infinity(input_upper)

        batch, out_channels, h_out, w_out = output.shape
        _, in_channels, h_k, w_k = weight.shape
        stride_y, stride_x = stride
        pad_top, pad_left, pad_bottom, pad_right = padding

        # if any side of the padding is non-zero, we need to pad the input
        if any(padding):
            # Pad the input bounds
            input_lower = np.pad(
                input_lower,
                ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=0,
            )
            input_upper = np.pad(
                input_upper,
                ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=0,
            )

        # ----- Sliding window view -----
        # Create sliding windows for the input, where each window has the same shape as the kernel.
        # This is done separately for the lower and upper bounds of the input.
        # Then, downsample the sliding windows by selecting every `stride_y` step along the height (vertical) axis
        # and every `stride_x` step along the width (horizontal) axis. This reduces the number of windows
        # by skipping positions based on the user-specified strides.
        #
        # After slicing, the axes are transposed to reorder the dimensions. Specifically:
        # - The original window positions (height and width axes) are moved to the end.
        # - The window content (channel and spatial dimensions) is brought forward.
        #
        # This ensures that the windows are processed in a left-to-right, top-to-bottom order,
        # prioritizing horizontal traversal first, which is the desired behavior.

        windows_lower = sliding_window_view(
            input_lower, (batch, in_channels, h_k, w_k)
        )
        windows_upper = sliding_window_view(
            input_upper, (batch, in_channels, h_k, w_k)
        )

        windows_lower = windows_lower[
            :, :, ::stride_y, ::stride_x, :, :, :
        ].transpose(0, 1, 4, 2, 3, 5, 6, 7)
        windows_upper = windows_upper[
            :, :, ::stride_y, ::stride_x, :, :, :
        ].transpose(0, 1, 4, 2, 3, 5, 6, 7)

        # # Reshape windows for vectorized computation
        windows_lower = windows_lower.reshape(
            batch, h_out, w_out, in_channels, h_k, w_k
        )
        windows_upper = windows_upper.reshape(
            batch, h_out, w_out, in_channels, h_k, w_k
        )

        # Split weights into positive and negative parts
        pos_weight = np.maximum(weight, 0)
        neg_weight = np.minimum(weight, 0)

        # Initialize output bounds
        output_lower = np.zeros(
            (batch, out_channels, h_out, w_out), dtype=out_arr_dtype
        )
        output_upper = np.zeros(
            (batch, out_channels, h_out, w_out), dtype=out_arr_dtype
        )

        if bias is None:
            bias = np.zeros(out_channels)

        for c_out in range(out_channels):
            # Lower bound: sum(input_lower * pos_weight + input_upper * neg_weight)
            output_lower[:, c_out, :, :] = (
                np.sum(
                    windows_lower
                    * pos_weight[c_out, np.newaxis, np.newaxis, :, :, :],
                    axis=(3, 4, 5),
                )
                + np.sum(
                    windows_upper
                    * neg_weight[c_out, np.newaxis, np.newaxis, :, :, :],
                    axis=(3, 4, 5),
                )
                + bias[c_out]
            )

            # Upper bound: sum(input_lower * neg_weight + input_upper * pos_weight)
            output_upper[:, c_out, :, :] = (
                np.sum(
                    windows_lower
                    * neg_weight[c_out, np.newaxis, np.newaxis, :, :, :],
                    axis=(3, 4, 5),
                )
                + np.sum(
                    windows_upper
                    * pos_weight[c_out, np.newaxis, np.newaxis, :, :, :],
                    axis=(3, 4, 5),
                )
                + bias[c_out]
            )

        if inf_exists:
            # Decode complex numbers back to real numbers if infinity values were used
            output_lower = self._decode_complex_array(output_lower)
            output_upper = self._decode_complex_array(output_upper)

        out_bounds_array = np.stack([output_lower, output_upper], axis=0)

        out_bounds = gp.Parameter(
            self.container,
            domain=dim([2, *output.shape]),
            records=out_bounds_array,
            name=utils._generate_name("p", self._name_prefix, "output_bounds"),
        )

        output.lo[...] = out_bounds[("0",) + tuple(output.domain)]
        output.up[...] = out_bounds[("1",) + tuple(output.domain)]

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
                self.container,
                domain=dim(expected_shape),
                name=utils._generate_name("v", self._name_prefix, "weight"),
            )

        if self.use_bias and self.bias is None:
            self.bias = gp.Variable(
                self.container,
                domain=dim([self.out_channels]),
                name=utils._generate_name("v", self._name_prefix, "bias"),
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
                    f"bias expected to be ({self.out_channels},)"
                )

        if self.weight is None:
            self.weight = gp.Parameter(
                self.container,
                domain=dim(expected_shape),
                records=weight,
                name=utils._generate_name("p", self._name_prefix, "weight"),
            )
        else:
            self.weight.setRecords(weight)

        self.weight_array = weight

        if self.use_bias:
            if self.bias is None:
                self.bias = gp.Parameter(
                    self.container,
                    domain=dim([self.out_channels]),
                    records=bias,
                    name=utils._generate_name("p", self._name_prefix, "bias"),
                )
            else:
                self.bias.setRecords(bias)

            self.bias_array = bias

        self._state = 1

    def __call__(
        self, input: gp.Parameter | gp.Variable, propagate_bounds: bool = True
    ) -> tuple[gp.Variable, list[gp.Equation]]:
        """
        Forward pass your input, generate output and equations required for
        calculating the convolution. If `propagate_bounds` is True,
        the `input` is of type variable, and `load_weights` was called, then
        the bounds of the input are propagated to the output.

        Parameters
        ----------
        input : gp.Parameter | gp.Variable
                input to the conv layer, must be in shape
                (batch x in_channels x height x width)
        propagate_bounds : bool = True
                If True, propagate bounds of the input to the output.
                Otherwise, the output variable is unbounded.

        """
        if not isinstance(propagate_bounds, bool):
            raise ValidationError("Expected a boolean for propagate_bounds")

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
            name=utils._generate_name("v", self._name_prefix, "output"),
        )

        N, C_out, H_out, W_out = out.domain

        set_out = gp.Equation(
            self.container,
            domain=out.domain,
            name=utils._generate_name("e", self._name_prefix, "set_output"),
        )

        if isinstance(self.padding, str):
            padding = utils._calc_same_padding(self.kernel_size)
        else:
            padding = self.padding

        # expr must have domain N, C_out, H_out, W_out
        top_index = (self.stride[0] * (gp.Ord(H_out) - 1)) - padding[0] + 1
        left_index = (self.stride[1] * (gp.Ord(W_out) - 1)) - padding[1] + 1

        _, _, Hf, Wf = self.weight.domain
        C_in, Hf, Wf, H_in, W_in = utils._next_domains(
            [C_in, Hf, Wf, H_in, W_in],
            out.domain,
        )

        subset = gp.Set(
            self.container,
            domain=[H_out, W_out, Hf, Wf, H_in, W_in],
            name=utils._generate_name("s", self._name_prefix, "conv_subset"),
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
            assert self.bias is not None
            expr = expr + self.bias[C_out]

        set_out[N, C_out, H_out, W_out] = out[N, C_out, H_out, W_out] == expr

        # If propagate_bounds is True, weight is a parameter and input is a variable,
        # we will propagate the bounds of the input to the output
        if (
            propagate_bounds
            and self._state == 1
            and isinstance(input, gp.Variable)
        ):
            self._propagate_bounds(
                input,
                out,
                self.weight_array,
                self.bias_array,
                self.stride,
                padding,
            )

        return out, [set_out]
