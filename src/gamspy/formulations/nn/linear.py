from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import gamspy as gp
import gamspy.formulations.nn.utils as utils
from gamspy.exceptions import ValidationError
from gamspy.math import dim

if TYPE_CHECKING:
    from gamspy import Parameter, Variable


class Linear:
    """
    Formulation generator for Linear layer in GAMS.

    Parameters
    ----------
    container : Container
        Container that will contain the new variable and equations.
    in_features : int
        Input feature size
    out_features : int
        Output feature size
    bias : bool = True
        Should bias be added after linear transformation, by Default: True
    name_prefix : str | None
        Prefix for generated GAMSPy symbols, by default None which means
        random prefix. Using the same name_prefix in different formulations causes name
        conflicts. Do not use the same name_prefix again.

    Examples
    --------
    >>> import gamspy as gp
    >>> import numpy as np
    >>> from gamspy.math import dim
    >>> m = gp.Container()
    >>> l1 = gp.formulations.Linear(m, 128, 64)
    >>> w = np.random.rand(64, 128)
    >>> b = np.random.rand(64)
    >>> l1.load_weights(w, b)
    >>> x = gp.Variable(m, "x", domain=dim([10, 128]))
    >>> y, set_y = l1(x)
    >>> [d.name for d in y.domain]
    ['DenseDim10_1', 'DenseDim64_1']

    """

    def __init__(
        self,
        container: gp.Container,
        in_features: int,
        out_features: int,
        bias: bool = True,
        name_prefix: str | None = None,
    ):
        if not isinstance(in_features, int) or in_features <= 0:
            raise ValidationError("in_features must be a positive integer")

        if not isinstance(out_features, int) or out_features <= 0:
            raise ValidationError("out_features must be a positive integer")

        if not isinstance(bias, bool):
            raise ValidationError("bias must be a boolean")

        self.container = container
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self._state = 0
        self.weight: Parameter | Variable | None = None
        self.weight_array = None
        self.bias: Parameter | Variable | None = None
        self.bias_array = None

        if name_prefix is None:
            name_prefix = gp.utils._get_unique_name()

        self._name_prefix = name_prefix

    def load_weights(
        self, weight: np.ndarray, bias: np.ndarray | None = None
    ) -> None:
        """
        Mark Linear as parameter and load weights from NumPy arrays.
        After this is called `make_variable` cannot be called. Use this
        when you already have the weights of your Linear layer.

        Parameters
        ----------
        weight : np.ndarray
               Linear layer weights in shape (out_features x in_features)
        bias : np.ndarray | None
               Linear layer bias in shape (out_features, ), only required when
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

        if len(weight.shape) != 2:
            raise ValidationError(
                f"expected 2D input for weight (got {len(weight.shape)}D input)"
            )

        expected_shape = (
            self.out_features,
            self.in_features,
        )
        if weight.shape != expected_shape:
            raise ValidationError(
                f"weight expected to be in shape {expected_shape}"
            )

        if bias is not None:
            if len(bias.shape) != 1:
                raise ValidationError(
                    f"expected 1D input for bias (got {len(bias.shape)}D input)"
                )

            if bias.shape[0] != self.out_features:
                raise ValidationError(
                    f"bias expected to be in shape ({self.out_features},)"
                )

        if self.weight is None:
            self.weight = gp.Parameter(
                self.container,
                name=utils._generate_name("p", self._name_prefix, "weight"),
                domain=dim(expected_shape),
                records=weight,
            )
        else:
            self.weight.setRecords(weight)
        self.weight_array = weight

        if self.use_bias:
            if self.bias is None:
                self.bias = gp.Parameter(
                    self.container,
                    name=utils._generate_name("p", self._name_prefix, "bias"),
                    domain=dim([self.out_features]),
                    records=bias,
                )
            else:
                self.bias.setRecords(bias)

            self.bias_array = bias

        self._state = 1

    def make_variable(self) -> None:
        """
        Mark Linear layer as variable. After this is called `load_weights`
        cannot be called. Use this when you need to learn the weights
        of your linear layer in your optimization model.

        This does not initialize the weights, it is highly recommended
        that you set initial values to `weight` and `bias` variables.
        """
        if self._state == 1:
            raise ValidationError(
                "make_variable cannot be used after calling load_weights"
            )

        expected_shape = (
            self.out_features,
            self.in_features,
        )

        if self.weight is None:
            self.weight = gp.Variable(
                self.container,
                name=utils._generate_name("v", self._name_prefix, "weight"),
                domain=dim(expected_shape),
            )

        if self.use_bias and self.bias is None:
            self.bias = gp.Variable(
                self.container,
                name=utils._generate_name("v", self._name_prefix, "bias"),
                domain=dim([self.out_features]),
            )

        self._state = 2

    def __call__(
        self, input: gp.Parameter | gp.Variable, propagate_bounds: bool = True
    ) -> tuple[gp.Variable, list[gp.Equation]]:
        """
        Forward pass your input, generate output and equations required for
        calculating the linear transformation. If `propagate_bounds` is True,
        the `input` is of type variable, and `load_weights` was called, then
        the bounds of the input are propagated to the output.

        Parameters
        ----------
        input : gp.Parameter | gp.Variable
                input to the linear layer, must be in shape
                (* x in_features)
        propagate_bounds : bool = True
                If True, propagate bounds of the input to the output.
                Otherwise, the output variable is unbounded.
        """
        if not isinstance(propagate_bounds, bool):
            raise ValidationError("propagate_bounds should be a boolean.")

        if self.weight is None:
            raise ValidationError(
                "You must call load_weights or make_variable first before using the Linear"
            )

        if len(input.domain) == 0:
            raise ValidationError(
                "expected an input with at least 1 dimension"
            )

        if len(input.domain[-1]) != self.in_features:
            raise ValidationError("in_features does not match")

        expr = input @ self.weight.t()

        if self.bias is not None:
            expr = expr + self.bias[expr.domain[-1]]

        out = gp.Variable(
            self.container,
            name=utils._generate_name("v", self._name_prefix, "output"),
            domain=expr.domain,
        )

        set_out = gp.Equation(
            self.container,
            name=utils._generate_name("e", self._name_prefix, "set_output"),
            domain=out.domain,
        )

        set_out[...] = out == expr

        # If propagate_bounds is True, weight is a parameter and input is a variable,
        # we will propagate the bounds of the input to the output
        if (
            propagate_bounds
            and self._state == 1
            and isinstance(input, gp.Variable)
        ):
            x_bounds = gp.Parameter(
                self.container,
                name=utils._generate_name(
                    "p", self._name_prefix, "input_bounds"
                ),
                domain=dim([2, *input.shape]),
            )
            x_bounds[("0",) + tuple(input.domain)] = input.lo[...]
            x_bounds[("1",) + tuple(input.domain)] = input.up[...]

            # If the bounds are all zeros (None in GAMSPy parameters);
            # we skip matrix multiplication as it will result in zero values
            if x_bounds.records is None:
                out_bounds_array = np.zeros(out.shape)

                if self.use_bias:
                    out_bounds_array = out_bounds_array + self.bias_array

                out_bounds = gp.Parameter(
                    self.container,
                    name=utils._generate_name(
                        "p", self._name_prefix, "output_bounds"
                    ),
                    domain=dim(out.shape),
                    records=out_bounds_array,
                )
                out.lo[...] = out_bounds
                out.up[...] = out_bounds

                return out, [set_out]

            x_lb, x_ub = x_bounds.toDense()

            # To deal with infinity values in the input bounds, we convert them into complex numbers
            # where if the value is -inf, we convert it to 0 - 1j
            # and if the value is inf, we convert it to 0 + 1j
            x_lb = np.where(x_lb == -np.inf, 0 - 1j, x_lb)
            x_ub = np.where(x_ub == np.inf, 0 + 1j, x_ub)

            # get the positive and negative weights separately
            w_pos = np.maximum(self.weight_array, 0)
            w_neg = np.minimum(self.weight_array, 0)

            lo_out = (x_lb @ w_pos.T) + (x_ub @ w_neg.T)
            up_out = (x_ub @ w_pos.T) + (x_lb @ w_neg.T)

            def _decode_complex_number(z: np.complex128) -> float:
                """
                Decode complex number to real number.
                5 + 0j -> 5
                3 + 1j -> inf
                7 - 3j -> -inf
                """
                # If imaginary part is zero, return real part
                if z.imag == 0:
                    return z.real
                # If imaginary part is positive, return positive infinity
                elif z.imag > 0:
                    return np.inf
                # If imaginary part is negative, return negative infinity
                else:
                    return -np.inf

            lo_out = np.vectorize(_decode_complex_number)(lo_out)
            up_out = np.vectorize(_decode_complex_number)(up_out)

            if self.use_bias:
                lo_out = lo_out + self.bias_array
                up_out = up_out + self.bias_array

            out_bounds_array = np.stack([lo_out, up_out], axis=0)

            out_bounds = gp.Parameter(
                self.container,
                name=utils._generate_name(
                    "p", self._name_prefix, "output_bounds"
                ),
                domain=dim([2, *out.shape]),
                records=out_bounds_array,
            )

            out.lo[...] = out_bounds[("0",) + tuple(out.domain)]
            out.up[...] = out_bounds[("1",) + tuple(out.domain)]

        return out, [set_out]
