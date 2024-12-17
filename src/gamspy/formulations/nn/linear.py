from __future__ import annotations

import uuid

import numpy as np

import gamspy as gp
from gamspy.exceptions import ValidationError
from gamspy.math import dim
import uuid


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
        self.weight = None
        self.bias = None

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
            name = "lin_weight" + str(uuid.uuid4()).split("-")[0]
            self.weight = gp.Parameter(
                self.container,
                name,
                domain=dim(expected_shape),
                records=weight,
            )
        else:
            self.weight.setRecords(weight)

        if self.use_bias:
            if self.bias is None:
                name = "lin_bias" + str(uuid.uuid4()).split("-")[0]
                self.bias = gp.Parameter(
                    self.container,
                    name,
                    domain=dim([self.out_features]),
                    records=bias,
                )
            else:
                self.bias.setRecords(bias)

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
                self.container, domain=dim(expected_shape)
            )

        if self.use_bias and self.bias is None:
            self.bias = gp.Variable(
                self.container,
                domain=dim([self.out_features]),
            )

        self._state = 2

    def __call__(
        self, input: gp.Parameter | gp.Variable, propagate_bounds: bool = True
    ) -> tuple[gp.Variable, list[gp.Equation]]:
        """
        Forward pass your input, generate output and equations required for
        calculating the linear transformation.

        Parameters
        ----------
        input : gp.Parameter | gp.Variable
                input to the linear layer, must be in shape
                (* x in_features)
        """
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

        name = "lin_eq" + str(uuid.uuid4()).split("-")[0]
        vname = "lin_var" + str(uuid.uuid4()).split("-")[0]

        out = gp.Variable(self.container, vname, domain=expr.domain)

        set_out = gp.Equation(self.container, name, domain=out.domain)

        set_out[...] = out == expr

        if propagate_bounds:
            x_lb = input.container.addParameter(f"x_lb_{uuid.uuid4()}".replace("-", "_"), domain=input.domain)
            x_lb[...] = input.lo[...]
            x_lb[...].where[x_lb[...] == "-inf"] = -1e9

            x_ub = input.container.addParameter(f"x_ub_{uuid.uuid4()}".replace("-", "_"), domain=input.domain)
            x_ub[...] = input.up[...]
            x_ub[...].where[x_ub[...] == "inf"] = 1e9

            w_pos = input.container.addParameter(f"w_pos_{uuid.uuid4()}".replace("-", "_"), domain=self.weight.domain)
            w_pos[...] = self.weight.where[self.weight[...] > 0]

            w_neg = input.container.addParameter(f"w_neg_{uuid.uuid4()}".replace("-", "_"), domain=self.weight.domain)
            w_neg[...] = self.weight.where[self.weight[...] < 0]

            lo_out_expr = (x_lb @ w_pos.t()) + (x_ub @ w_neg.t())
            up_out_expr = (x_ub @ w_pos.t()) + (x_lb @ w_neg.t())

            if self.bias is not None:
                lo_out_expr = lo_out_expr + self.bias[lo_out_expr.domain[-1]]
                up_out_expr = up_out_expr + self.bias[up_out_expr.domain[-1]]

            out.lo[...] = lo_out_expr
            out.up[...] = up_out_expr

        return out, [set_out]
