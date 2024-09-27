from __future__ import annotations

import uuid

import gamspy as gp
import gamspy.formulations.nn.utils as utils
from gamspy.exceptions import ValidationError
from gamspy.math import dim


class AvgPool2d:
    """
    Formulation generator for 2D Avg Pooling in GAMS.

    Parameters
    ----------
    container : Container
        Container that will contain the new variable and equations.
    kernel_size : int | tuple[int, int]
        Filter size
    stride : int | tuple[int, int] | None
        Stride in the avg pooling, it is equal to kernel_size if not provided
    padding : int | tuple[int, int]
        Amount of padding to be added to input, by default 0

    Examples
    --------
    >>> import gamspy as gp
    >>> from gamspy.math import dim
    >>> m = gp.Container()
    >>> # 2x2 avg pooling
    >>> ap1 = gp.formulations.AvgPool2d(m, (2, 2))
    >>> inp = gp.Variable(m, domain=dim((10, 1, 24, 24)))
    >>> out, eqs = ap1(inp)
    >>> type(out)
    <class 'gamspy._symbols.variable.Variable'>
    >>> [len(x) for x in out.domain]
    [10, 1, 12, 12]

    """

    def __init__(
        self,
        container: gp.Container,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int = 0,
    ):
        _kernel_size = utils._check_tuple_int(kernel_size, "kernel_size")
        if stride is None:
            stride = kernel_size

        _stride = utils._check_tuple_int(stride, "stride")
        _padding = utils._check_tuple_int(padding, "padding", allow_zero=True)

        self.container = container
        self.kernel_size = _kernel_size
        self.stride = _stride
        self.padding = _padding

    def _get_bounds(
        self, input: gp.Parameter | gp.Variable
    ) -> tuple[gp.Parameter, gp.Parameter]:
        # this can be done more fine-grained
        N, C, H, W = input.domain
        lb = gp.Parameter(self.container, domain=[N, C])
        ub = gp.Parameter(self.container, domain=[N, C])

        if isinstance(input, gp.Variable):
            ub[...] = gp.Smax([H, W], input.up[N, C, H, W])
            lb[...] = gp.Smin([H, W], input.lo[N, C, H, W])
        else:
            ub[...] = gp.Smax([H, W], input[N, C, H, W])
            lb[...] = gp.Smin([H, W], input[N, C, H, W])

        # 0 padding on the edges can shift the minimum and max value
        scale = (
            (self.kernel_size[0] - self.padding[0])
            * (self.kernel_size[1] - self.padding[1])
        ) / (self.kernel_size[0] * self.kernel_size[1])

        lb[N, C].where[lb[N, C] > 0] = lb[N, C] * scale
        ub[N, C].where[ub[N, C] < 0] = ub[N, C] * scale

        return lb, ub

    def __call__(
        self, input: gp.Parameter | gp.Variable
    ) -> tuple[gp.Variable, list[gp.Equation]]:
        """
        Forward pass your input, generate output and equations required for
        calculating the average pooling. Unlike the min or max pooling avg
        pooling does not require binary variables or the big-M formulation.
        Returns the output variable and the list of equations required for
        the avg pooling formulation

        Parameters
        ----------
        input : gp.Parameter | gp.Variable
                input to the max pooling 2d layer, must be in shape
                (batch x in_channels x height x width)

        Returns
        -------
        tuple[gp.Variable, list[gp.Equation]]

        """
        if not isinstance(input, (gp.Parameter, gp.Variable)):
            raise ValidationError("Expected a parameter or a variable input")

        if len(input.domain) != 4:
            raise ValidationError(
                f"expected 4D input (got {len(input.domain)}D input)"
            )

        N, C_in, H_in, W_in = input.domain

        h_in = len(H_in)
        w_in = len(W_in)

        h_out, w_out = utils._calc_hw(
            self.padding, self.kernel_size, self.stride, h_in, w_in
        )

        lb, ub = self._get_bounds(input)
        out_var = gp.Variable(
            self.container, domain=dim([len(N), len(C_in), h_out, w_out])
        )
        out_var.lo[...] = lb
        out_var.up[...] = ub

        N, C, H_out, W_out = out_var.domain

        set_out = gp.Equation(self.container, domain=out_var.domain)

        # expr must have domain N, C, H_out, W_out
        top_index = (
            (self.stride[0] * (gp.Ord(H_out) - 1)) - self.padding[0] + 1
        )
        left_index = (
            (self.stride[1] * (gp.Ord(W_out) - 1)) - self.padding[1] + 1
        )
        coeff = 1 / (self.kernel_size[0] * self.kernel_size[1])

        Hf, Wf = gp.math._generate_dims(self.container, self.kernel_size)
        Hf, Wf, H_in, W_in = utils._next_domains(
            [Hf, Wf, H_in, W_in], out_var.domain
        )

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
            subset[H_out, W_out, Hf, Wf, H_in, W_in],
            input[N, C, H_in, W_in] * coeff,
        )

        set_out[...] = out_var[N, C, H_out, W_out] == expr
        return out_var, [set_out]
