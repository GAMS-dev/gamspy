from __future__ import annotations

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
    name_prefix : str | None
        Prefix for generated GAMSPy symbols, by default None which means
        random prefix. Using the same name_prefix in different formulations causes name
        conflicts. Do not use the same name_prefix again.

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
        name_prefix: str | None = None,
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

        if name_prefix is None:
            name_prefix = gp.utils._get_unique_name()

        self._name_prefix = name_prefix

    def _set_bounds(
        self,
        input: gp.Parameter | gp.Variable,
        subset: gp.Set,
    ) -> tuple[gp.Parameter, gp.Parameter]:
        # Extract batch and channel dimensions from input domain
        N, C = input.domain[:2]
        H_out, W_out, Hf, Wf, H_in, W_in = subset.domain

        # Create subset2 mapping output positions to input positions
        subset2 = gp.Set(
            self.container,
            domain=[H_out, W_out, H_in, W_in],
            name=utils._generate_name(
                "s", self._name_prefix, "in_out_matching_2"
            ),
        )

        subset2[H_out, W_out, H_in, W_in] = gp.Sum(
            [Hf, Wf], subset[H_out, W_out, Hf, Wf, H_in, W_in]
        )

        # Initialize parameters for bounds
        lb = gp.Parameter(
            self.container,
            domain=[N, C, H_out, W_out],
            name=utils._generate_name("p", self._name_prefix, "output_lb"),
        )
        ub = gp.Parameter(
            self.container,
            domain=[N, C, H_out, W_out],
            name=utils._generate_name("p", self._name_prefix, "output_ub"),
        )

        # Calculate upper/lower bounds based on input type
        if isinstance(input, gp.Variable):
            # Use variable bounds if input is a variable
            ub[...] = gp.Smax(
                gp.Domain(H_in, W_in).where[subset2[H_out, W_out, H_in, W_in]],
                input.up[N, C, H_in, W_in],
            )
            lb[...] = gp.Smin(
                gp.Domain(H_in, W_in).where[subset2[H_out, W_out, H_in, W_in]],
                input.lo[N, C, H_in, W_in],
            )
        else:
            # Use parameter values directly if input is a parameter
            ub[...] = gp.Smax(
                gp.Domain(H_in, W_in).where[subset2[H_out, W_out, H_in, W_in]],
                input[N, C, H_in, W_in],
            )
            lb[...] = gp.Smin(
                gp.Domain(H_in, W_in).where[subset2[H_out, W_out, H_in, W_in]],
                input[N, C, H_in, W_in],
            )

        # 0 padding on the edges can shift the minimum and max value
        kh, kw = self.kernel_size
        ph, pw = self.padding
        scale = ((kh - ph) * (kw - pw)) / (kh * kw)

        lb.where[lb > 0] = lb * scale
        ub.where[ub < 0] = ub * scale

        return lb, ub

    def __call__(
        self,
        input: gp.Parameter | gp.Variable,
        propagate_bounds: bool = True,
    ) -> tuple[gp.Variable, list[gp.Equation]]:
        """
        Forward pass your input, generate output and equations required for
        calculating the average pooling. Unlike the min or max pooling avg
        pooling does not require binary variables or the big-M formulation.
        if propagate_bounds is True, it will also set the bounds for the
        output variable based on the input.
        Returns the output variable and the list of equations required for
        the avg pooling formulation.

        Parameters
        ----------
        input : gp.Parameter | gp.Variable
                input to the max pooling 2d layer, must be in shape
                (batch x in_channels x height x width)
        propagate_bounds: bool
                If True, it will set the bounds for the output variable
                based on the input.
                Default value: True

        Returns
        -------
        tuple[gp.Variable, list[gp.Equation]]

        """
        if not isinstance(input, (gp.Parameter, gp.Variable)):
            raise ValidationError("Expected a parameter or a variable input")

        if not isinstance(propagate_bounds, bool):
            raise ValidationError("Expected a boolean for propagate_bounds")

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

        out_var = gp.Variable(
            self.container,
            domain=dim([len(N), len(C_in), h_out, w_out]),
            name=utils._generate_name("v", self._name_prefix, "output"),
        )

        N, C, H_out, W_out = out_var.domain

        set_out = gp.Equation(
            self.container,
            domain=out_var.domain,
            name=utils._generate_name("e", self._name_prefix, "set_output"),
        )

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

        subset = gp.Set(
            self.container,
            name=utils._generate_name(
                "s", self._name_prefix, "in_out_matching_1"
            ),
            domain=[H_out, W_out, Hf, Wf, H_in, W_in],
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

        # Set variable bounds if propagate_bounds is True
        if propagate_bounds:
            lb, ub = self._set_bounds(input, subset)
            out_var.lo[...] = lb
            out_var.up[...] = ub

        return out_var, [set_out]
