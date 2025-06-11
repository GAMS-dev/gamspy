from __future__ import annotations

from typing import Literal

import gamspy as gp
import gamspy.formulations.nn.utils as utils
from gamspy.exceptions import ValidationError
from gamspy.math import dim


class _MPool2d:
    def __init__(
        self,
        sense: Literal["min", "max"],
        container: gp.Container,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int = 0,
        name_prefix: str | None = None,
    ):
        # Validate pooling type
        if sense not in ("min", "max"):
            raise ValidationError("_MPool2d expects min or max")

        # Convert kernel_size, stride, and padding to tuples
        _kernel_size = utils._check_tuple_int(kernel_size, "kernel_size")
        if stride is None:
            stride = kernel_size  # Default stride = kernel size

        _stride = utils._check_tuple_int(stride, "stride")
        _padding = utils._check_tuple_int(padding, "padding", allow_zero=True)

        # Store configurations
        self.container = container
        self.kernel_size = _kernel_size
        self.stride = _stride
        self.padding = _padding
        self.sense = sense

        if name_prefix is None:
            name_prefix = gp.utils._get_unique_name()

        self._name_prefix = name_prefix

    def _set_bounds_and_big_M(
        self,
        input: gp.Parameter | gp.Variable,
        default_big_m: int,
        subset: gp.Set,
        out_var: gp.Variable,
    ) -> tuple[gp.Parameter, gp.Parameter, gp.Parameter]:
        # Extract batch and channel dimensions from input domain
        N, C = input.domain[:2]
        H_out, W_out, Hf, Wf, H_in, W_in = subset.domain

        # Create subset2 mapping output positions to input positions
        subset2 = gp.Set(
            self.container,
            name=utils._generate_name(
                "s", self._name_prefix, "in_out_matching_2"
            ),
            domain=[H_out, W_out, H_in, W_in],
        )
        subset2[H_out, W_out, H_in, W_in] = gp.Sum(
            [Hf, Wf], subset[H_out, W_out, Hf, Wf, H_in, W_in]
        )

        # Initialize parameters for bounds and Big-M
        big_m_par = gp.Parameter(
            self.container,
            name=utils._generate_name("p", self._name_prefix, "bigM_1"),
            domain=[N, C, H_out, W_out],
        )
        lb = gp.Parameter(
            self.container,
            name=utils._generate_name("p", self._name_prefix, "output_lb"),
            domain=[N, C, H_out, W_out],
        )
        ub = gp.Parameter(
            self.container,
            name=utils._generate_name("p", self._name_prefix, "output_ub"),
            domain=[N, C, H_out, W_out],
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

        # Set output variable bounds
        out_var.lo[...] = lb
        out_var.up[...] = ub

        # Calculate Big-M values and handle infinity
        big_m_par[N, C, H_out, W_out] = ub - lb
        big_m_par[...].where[big_m_par[...] == "inf"] = default_big_m

        return big_m_par

    def __call__(
        self,
        input: gp.Parameter | gp.Variable,
        big_m: int = 1000,
        propagate_bounds: bool = True,
    ) -> tuple[gp.Variable, list[gp.Equation]]:
        # User input validation
        if not isinstance(input, (gp.Parameter, gp.Variable)):
            raise ValidationError("Expected a parameter or a variable input")

        if not isinstance(propagate_bounds, bool):
            raise ValidationError("Expected a boolean for propagate_bounds")

        if len(input.domain) != 4:
            raise ValidationError(
                f"expected 4D input (got {len(input.domain)}D input)"
            )

        # Extract dimensions from input (Batch, Channel, Height, Width)
        N, C, H_in, W_in = input.domain
        h_in = len(H_in)
        w_in = len(W_in)

        # Calculate output dimensions using padding, kernel, and stride
        h_out, w_out = utils._calc_hw(
            self.padding, self.kernel_size, self.stride, h_in, w_in
        )

        # Create output variable
        out_var = gp.Variable(
            self.container,
            name=utils._generate_name("v", self._name_prefix, "output"),
            domain=dim([len(N), len(C), h_out, w_out]),
        )
        N, C, H_out, W_out = out_var.domain

        # Calculate input window positions (top - left)
        # These indices determine where pooling windows start in the input tensor
        # Formula accounts for padding and stride
        top_index = (
            (self.stride[0] * (gp.Ord(H_out) - 1)) - self.padding[0] + 1
        )
        left_index = (
            (self.stride[1] * (gp.Ord(W_out) - 1)) - self.padding[1] + 1
        )

        # Create filter dimensions and domain relationships
        Hf, Wf = gp.math._generate_dims(self.container, self.kernel_size)
        Hf, Wf, H_in, W_in = utils._next_domains(
            [Hf, Wf, H_in, W_in], out_var.domain
        )

        # Create mapping between input/output positions
        subset = gp.Set(
            self.container,
            name=utils._generate_name(
                "s", self._name_prefix, "in_out_matching_1"
            ),
            domain=[H_out, W_out, Hf, Wf, H_in, W_in],
        )
        # Create relationship between output positions and their corresponding input windows
        subset[
            H_out,
            W_out,
            Hf,
            Wf,
            H_in,
            W_in,
        ].where[
            (gp.Ord(H_in) == (top_index + gp.Ord(Hf) - 1))  # Vertical position
            & (gp.Ord(W_in) == (left_index + gp.Ord(Wf) - 1))  # Horizontal
        ] = True

        # Create a binary variable
        bin_var = gp.Variable(
            self.container,
            name=utils._generate_name("v", self._name_prefix, "aux_variable"),
            type="binary",
            domain=[N, C, H_out, W_out, H_in, W_in],
        )

        # Create constraint equations
        greater_than = gp.Equation(
            self.container,
            name=utils._generate_name("e", self._name_prefix, "gte"),
            domain=[N, C, H_out, W_out, Hf, Wf, H_in, W_in],
        )
        less_than = gp.Equation(
            self.container,
            name=utils._generate_name("e", self._name_prefix, "lte"),
            domain=[N, C, H_out, W_out, Hf, Wf, H_in, W_in],
        )

        # Set up Big-M parameter
        if propagate_bounds:
            _big_m = self._set_bounds_and_big_M(input, big_m, subset, out_var)
        else:
            _big_m = gp.Parameter(
                self.container,
                name=utils._generate_name("p", self._name_prefix, "bigM_2"),
                records=big_m,
            )

        big_m_expr = _big_m * (1 - bin_var[N, C, H_out, W_out, H_in, W_in])

        # Build constraints based on pooling type
        if self.sense == "max":
            # For max pooling:
            greater_than[N, C, subset[H_out, W_out, Hf, Wf, H_in, W_in]] = (
                out_var[N, C, H_out, W_out] >= input[N, C, H_in, W_in]
            )
            less_than[N, C, subset[H_out, W_out, Hf, Wf, H_in, W_in]] = (
                out_var[N, C, H_out, W_out]
                <= input[N, C, H_in, W_in] + big_m_expr
            )
        else:
            # For min pooling:
            greater_than[N, C, subset[H_out, W_out, Hf, Wf, H_in, W_in]] = (
                out_var[N, C, H_out, W_out] + big_m_expr
                >= input[N, C, H_in, W_in]
            )
            less_than[N, C, subset[H_out, W_out, Hf, Wf, H_in, W_in]] = (
                out_var[N, C, H_out, W_out] <= input[N, C, H_in, W_in]
            )

        # Ensure exactly one element is selected per window
        pick_one = gp.Equation(
            self.container,
            name=utils._generate_name("e", self._name_prefix, "pick_one"),
            domain=[N, C, H_out, W_out],
        )
        pick_one[N, C, H_out, W_out] = (
            gp.Sum(
                [Hf, Wf],
                gp.Sum(
                    subset[H_out, W_out, Hf, Wf, H_in, W_in],
                    bin_var[N, C, H_out, W_out, H_in, W_in],
                ),
            )
            == 1
        )

        return out_var, [less_than, greater_than, pick_one]
