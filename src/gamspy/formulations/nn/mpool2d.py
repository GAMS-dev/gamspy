from __future__ import annotations

import uuid
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
    ):
        if sense not in ["min", "max"]:
            raise ValidationError("_MPool2d expects min or max")

        _kernel_size = utils._check_tuple_int(kernel_size, "kernel_size")
        if stride is None:
            stride = kernel_size

        _stride = utils._check_tuple_int(stride, "stride")
        _padding = utils._check_tuple_int(padding, "padding", allow_zero=True)

        self.container = container
        self.kernel_size = _kernel_size
        self.stride = _stride
        self.padding = _padding
        self.sense = sense

    def _get_big_M_for_N_C(
        self,
        input: gp.Parameter | gp.Variable,
        default_big_m: int,
        subset: gp.Set,
    ) -> tuple[gp.Parameter, gp.Parameter, gp.Parameter]:
        N, C = input.domain[:2]
        H_out, W_out, Hf, Wf, H_in, W_in = subset.domain
        subset2 = gp.Set(self.container, domain=[H_out, W_out, H_in, W_in])
        subset2[H_out, W_out, H_in, W_in] = gp.Sum(
            [Hf, Wf], subset[H_out, W_out, Hf, Wf, H_in, W_in]
        )

        big_m_par = gp.Parameter(self.container, domain=[N, C, H_out, W_out])
        lb = gp.Parameter(self.container, domain=[N, C, H_out, W_out])
        ub = gp.Parameter(self.container, domain=[N, C, H_out, W_out])

        if isinstance(input, gp.Variable):
            ub[...] = gp.Smax(
                gp.Domain(H_in, W_in).where[subset2[H_out, W_out, H_in, W_in]],
                input.up[N, C, H_in, W_in],
            )
            lb[...] = gp.Smin(
                gp.Domain(H_in, W_in).where[subset2[H_out, W_out, H_in, W_in]],
                input.lo[N, C, H_in, W_in],
            )
        else:
            ub[...] = gp.Smax(
                gp.Domain(H_in, W_in).where[subset2[H_out, W_out, H_in, W_in]],
                input[N, C, H_in, W_in],
            )
            lb[...] = gp.Smin(
                gp.Domain(H_in, W_in).where[subset2[H_out, W_out, H_in, W_in]],
                input[N, C, H_in, W_in],
            )

        big_m_par[N, C, H_out, W_out] = ub - lb
        big_m_par[...].where[big_m_par[...] == "inf"] = default_big_m
        return big_m_par, lb, ub

    def __call__(
        self, input: gp.Parameter | gp.Variable, big_m: int = 1000
    ) -> tuple[gp.Variable, list[gp.Equation]]:
        if not isinstance(input, (gp.Parameter, gp.Variable)):
            raise ValidationError("Expected a parameter or a variable input")

        if len(input.domain) != 4:
            raise ValidationError(
                f"expected 4D input (got {len(input.domain)}D input)"
            )

        N, C, H_in, W_in = input.domain

        h_in = len(H_in)
        w_in = len(W_in)

        h_out, w_out = utils._calc_hw(
            self.padding, self.kernel_size, self.stride, h_in, w_in
        )

        out_var = gp.Variable(
            self.container, domain=dim([len(N), len(C), h_out, w_out])
        )

        N, C, H_out, W_out = out_var.domain

        # expr must have domain N, C, H_out, W_out
        top_index = (
            (self.stride[0] * (gp.Ord(H_out) - 1)) - self.padding[0] + 1
        )
        left_index = (
            (self.stride[1] * (gp.Ord(W_out) - 1)) - self.padding[1] + 1
        )

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

        bin_var = gp.Variable(
            self.container,
            type="binary",
            domain=[N, C, H_out, W_out, H_in, W_in],
        )

        greater_than = gp.Equation(
            self.container, domain=[N, C, H_out, W_out, Hf, Wf, H_in, W_in]
        )

        less_than = gp.Equation(
            self.container, domain=[N, C, H_out, W_out, Hf, Wf, H_in, W_in]
        )

        # can be done better
        _big_m, lb, ub = self._get_big_M_for_N_C(input, big_m, subset)
        big_m_expr = _big_m * (1 - bin_var[N, C, H_out, W_out, H_in, W_in])
        if self.sense == "max":
            greater_than[N, C, subset[H_out, W_out, Hf, Wf, H_in, W_in]] = (
                out_var[N, C, H_out, W_out] >= input[N, C, H_in, W_in]
            )
            less_than[N, C, subset[H_out, W_out, Hf, Wf, H_in, W_in]] = (
                out_var[N, C, H_out, W_out]
                <= input[N, C, H_in, W_in] + big_m_expr
            )
        else:
            greater_than[N, C, subset[H_out, W_out, Hf, Wf, H_in, W_in]] = (
                out_var[N, C, H_out, W_out] + big_m_expr
                >= input[N, C, H_in, W_in]
            )
            less_than[N, C, subset[H_out, W_out, Hf, Wf, H_in, W_in]] = (
                out_var[N, C, H_out, W_out] <= input[N, C, H_in, W_in]
            )

        pick_one = gp.Equation(self.container, domain=[N, C, H_out, W_out])
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

        out_var.lo[...] = lb
        out_var.up[...] = ub

        return out_var, [less_than, greater_than, pick_one]
