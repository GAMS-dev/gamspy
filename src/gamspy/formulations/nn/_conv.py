from __future__ import annotations

import functools
import math
import operator
from abc import abstractmethod

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import gamspy as gp
import gamspy.formulations.utils as utils
from gamspy.exceptions import ValidationError
from gamspy.formulations.nn._weighted import _WeightedLayer
from gamspy.formulations.result import FormulationResult
from gamspy.math import dim


class _ConvNd(_WeightedLayer):
    """
    Convolution over `_n_spatial` spatial dimensions. Inputs are
    (batch x in_channels x *spatial) and weights are
    (out_channels x in_channels x *kernel_size).

    Padding is stored as all leading paddings followed by all
    trailing paddings, or "same".
    """

    _n_spatial: int
    kernel_size: int | tuple[int, ...]
    stride: int | tuple[int, ...]
    padding: tuple[int, ...] | str

    def __init__(
        self,
        container: gp.Container,
        in_channels: int,
        out_channels: int,
        name_prefix: str | None,
        bias: bool,
    ) -> None:
        if not (isinstance(in_channels, int) and in_channels > 0):
            raise ValidationError("in_channels must be a positive integer")

        if not (isinstance(out_channels, int) and out_channels > 0):
            raise ValidationError("out_channels must be a positive integer")

        super().__init__(container, name_prefix, bias)
        self.in_channels = in_channels
        self.out_channels = out_channels

    @property
    @abstractmethod
    def _kernel_dims(self) -> tuple[int, ...]: ...

    @property
    @abstractmethod
    def _stride_dims(self) -> tuple[int, ...]: ...

    @property
    @abstractmethod
    def _padding_dims(self) -> tuple[int, ...] | str: ...

    @property
    def _weight_shape(self) -> tuple[int, ...]:
        return (self.out_channels, self.in_channels, *self._kernel_dims)

    @property
    def _fan_in(self) -> int:
        return self.in_channels * math.prod(self._kernel_dims)

    def _resolved_padding(self) -> tuple[int, ...]:
        padding = self._padding_dims
        if isinstance(padding, str):
            return utils._calc_same_padding(self._kernel_dims)

        return padding

    def _bias_for_output(self) -> np.ndarray | float:
        if self.bias_array is None:
            return 0.0

        return self.bias_array.reshape(-1, *([1] * self._n_spatial))

    def _affine_bounds(
        self, lower: np.ndarray, upper: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self.weight_array is not None
        n = self._n_spatial
        padding = self._resolved_padding()
        spatial_axes = tuple(range(2, 2 + n))

        if any(padding):
            pad_width = ((0, 0), (0, 0), *zip(padding[:n], padding[n:]))
            lower = np.pad(lower, pad_width, mode="constant", constant_values=0)
            upper = np.pad(upper, pad_width, mode="constant", constant_values=0)

        # (batch, in_channels, *output positions, *kernel)
        strided = (
            slice(None),
            slice(None),
            *(slice(None, None, s) for s in self._stride_dims),
        )
        windows_lower = sliding_window_view(
            lower, self._kernel_dims, axis=spatial_axes
        )[strided]
        windows_upper = sliding_window_view(
            upper, self._kernel_dims, axis=spatial_axes
        )[strided]

        pos_weight = np.maximum(self.weight_array, 0)
        neg_weight = np.minimum(self.weight_array, 0)

        # Contract in_channels and kernel axes with the weights
        window_axes = [1, *range(2 + n, 2 + 2 * n)]
        weight_axes = list(range(1, 2 + n))

        def contract(windows: np.ndarray, weight: np.ndarray) -> np.ndarray:
            out = utils._apply_real_linear(
                lambda x: np.tensordot(x, weight, axes=(window_axes, weight_axes)),
                windows,
            )
            return np.moveaxis(out, -1, 1)

        output_lower = contract(windows_lower, pos_weight) + contract(
            windows_upper, neg_weight
        )
        output_upper = contract(windows_lower, neg_weight) + contract(
            windows_upper, pos_weight
        )
        return output_lower, output_upper

    def _convolve(
        self, input: gp.Parameter | gp.Variable, propagate_bounds: bool
    ) -> FormulationResult:
        self._check_callable(propagate_bounds)
        assert self.weight is not None

        n = self._n_spatial
        if len(input.domain) != n + 2:
            raise ValidationError(
                f"expected {n + 2}D input (got {len(input.domain)}D input)"
            )

        N, C_in, *S_in = utils._get_domain(input)

        if len(C_in) != self.in_channels:
            raise ValidationError("in_channels does not match")

        if isinstance(self._padding_dims, str):
            out_sizes = tuple(len(s) for s in S_in)
        else:
            out_sizes = utils._calc_output_size(
                self._padding_dims,
                self._kernel_dims,
                self._stride_dims,
                [len(s) for s in S_in],
            )

        out = gp.Variable(
            self.container,
            domain=dim([len(N), self.out_channels, *out_sizes]),
            name=utils._generate_name("v", self._name_prefix, "output"),
        )

        N, C_out, *S_out = utils._get_domain(out)

        set_out = gp.Equation(
            self.container,
            domain=out.domain,
            name=utils._generate_name("e", self._name_prefix, "set_output"),
        )

        padding = self._resolved_padding()

        # First input position of the window that produces each output position
        starts = [
            (stride * (gp.Ord(s_out) - 1)) - pad + 1
            for stride, s_out, pad in zip(self._stride_dims, S_out, padding)
        ]

        _, _, *F = utils._get_domain(self.weight)
        C_in, *rest = utils._next_domains(
            [C_in, *F, *S_in],
            utils._get_domain(out),
        )
        F, S_in = rest[:n], rest[n:]

        subset = gp.Set(
            self.container,
            domain=[*S_out, *F, *S_in],
            name=utils._generate_name("s", self._name_prefix, "conv_subset"),
        )
        subset[(*S_out, *F, *S_in)].where[
            functools.reduce(
                operator.and_,
                [
                    gp.Ord(s_in) == (start + gp.Ord(f) - 1)
                    for s_in, start, f in zip(S_in, starts, F)
                ],
            )
        ] = True

        expr = gp.Sum(
            [C_in],
            gp.Sum(
                subset[(*S_out, *F, *S_in)],
                input[(N, C_in, *S_in)] * self.weight[(C_out, C_in, *F)],
            ),
        )

        if self.bias is not None:
            expr = expr + self.bias[C_out]

        set_out[(N, C_out, *S_out)] = out[(N, C_out, *S_out)] == expr

        result = FormulationResult(
            result=out,
            equations_created={"set_output": set_out},
        )
        result.variables_created["output"] = out
        result.sets_created["conv_subset"] = subset
        self._add_weights_to_result(result)
        self._propagate_bounds(input, out, result, propagate_bounds=propagate_bounds)

        return result

    def __str__(self) -> str:
        return (
            f"Conv{self._n_spatial}D(\n"
            f"  in_channels={self.in_channels}\n"
            f"  out_channels={self.out_channels}\n"
            f"  kernel_size={self.kernel_size}\n"
            f"  stride={self.stride}\n"
            f"  padding={self.padding}\n"
            f"  bias={self.use_bias}\n"
            f"  weights_loaded={'True' if self._state == 1 else 'False'}\n)"
        )
