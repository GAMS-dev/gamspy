from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

import gamspy as gp
import gamspy.formulations.utils as utils
from gamspy.exceptions import ValidationError
from gamspy.math import dim

if TYPE_CHECKING:
    from gamspy import Parameter, Variable
    from gamspy.formulations.result import FormulationResult


class _WeightedLayer(ABC):
    """
    Shared weight handling of layers with a weight and an optional bias.

    `_state` is 0 until `load_weights` (1) or `make_variable` (2) is called.
    """

    def __init__(
        self, container: gp.Container, name_prefix: str | None, bias: bool
    ) -> None:
        if not isinstance(bias, bool):
            raise ValidationError("bias must be a boolean")

        self.container = container
        self.use_bias = bias
        self._state = 0
        self.weight: Parameter | Variable | None = None
        self.weight_array: np.ndarray | None = None
        self.bias: Parameter | Variable | None = None
        self.bias_array: np.ndarray | None = None

        if name_prefix is None:
            name_prefix = gp.utils._get_unique_name()

        self._name_prefix = name_prefix

    @property
    @abstractmethod
    def _weight_shape(self) -> tuple[int, ...]: ...

    @property
    @abstractmethod
    def _fan_in(self) -> int: ...

    @abstractmethod
    def _bias_for_output(self) -> np.ndarray | float: ...

    @abstractmethod
    def _affine_bounds(
        self, lower: np.ndarray, upper: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def _load_weights(self, weight: np.ndarray, bias: np.ndarray | None) -> None:
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

        expected_shape = self._weight_shape
        if len(weight.shape) != len(expected_shape):
            raise ValidationError(
                f"expected {len(expected_shape)}D input for weight (got {len(weight.shape)}D input)"
            )

        if weight.shape != expected_shape:
            raise ValidationError(f"weight expected to be in shape {expected_shape}")

        out_size = expected_shape[0]
        if bias is not None:
            if len(bias.shape) != 1:
                raise ValidationError(
                    f"expected 1D input for bias (got {len(bias.shape)}D input)"
                )

            if bias.shape != (out_size,):
                raise ValidationError(f"bias expected to be in shape ({out_size},)")

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
                    domain=dim([out_size]),
                    records=bias,
                    name=utils._generate_name("p", self._name_prefix, "bias"),
                )
            else:
                self.bias.setRecords(bias)

            self.bias_array = bias

        self._state = 1

    def _make_variable(self, init_weights: bool) -> None:
        if self._state == 1:
            raise ValidationError(
                "make_variable cannot be used after calling load_weights"
            )

        expected_shape = self._weight_shape
        sk = math.sqrt(1 / self._fan_in)
        if self.weight is None:
            self.weight = gp.Variable(
                self.container,
                domain=dim(expected_shape),
                name=utils._generate_name("v", self._name_prefix, "weight"),
            )
            if init_weights:
                self.weight.l[...] = gp.math.uniform(-sk, sk)

        if self.use_bias and self.bias is None:
            self.bias = gp.Variable(
                self.container,
                domain=dim([expected_shape[0]]),
                name=utils._generate_name("v", self._name_prefix, "bias"),
            )
            if init_weights:
                self.bias.l[...] = gp.math.uniform(-sk, sk)

        self._state = 2

    def _check_callable(self, propagate_bounds: bool) -> None:
        if not isinstance(propagate_bounds, bool):
            raise ValidationError("propagate_bounds must be a boolean")

        if self.weight is None:
            raise ValidationError(
                "You must call load_weights or make_variable first before using the"
                f" {type(self).__name__}"
            )

    def _add_weights_to_result(self, result: FormulationResult) -> None:
        if isinstance(self.weight, gp.Variable):
            result.variables_created["weight"] = self.weight
        elif isinstance(self.weight, gp.Parameter):
            result.parameters_created["weight"] = self.weight

        if isinstance(self.bias, gp.Variable):
            result.variables_created["bias"] = self.bias
        elif isinstance(self.bias, gp.Parameter):
            result.parameters_created["bias"] = self.bias

    def _propagate_bounds(
        self,
        input: gp.Parameter | gp.Variable,
        output: gp.Variable,
        result: FormulationResult,
        *,
        propagate_bounds: bool,
    ) -> None:
        if not (
            propagate_bounds and self._state == 1 and isinstance(input, gp.Variable)
        ):
            return

        input_bounds = gp.Parameter(
            self.container,
            domain=dim([2, *input.shape]),
            name=utils._generate_name("p", self._name_prefix, "input_bounds"),
        )
        input_bounds[("0",) + tuple(input.domain)] = input.lo[...]
        input_bounds[("1",) + tuple(input.domain)] = input.up[...]
        result.parameters_created["input_bounds"] = input_bounds

        name = utils._generate_name("p", self._name_prefix, "output_bounds")

        # All-zero bounds have no records; the output is then just the bias
        if input_bounds.records is None:
            out_bounds = gp.Parameter(
                self.container,
                domain=dim(output.shape),
                records=np.zeros(output.shape) + self._bias_for_output(),
                name=name,
            )
            output.lo[...] = out_bounds
            output.up[...] = out_bounds
            result.parameters_created["output_bounds"] = out_bounds
            return

        lower, upper = input_bounds.toDense()

        inf_exists = input_bounds.countNegInf() or input_bounds.countPosInf()
        if inf_exists:
            lower = utils._encode_infinity(lower)
            upper = utils._encode_infinity(upper)

        output_lower, output_upper = self._affine_bounds(lower, upper)
        output_lower = output_lower + self._bias_for_output()
        output_upper = output_upper + self._bias_for_output()

        if inf_exists:
            output_lower = utils._decode_complex_array(output_lower)
            output_upper = utils._decode_complex_array(output_upper)

        out_bounds = gp.Parameter(
            self.container,
            domain=dim([2, *output.shape]),
            records=np.stack([output_lower, output_upper], axis=0),
            name=name,
        )

        output.lo[...] = out_bounds[("0",) + tuple(output.domain)]
        output.up[...] = out_bounds[("1",) + tuple(output.domain)]

        result.parameters_created["output_bounds"] = out_bounds
