from __future__ import annotations

import numpy as np

import gamspy as gp
import gamspy.formulations.utils as utils
from gamspy.exceptions import ValidationError
from gamspy.formulations.nn._weighted import _WeightedLayer
from gamspy.formulations.result import FormulationResult


class Linear(_WeightedLayer):
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
        name_prefix: str | None = None,
        *,
        bias: bool = True,
    ) -> None:
        if not isinstance(in_features, int) or in_features <= 0:
            raise ValidationError("in_features must be a positive integer")

        if not isinstance(out_features, int) or out_features <= 0:
            raise ValidationError("out_features must be a positive integer")

        super().__init__(container, name_prefix, bias)
        self.in_features = in_features
        self.out_features = out_features

    @property
    def _weight_shape(self) -> tuple[int, ...]:
        return (self.out_features, self.in_features)

    @property
    def _fan_in(self) -> int:
        return self.in_features

    def _bias_for_output(self) -> np.ndarray | float:
        return 0.0 if self.bias_array is None else self.bias_array

    def _affine_bounds(
        self, lower: np.ndarray, upper: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self.weight_array is not None
        w_pos = np.maximum(self.weight_array, 0)
        w_neg = np.minimum(self.weight_array, 0)

        def matmul(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
            return utils._apply_real_linear(lambda v: v @ weight.T, x)

        output_lower = matmul(lower, w_pos) + matmul(upper, w_neg)
        output_upper = matmul(upper, w_pos) + matmul(lower, w_neg)
        return output_lower, output_upper

    def load_weights(self, weight: np.ndarray, bias: np.ndarray | None = None) -> None:
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
        self._load_weights(weight, bias)

    def make_variable(self, *, init_weights: bool = False) -> None:
        """
        Mark Linear layer as variable. After this is called `load_weights`
        cannot be called. Use this when you need to learn the weights
        of your linear layer in your optimization model.


        Parameters
        ----------
        init_weights : Optional[bool]
               False by default.
               Whether to initialize weights. It is suggested you set
               this to True unless you want to initialize weights yourself.
               When `init_weights` is set to True, values are initialized from
               :math:`\\mathcal{U}(-\\sqrt{k},\\sqrt{k})`, where :math:`k = 1/in\\_features`.
        """
        self._make_variable(init_weights)

    def __call__(
        self, input: gp.Parameter | gp.Variable, *, propagate_bounds: bool = True
    ) -> FormulationResult:
        """
        Forward pass your input, generate output and equations required for
        calculating the linear transformation. If `propagate_bounds` is True,
        the `input` is of type variable, and `load_weights` was called, then
        the bounds of the input are propagated to the output.

        Returns `FormulationResult` which can be unpacked as a output variable and list of equations.

        FormulationResult:
            - equations_created: ["set_output"]
            - variables_created: ["output", "weight", "bias"]
            - parameters_created: ["weight", "bias", "input_bounds", "output_bounds"]

        Note:
            - For backward compatibility, this result object can be unpacked as a tuple: `output, equations = linear(input)`.
            - `weight` and `bias` are available as variables if `make_variable` was called.
            - `weight` and `bias` are available as parameters if `load_weights` was called.
            - `input_bounds` and `output_bounds`are available as parameters if `propogate_bounds=True`.

        Parameters
        ----------
        input : gp.Parameter | gp.Variable
                input to the linear layer, must be in shape
                (* x in_features)
        propagate_bounds : bool = True
                If True, propagate bounds of the input to the output.
                Otherwise, the output variable is unbounded.

        Returns
        -------
        FormulationResult
        """
        self._check_callable(propagate_bounds)
        assert self.weight is not None

        if len(input.domain) == 0:
            raise ValidationError("expected an input with at least 1 dimension")

        if len(utils._get_domain(input)[-1]) != self.in_features:
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

        result = FormulationResult(
            result=out,
            equations_created={"set_output": set_out},
        )
        result.variables_created["output"] = out
        self._add_weights_to_result(result)
        self._propagate_bounds(input, out, result, propagate_bounds=propagate_bounds)

        return result

    def __str__(self) -> str:
        return (
            "Linear(\n"
            f"  in_features={self.in_features}\n"
            f"  out_features={self.out_features}\n"
            f"  bias={self.use_bias}\n"
            f"  weights_loaded={'True' if self._state == 1 else 'False'}\n)"
        )
