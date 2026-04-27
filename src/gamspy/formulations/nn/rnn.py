from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import numpy as np

    from gamspy import Parameter

import gamspy as gp
import gamspy.formulations.utils as utils
from gamspy.exceptions import ValidationError
from gamspy.formulations.result import FormulationResult
from gamspy.math import dim


class RNN:
    """
    Formulation generator for Recurrent Neural Networks in GAMSPy. It can
    be used to embed trained Recurrent neural networks in your problem.

    Note: It currently does **NOT** support Bidirectional RNNs and Dropout layers.

    Parameters
    ----------
    container : Container
        Container that will hold the new variables and equations.
    input_size : int
        The number of expected features in the input sequence.
    hidden_size : int
        The number of features in the hidden state.
    activation : Literal["tanh", "relu", "linear"]
        The activation function applied to the hidden state update. By default "tanh".

    Examples
    --------
    >>> import gamspy as gp
    >>> import numpy as np
    >>> from gamspy.math import dim
    >>> m = gp.Container()
    >>> # 2 input features, 4 hidden units
    >>> rnn = gp.formulations.RNN(m, input_size=2, hidden_size=4)
    >>> w_ih = np.random.rand(4, 2)
    >>> w_hh = np.random.rand(4, 4)
    >>> b_ih = np.random.rand(4)
    >>> b_hh = np.random.rand(4)
    >>> rnn.load_weights(w_ih, w_hh, b_ih, b_hh)
    >>> batch, time_step, features = [1, 3, 2]
    >>> input_domain = dim([batch, time_step, features])
    >>> x = gp.Parameter(m, name="x_in", domain=input_domain, records=np.random.rand(1, 3, 2))
    >>> out_var = rnn(x).result
    >>> type(out_var)
    <class 'gamspy._symbols.variable.Variable'>
    >>> [d.name for d in out_var.domain]
    ['DenseDim1_1', 'DenseDim3_1', 'DenseDim4_1']

    """

    def __init__(
        self,
        container: gp.Container,
        input_size: int,
        hidden_size: int,
        activation: Literal["tanh", "relu", "linear"] = "tanh",
    ):
        if not isinstance(input_size, int) or input_size <= 0:
            raise ValidationError("input_size must be a positive integer")

        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValidationError("hidden_size must be a positive integer")

        if activation not in ["tanh", "relu", "linear"]:
            raise ValidationError(
                f"""activation must be either of ["tanh", "relu", "linear"] and not >{activation}<"""
            )

        self.container = container
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation

        self._state = 0
        self.weight_ih: Parameter | None = None
        self.weight_hh: Parameter | None = None
        self.bias_ih: Parameter | None = None
        self.bias_hh: Parameter | None = None

    def load_weights(
        self,
        weight_ih: np.ndarray,
        weight_hh: np.ndarray,
        bias_ih: np.ndarray | None = None,
        bias_hh: np.ndarray | None = None,
    ) -> None:
        """
        Mark RNN as parameter and load weights from NumPy arrays.
        Use this when you already have the weights of your hidden layers.

        Parameters
        ----------
        weight_ih : np.ndarray
                    The input-to-hidden layer weights.
                    Shape: (hidden_size, input_size)
        weight_hh : np.ndarray
                    The hidden-to-hidden layer weights.
                    Shape: (hidden_size, hidden_size)
        bias_ih : np.ndarray | None
                  The input-to-hidden layer bias.
                  Shape: (hidden_size, )
        bias_hh : np.ndarray | None
                  The hidden-to-hidden layer bias.
                  Shape: (hidden_size, )
        """
        import numpy as np

        if weight_ih.shape != (self.hidden_size, self.input_size):
            raise ValidationError(
                f"weight_ih shape mismatch: expected {(self.hidden_size, self.input_size)}"
            )
        if weight_hh.shape != (self.hidden_size, self.hidden_size):
            raise ValidationError(
                f"weight_hh shape mismatch: expected {(self.hidden_size, self.hidden_size)}"
            )

        # Create Parameters
        self.weight_ih = gp.Parameter(
            self.container,
            domain=dim(weight_ih.shape),
            records=weight_ih,
        )

        assert self.weight_ih is not None
        H_set = self.weight_ih.domain[0]
        H_prev = gp.Alias(
            self.container,
            alias_with=H_set,
        )

        self.weight_hh = gp.Parameter(
            self.container,
            domain=[H_set, H_prev],
            records=weight_hh,
        )

        # Handle Biases (default to 0 if None)
        if bias_ih is None:
            bias_ih = np.zeros(self.hidden_size)
        elif bias_ih.shape != (self.hidden_size,):
            raise ValidationError(
                f"bias_ih shape mismatch: expected {(self.hidden_size,)}"
            )

        if bias_hh is None:
            bias_hh = np.zeros(self.hidden_size)
        elif bias_hh.shape != (self.hidden_size,):
            raise ValidationError(
                f"bias_hh shape mismatch: expected {(self.hidden_size,)}"
            )

        self.bias_ih = gp.Parameter(
            self.container,
            domain=dim([self.hidden_size]),
            records=bias_ih,
        )
        self.bias_hh = gp.Parameter(
            self.container,
            domain=dim([self.hidden_size]),
            records=bias_hh,
        )

        self.weight_ih_array = weight_ih
        self.weight_hh_array = weight_hh
        self.bias_ih_array = bias_ih
        self.bias_hh_array = bias_hh

        self._state = 1

    def _propagate_bounds(
        self,
        input: gp.Variable,
        result: FormulationResult,
        h_out_shape: tuple,
        h0: gp.Parameter | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """
        Sequentially calculates the tight lower and upper bounds for the RNN hidden state.
        """
        import numpy as np

        x_bounds = gp.Parameter(
            self.container,
            domain=dim([2, *input.shape]),
        )
        x_bounds[("0",) + tuple(input.domain)] = input.lo[...]
        x_bounds[("1",) + tuple(input.domain)] = input.up[...]
        result.parameters_created["input_bounds"] = x_bounds
        h0_bounds_array = None
        if h0 is not None:
            h0_bounds_array = h0.toDense()

        # If the bounds are all zeros (None in GAMSPy parameters)
        if x_bounds.records is None:
            x_lb = np.zeros(shape=input.shape)
            x_ub = np.zeros(shape=input.shape)
        else:
            x_lb, x_ub = x_bounds.toDense()

        batch_size, time_steps, _ = input.shape

        inf_exists = x_bounds.countNegInf() or x_bounds.countPosInf()
        if inf_exists:
            x_lb = utils._encode_infinity(x_lb)
            x_ub = utils._encode_infinity(x_ub)

        w_ih_pos = np.maximum(self.weight_ih_array, 0)
        w_ih_neg = np.minimum(self.weight_ih_array, 0)
        w_hh_pos = np.maximum(self.weight_hh_array, 0)
        w_hh_neg = np.minimum(self.weight_hh_array, 0)

        pre_act_lb_seq, pre_act_ub_seq = [], []
        h_lb_seq, h_ub_seq = [], []

        if h0_bounds_array is not None:
            prev_lb = h0_bounds_array
            prev_ub = h0_bounds_array
        else:
            prev_lb = np.zeros((batch_size, self.hidden_size))
            prev_ub = np.zeros((batch_size, self.hidden_size))

        for t in range(time_steps):
            x_lb_t = x_lb[:, t, :]
            x_ub_t = x_ub[:, t, :]

            in_lb = (x_lb_t @ w_ih_pos.T) + (x_ub_t @ w_ih_neg.T) + self.bias_ih_array
            in_ub = (x_ub_t @ w_ih_pos.T) + (x_lb_t @ w_ih_neg.T) + self.bias_ih_array

            if t == 0 and h0_bounds_array is None:
                hid_lb = self.bias_hh_array
                hid_ub = self.bias_hh_array
            else:
                hid_lb = (
                    (prev_lb @ w_hh_pos.T) + (prev_ub @ w_hh_neg.T) + self.bias_hh_array
                )
                hid_ub = (
                    (prev_ub @ w_hh_pos.T) + (prev_lb @ w_hh_neg.T) + self.bias_hh_array
                )

            pre_lb = in_lb + hid_lb
            pre_ub = in_ub + hid_ub

            # Decode before activation so Numpy does not evaluate wrong bounds
            # np.maximum(8, 2 + 1j) => 8
            pre_lb_real = utils._decode_complex_array(pre_lb)
            pre_ub_real = utils._decode_complex_array(pre_ub)

            if self.activation == "relu":
                pre_act_lb_seq.append(pre_lb_real)
                pre_act_ub_seq.append(pre_ub_real)
                h_lb_real = np.maximum(0, pre_lb_real)
                h_ub_real = np.maximum(0, pre_ub_real)
            elif self.activation == "tanh":
                h_lb_real = np.tanh(pre_lb_real)
                h_ub_real = np.tanh(pre_ub_real)
            else:  # linear
                h_lb_real = pre_lb_real
                h_ub_real = pre_ub_real

            h_lb_seq.append(h_lb_real)
            h_ub_seq.append(h_ub_real)

            # Re-encode for the next time step's matmul
            if inf_exists:
                prev_lb = utils._encode_infinity(h_lb_real)
                prev_ub = utils._encode_infinity(h_ub_real)
            else:
                prev_lb = h_lb_real
                prev_ub = h_ub_real

        # stacked arrays are Real
        if self.activation == "relu":
            pre_act_bounds = np.stack(
                [np.stack(pre_act_lb_seq, axis=1), np.stack(pre_act_ub_seq, axis=1)],
                axis=0,
            )
            relu_bounds = gp.Parameter(
                self.container,
                domain=dim([2, *h_out_shape]),
                records=pre_act_bounds,
            )
            result.parameters_created["relu_bounds"] = relu_bounds
        else:
            pre_act_bounds = None
            relu_bounds = None

        output_bounds = np.stack(
            [np.stack(h_lb_seq, axis=1), np.stack(h_ub_seq, axis=1)], axis=0
        )
        out_bounds = gp.Parameter(
            self.container,
            domain=dim([2, *h_out_shape]),
            records=output_bounds,
        )
        result.parameters_created["out_bounds"] = out_bounds

        return relu_bounds, out_bounds

    def __call__(
        self,
        input_seq: gp.Parameter | gp.Variable,
        h0: gp.Parameter | None = None,
        propagate_bounds: bool = True,
    ) -> FormulationResult:
        """
        Forward pass your input sequence, generating the output hidden states and
        equations required for calculating the recurrent neural network steps over time.
        If `propagate_bounds` is True (default), the `input_seq` is of type variable, and
        `load_weights` was called, then the bounds of the input are propagated to the output.

        Returns `FormulationResult` which can be unpacked as a output variable and list of equations.

        FormulationResult:
            - equations_created: ["set_output", "set_pre_act", "y_gte_x", "y_lte_x_1", "y_lte_x_2"]
            - variables_created: ["output", "pre_act", "binary"]
            - parameters_created: ["w_ih", "w_hh", "b_ih", "b_hh", "input_bounds", "out_bounds", "relu_bounds"]

        Note:
            - The `output` variable will have the domain (batch, time_steps, hidden_size).
            - Following equations are available only when `activation="relu"`,
              ["set_pre_act", "y_gte_x", "y_lte_x_1", "y_lte_x_2"].
            - Following variables are available only when `activation="relu"`,
              ["pre_act", "binary"].
            - Following parameters are available only when `propagate_bounds=True`,
              ["input_bounds", "out_bounds", "relu_bounds"]. Further, `relu_bounds` is only
              available when `activation="relu"`.
            - For backward compatibility, this result object can be unpacked as a tuple:
              `output, equations = rnn_layer(input_seq)`.

        Parameters
        ----------
        input_seq : gp.Parameter | gp.Variable
            Input sequence to the RNN layer. It must be a 3D symbol of the following
            shape (batch_size, time_steps, input_features).
        h0 : gp.Parameter | None
            Initial hidden state for the first time step. If None, the initial hidden
            state is assumed to be a vector of zeros. By default None.
            Shape: (batch, hidden_size)
        propagate_bounds : bool = True
                If True, propagate bounds of the input to the output.
                Otherwise, the output variable is unbounded.

        Returns
        -------
        FormulationResult
        """
        if self._state != 1:
            raise ValidationError("Call load_weights before generating formulation.")

        if not isinstance(propagate_bounds, bool):
            raise ValidationError("propagate_bounds should be a boolean.")

        assert self.weight_ih is not None
        assert self.weight_hh is not None
        assert self.bias_ih is not None
        assert self.bias_hh is not None

        if len(input_seq.domain) != 3:
            raise ValidationError(
                f"Expected 3D input (batch, time_step, feature), got {len(input_seq.domain)}"
            )

        if len(input_seq.domain[-1]) != self.weight_ih.shape[-1]:
            raise ValidationError(
                f"Last dimension of Input sequence does not match. Expected {self.weight_ih.shape[-1]}, got {len(input_seq.domain[-1])}."
            )

        if h0 is not None:
            expected = (len(input_seq.domain[0]), self.weight_hh.shape[-1])
            actual = tuple(len(i) for i in h0.domain)
            if expected != actual:
                raise ValidationError(
                    f"h0 shape mismatch: expected {expected}, got {actual}"
                )

        linear_input = input_seq @ self.weight_ih.t()
        linear_input = linear_input + self.bias_ih[linear_input.domain[-1]]

        h_out = gp.Variable(
            self.container,
            domain=linear_input.domain,
        )

        N_set, T_set, H_set = h_out.domain
        _, H_prev = self.weight_hh.domain

        if len(T_set) == 1:
            linear_hidden = self.bias_hh[H_set]
            if h0 is not None:
                h0_effect = gp.Sum(
                    H_prev, h0[N_set, H_prev] * self.weight_hh[H_set, H_prev]
                )
                linear_hidden = linear_hidden + h0_effect
        else:
            linear_hidden = (
                gp.Sum(
                    H_prev,
                    h_out[N_set, T_set.lag(1), H_prev] * self.weight_hh[H_set, H_prev],
                )
                + self.bias_hh[H_set]
            )
            if h0 is not None:
                h0_effect = gp.Sum(
                    H_prev, h0[N_set, H_prev] * self.weight_hh[H_set, H_prev]
                )
                linear_hidden = linear_hidden + h0_effect.where[gp.Ord(T_set) == 1]

        total_pre_activation = linear_input + linear_hidden
        result = FormulationResult(result=h_out, equations_created={})
        result.parameters_created.update(
            {
                "w_ih": self.weight_ih,
                "w_hh": self.weight_hh,
                "b_ih": self.bias_ih,
                "b_hh": self.bias_hh,
            }
        )

        pre_act_bounds = None
        if propagate_bounds and isinstance(input_seq, gp.Variable):
            pre_act_bounds, output_bounds = self._propagate_bounds(
                input=input_seq,
                result=result,
                h_out_shape=h_out.shape,
                h0=h0,
            )

            h_out.lo[...] = output_bounds[("0",) + tuple(h_out.domain)]
            h_out.up[...] = output_bounds[("1",) + tuple(h_out.domain)]
        if self.activation == "tanh":
            rnn_eq = gp.Equation(
                self.container,
                domain=h_out.domain,
            )
            rnn_eq[...] = h_out == gp.math.tanh(total_pre_activation)

            result.equations_created["set_output"] = rnn_eq
            result.variables_created["output"] = h_out

        elif self.activation == "relu":
            from gamspy.math.activation import relu_with_binary_var

            pre_act_var = gp.Variable(
                self.container,
                domain=h_out.domain,
            )
            if pre_act_bounds is not None:
                pre_act_var.lo[...] = pre_act_bounds[("0",) + tuple(h_out.domain)]
                pre_act_var.up[...] = pre_act_bounds[("1",) + tuple(h_out.domain)]

            pre_act_eq = gp.Equation(
                self.container,
                domain=h_out.domain,
            )
            pre_act_eq[...] = pre_act_var == total_pre_activation
            relu_res = relu_with_binary_var(pre_act_var)

            link_eq = gp.Equation(
                self.container,
                domain=h_out.domain,
            )
            link_eq[...] = h_out == relu_res.result

            result.equations_created.update(
                {
                    "set_pre_act": pre_act_eq,
                    "set_output": link_eq,
                    **relu_res.equations_created,
                }
            )

            result.variables_created.update(
                {"output": h_out, "pre_act": pre_act_var, **relu_res.variables_created}
            )

        else:  # linear
            rnn_eq = gp.Equation(
                self.container,
                domain=h_out.domain,
            )
            rnn_eq[...] = h_out == total_pre_activation
            result.equations_created["set_output"] = rnn_eq
            result.variables_created["output"] = h_out

        return result

    def __str__(self) -> str:
        return (
            "RNN(\n"
            f"  input_size={self.input_size}\n"
            f"  hidden_size={self.hidden_size}\n"
            f"  activation={self.activation}\n"
            f"  weights_loaded={'True' if self._state == 1 else 'False'}\n)"
        )
