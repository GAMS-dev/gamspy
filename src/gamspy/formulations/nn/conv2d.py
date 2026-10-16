from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import gamspy as gp
import gamspy.formulations.utils as utils
from gamspy.exceptions import ValidationError
from gamspy.formulations.nn._conv import _ConvNd

if TYPE_CHECKING:
    import numpy as np

    from gamspy.formulations.result import FormulationResult


class Conv2d(_ConvNd):
    """
    Formulation generator for 2D Convolution symbol in GAMS. It can
    be used to embed convolutional layers of trained neural networks
    in your problem. It can also be used to embed convolutional layers
    when you need weights as variables.

    Parameters
    ----------
    container : Container
        Container that will contain the new variable and equations.
    in_channel : int
        Number of channels in the input
    out_channel : int
        Number of channels in the output
    kernel_size : int | tuple[int, int]
        Filter size
    stride : int | tuple[int, int]
        Stride in the convolution, by default 1
    padding : int | tuple[int, int] | Literal["same", "valid"]
        Specifies the amount of padding to apply to the input, by default 0.
        If an integer is provided, that padding is applied to both the height and width.
        If a tuple of two integers is given, the first value determines the padding for the
        top and bottom, while the second value sets the padding for the left and right.
        It is also possible to provide string literals "same" and "valid". "same" pads
        the input so the output has the shape as the input. "valid" is the same as no
        padding.
    bias : bool
        Is bias added after the convolution, by default True
    name_prefix : str | None
        Prefix for names of the GAMS symbols generated, by default None which means
        random prefix. Using same name_prefix in different formulations causes name
        conflicts. Do not use same name_prefix again.

    Examples
    --------
    >>> import gamspy as gp
    >>> import numpy as np
    >>> from gamspy.math import dim
    >>> w1 = np.random.rand(2, 1, 3, 3)
    >>> b1 = np.random.rand(2)
    >>> m = gp.Container()
    >>> # in_channels=1, out_channels=2, kernel_size=3x3
    >>> conv1 = gp.formulations.Conv2d(m, 1, 2, 3)
    >>> conv1.load_weights(w1, b1)
    >>> # 10 images, 1 channel, 24 by 24
    >>> inp = gp.Variable(m, domain=dim((10, 1, 24, 24)))
    >>> out, eqs = conv1(inp)
    >>> type(out)
    <class 'gamspy._symbols.variable.Variable'>
    >>> [len(x) for x in out.domain]
    [10, 2, 22, 22]

    """

    _n_spatial = 2
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int, int, int] | str

    def __init__(
        self,
        container: gp.Container,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | Literal["same", "valid"] = 0,
        name_prefix: str | None = None,
        *,
        bias: bool = True,
    ):
        _kernel_size = utils._check_tuple_int(kernel_size, "kernel_size")
        _stride = utils._check_tuple_int(stride, "stride")

        if isinstance(padding, str):
            if padding not in {"same", "valid"}:
                raise ValidationError(
                    "padding must be 'same' or 'valid' when it is a string"
                )

            if padding == "same" and _stride != (1, 1):
                raise ValidationError("'same' padding can only be used with stride=1")

            _padding: tuple[int, int, int, int] | str = (
                (0, 0, 0, 0) if padding == "valid" else "same"
            )

        else:
            _padding = utils._check_padding(padding)

        super().__init__(container, in_channels, out_channels, name_prefix, bias)
        self.kernel_size = _kernel_size
        self.stride = _stride
        self.padding = _padding

    @property
    def _kernel_dims(self) -> tuple[int, ...]:
        return self.kernel_size

    @property
    def _stride_dims(self) -> tuple[int, ...]:
        return self.stride

    @property
    def _padding_dims(self) -> tuple[int, ...] | str:
        return self.padding

    def make_variable(self, *, init_weights: bool = False) -> None:
        """
        Mark Conv2d as variable. After this is called `load_weights`
        cannot be called. Use this when you need to learn the weights
        of your convolutional layer in your optimization model.

        Parameters
        ----------
        init_weights : Optional[bool]
               False by default.
               Whether to initialize weights. It is suggested you set
               this to True unless you want to initialize weights yourself.
               When `init_weights` is set to True, values are initialized from
               :math:`\\mathcal{U}(-\\sqrt{k},\\sqrt{k})`, where :math:`k = 1/[C_{in} * \\prod_{i=0}^{1}{kernel\\_size_n}]`.
        """
        self._make_variable(init_weights)

    def load_weights(self, weight: np.ndarray, bias: np.ndarray | None = None) -> None:
        """
        Mark Conv2d as parameter and load weights from NumPy arrays.
        After this is called `make_variable` cannot be called. Use this
        when you already have the weights of your convolutional layer.

        Parameters
        ----------
        weight : np.ndarray
                 Conv2d layer weights in shape
                 (out_channels x in_channels x kernel_size[0] x kernel_size[1])
        bias : np.ndarray | None
               Conv2d layer bias in shape (out_channels, ), only required when
               bias=True during initialization

        """
        self._load_weights(weight, bias)

    def __call__(
        self, input: gp.Parameter | gp.Variable, *, propagate_bounds: bool = True
    ) -> FormulationResult:
        """
        Forward pass your input, generate output and equations required for
        calculating the convolution. If `propagate_bounds` is True,
        the `input` is of type variable, and `load_weights` was called, then
        the bounds of the input are propagated to the output.

        Returns `FormulationResult` which can be unpacked as a output variable and list of equations.

        FormulationResult:
            - equations_created: ["set_output"]
            - variables_created: ["output", "weight", "bias"]
            - parameters_created: ["weight", "bias", "input_bounds", "output_bounds"]
            - sets_creates: ["conv_subset"]

        Note:
            - For backward compatibility, this result object can be unpacked as a tuple: `output, equations = conv2d(input)`.
            - `weight` and `bias` are available as variables if `make_variable` was called.
            - `weight` and `bias` are available as parameters if `load_weights` was called.
            - `input_bounds` and `output_bounds`are available as parameters if `propogate_bounds=True`.
            - The subset used to map input indices to output indices based on stride and padding.

        Parameters
        ----------
        input : gp.Parameter | gp.Variable
                input to the conv layer, must be in shape
                (batch x in_channels x height x width)
        propagate_bounds : bool = True
                If True, propagate bounds of the input to the output.
                Otherwise, the output variable is unbounded.

        Returns
        -------
        FormulationResult
        """
        return self._convolve(input, propagate_bounds)
