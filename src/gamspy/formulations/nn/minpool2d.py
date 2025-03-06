from __future__ import annotations

import gamspy as gp
from gamspy.formulations.nn.mpool2d import _MPool2d


class MinPool2d(_MPool2d):
    """
    Formulation generator for 2D Min Pooling in GAMS.

    Parameters
    ----------
    container : Container
        Container that will contain the new variable and equations.
    kernel_size : int | tuple[int, int]
        Filter size
    stride : int | tuple[int, int] | None
        Stride in the min pooling, it is equal to kernel_size if not provided
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
    >>> # 2x2 min pooling
    >>> mp1 = gp.formulations.MinPool2d(m, (2, 2))
    >>> inp = gp.Variable(m, domain=dim((10, 1, 24, 24)))
    >>> out, eqs = mp1(inp)
    >>> type(out)
    <class 'gamspy._symbols.variable.Variable'>
    >>> [len(x) for x in out.domain]
    [10, 1, 12, 12]

    """

    def __init__(
        self,
        container: gp.Container,
        kernel_size: int | tuple[int, int],
        stride: int | None = None,
        padding: int = 0,
        name_prefix: str | None = None,
    ):
        super().__init__(
            "min", container, kernel_size, stride, padding, name_prefix
        )

    def __call__(
        self,
        input: gp.Parameter | gp.Variable,
        big_m: int = 1000,
        propagate_bounds: bool = True,
    ) -> tuple[gp.Variable, list[gp.Equation]]:
        """
        Forward pass your input, generate output and equations required for
        calculating the min pooling. Returns the output variable and the list
        of equations required for the min pooling formulation. if propagate_bounds
        is True, it will also set the bounds for the output variable based on the input.
        It will also compute the big M value required for the pooling operation
        using the bounds.

        Parameters
        ----------
        input : gp.Parameter | gp.Variable
                input to the min pooling 2d layer, must be in shape
                (batch x in_channels x height x width)
        big_m: int
               Big M value that is required for the pooling operation.
               Default value: 1000.
        propagate_bounds: bool
                If True, it will set the bounds for the output variable
                based on the input.
                Default value: True

        Returns
        -------
        tuple[gp.Variable, list[gp.Equation]]

        """
        return super().__call__(input, big_m, propagate_bounds)
