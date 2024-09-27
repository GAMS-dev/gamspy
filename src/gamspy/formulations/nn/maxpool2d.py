from __future__ import annotations

import gamspy as gp
from gamspy.formulations.nn.mpool2d import _MPool2d


class MaxPool2d(_MPool2d):
    """
    Formulation generator for 2D Max Pooling in GAMS.

    Parameters
    ----------
    container : Container
        Container that will contain the new variable and equations.
    kernel_size : int | tuple[int, int]
        Filter size
    stride : int | tuple[int, int] | None
        Stride in the max pooling, it is equal to kernel_size if not provided
    padding : int | tuple[int, int]
        Amount of padding to be added to input, by default 0

    Examples
    --------
    >>> import gamspy as gp
    >>> from gamspy.math import dim
    >>> m = gp.Container()
    >>> # 2x2 max pooling
    >>> mp1 = gp.formulations.MaxPool2d(m, (2, 2))
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
    ):
        super().__init__("max", container, kernel_size, stride, padding)

    def __call__(
        self, input: gp.Parameter | gp.Variable, big_m: int = 1000
    ) -> tuple[gp.Variable, list[gp.Equation]]:
        """
        Forward pass your input, generate output and equations required for
        calculating the max pooling. Returns the output variable and the list
        of equations required for the max pooling formulation.

        Parameters
        ----------
        input : gp.Parameter | gp.Variable
                input to the max pooling 2d layer, must be in shape
                (batch x in_channels x height x width)
        big_m: int
               Big M value that is required for the pooling operation.
               Default value: 1000.

        Returns
        -------
        tuple[gp.Variable, list[gp.Equation]]

        """
        return super().__call__(input, big_m)
