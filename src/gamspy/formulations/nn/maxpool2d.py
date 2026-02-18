from __future__ import annotations

from typing import TYPE_CHECKING

from gamspy.formulations.nn.mpool2d import _MPool2d

if TYPE_CHECKING:
    import gamspy as gp
    from gamspy.formulations.result import FormulationResult


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
    name_prefix : str | None
        Prefix for generated GAMSPy symbols, by default None which means
        random prefix. Using the same name_prefix in different formulations causes name
        conflicts. Do not use the same name_prefix again.

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
        name_prefix: str | None = None,
    ):
        super().__init__("max", container, kernel_size, stride, padding, name_prefix)

    def __call__(
        self,
        input: gp.Parameter | gp.Variable,
        big_m: int = 1000,
        propagate_bounds: bool = True,
    ) -> FormulationResult:
        """
        Forward pass your input, generate output and equations required for
        calculating the max pooling. Returns the output variable and the list
        of equations required for the max pooling formulation. if propagate_bounds
        is True, it will also set the bounds for the output variable based on the input.
        It will also compute the big M value required for the pooling operation
        using the bounds.

        Returns `FormulationResult` which can be unpacked as a output variable and list of equations.

        FormulationResult:
            - equations_created: ["lte", "gte", "pick_one"]
            - variables_created: ["output", "aux_variable"]
            - parameters_created: ["bigM", "output_lb", "output_ub"]
            - sets_created: ["in_out_matching_1", "in_out_matching_2"]

        Note:
            - For backward compatibility, this result object can be unpacked as a tuple: `output, equations = maxpool(input)`.
            - `aux_variable` is the binary variable selecting the max element.
            - `output_lb` and `output_ub` are available as parameters if `propagate_bounds=True`.
            - `in_out_matching_1`is the subset used to map input indices to output indices based on stride and padding.
            - `in_out_matching_2` is the subset used specifically for bound propagation.
            It gets created only if `propogate_bounds=True`.

        Parameters
        ----------
        input : gp.Parameter | gp.Variable
                input to the max pooling 2d layer, must be in shape
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
        FormulationResult
        """
        return super().__call__(input, big_m, propagate_bounds)
