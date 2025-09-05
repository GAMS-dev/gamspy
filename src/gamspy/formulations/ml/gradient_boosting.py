from __future__ import annotations

import importlib
import itertools
import uuid
from typing import TYPE_CHECKING, Any

import gamspy as gp
import gamspy.formulations.utils as utils
from gamspy.exceptions import ValidationError
from gamspy.formulations.ml.decision_tree_struct import DecisionTreeStruct
from gamspy.formulations.ml.regression_tree import RegressionTree

if TYPE_CHECKING:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.tree import DecisionTreeRegressor

    from gamspy import (
        Alias,
        Equation,
        Parameter,
        Set,
        Variable,
    )


class GradientBoosting:
    """
    Formulation generator for Gradient Boosted Trees in GAMSPy.

    Parameters
    ----------
    container : Container
        Container that will contain the new variable and equations.
    ensemble: GradientBoostingRegressor | list[DecisionTreeStruct]
        - A fitted `sklearn.ensemble.GradientBoostingRegressor` instance,
        - If `sklearn.ensemble.GradientBoostingRegressor` is not utilized, the ensembled trees information
          can be supplied via a list of `DecisionTreeStruct` dataclasse instances, which represents
          the same components as those in `sklearn.tree`.
          See :meth:`DecisionTreeStruct <gamspy.formulations.DecisionTreeStruct>` for details on required attributes.

    name_prefix : str | None
        Prefix for generated GAMSPy symbols, by default None which means
        random prefix. Using the same name_prefix in different formulations causes name
        conflicts. Do not use the same name_prefix again.

    bias: float | 1
        Bias term used to consolidate the final output using the contribution of each tree. This is generally
        the average of the output data used for training and it is useful when `ensemble` is a `list[DecisionTreeStruct]`.
        Otherwise, this is deduced from `ensemble` itself.

    learning_rate: float | 0.1
        Rate at which each tree's contribution is reduced, by default is 0.1. This is useful when `ensemble`
        is a `list[DecisionTreeStruct]`. Otherwise, this is deduced from `ensemble` itself.

    Examples
    --------
    >>> import gamspy as gp
    >>> import numpy as np
    >>> from gamspy.math import dim
    >>> np.random.seed(42)
    >>> m = gp.Container()
    >>> in_data = np.random.randint(0, 10, size=(5, 2))
    >>> out_data = np.random.randint(1, 3, size=(5, 1))
    >>> tree1_attribute = {
    ...     "capacity": 3,
    ...     "children_left": np.array([1, -1, -1]),
    ...     "children_right": np.array([2, -1, -1]),
    ...     "feature": np.array([0, -2, -2]),
    ...     "n_features": 2,
    ...     "threshold": np.array([4.0, -2.0, -2.0]),
    ...     "value": np.array([[-4.4408921e-17], [-8.0000000e-01], [2.0000000e-01]]),
    ... }
    >>> tree2_attribute = {
    ...     "capacity": 3,
    ...     "children_left": np.array([1, -1, -1]),
    ...     "children_right": np.array([2, -1, -1]),
    ...     "feature": np.array([0, -2, -2]),
    ...     "n_features": 2,
    ...     "threshold": np.array([4.0, -2.0, -2.0]),
    ...     "value": np.array([[-8.8817842e-17], [-6.4000000e-01], [1.6000000e-01]]),
    ... }
    >>> gb_trees = [gp.formulations.DecisionTreeStruct(**tree1_attribute), gp.formulations.DecisionTreeStruct(**tree2_attribute)]
    >>> dt_model = gp.formulations.GradientBoosting(m, gb_trees)
    >>> x = gp.Variable(m, "x", domain=dim((5, 2)), type="positive")
    >>> x.up[:, :] = 10
    >>> y, eqns = dt_model(x)
    >>> set_of_samples = y.domain[0]
    >>> set_of_samples.name
    'DenseDim5_1'

    """

    def __init__(
        self,
        container: gp.Container,
        ensemble: GradientBoostingRegressor | list[DecisionTreeStruct],
        name_prefix: str | None = None,
        bias: float = 1,
        learning_rate: float = 0.1,
    ):
        if not isinstance(container, gp.Container):
            raise ValidationError(f"{container} is not a gp.Container.")

        def _validate_ensemble(
            ensemble,
        ) -> list[DecisionTreeRegressor | DecisionTreeStruct]:
            if isinstance(ensemble, list) and all(
                isinstance(item, DecisionTreeStruct) for item in ensemble
            ):
                return ensemble
            else:
                try:
                    sklearn_boosting = importlib.import_module(
                        "sklearn.ensemble"
                    )
                    if isinstance(
                        ensemble, sklearn_boosting.GradientBoostingRegressor
                    ):
                        if not hasattr(ensemble, "estimators_"):
                            raise ValidationError(
                                f"{ensemble} must be a trained/fitted instance of >sklearn.ensemble.GradientBoostingRegressor<."
                            )
                        return ensemble.estimators_.flatten()
                    else:
                        raise ValidationError(
                            f"{ensemble} must be an instance of either >sklearn.ensemble.GradientBoostingRegressor< or a list of >DecisionTreeStruct<"
                        )
                except ModuleNotFoundError:
                    raise ValidationError(
                        ">sklearn.ensemble< module not found."
                    ) from None

        self.container = container

        if name_prefix is None:
            name_prefix = str(uuid.uuid4()).split("-")[0]

        self._name_prefix = name_prefix

        self._list_of_trees: list[RegressionTree] = []

        for tree in _validate_ensemble(ensemble):
            rt_instance = RegressionTree(
                self.container,
                regressor=tree,
                name_prefix=self._name_prefix,
            )
            self._list_of_trees.append(rt_instance)

        if isinstance(ensemble, list):
            # list[DecisionTreeStruct]
            self._bias = bias
            self._learning_rate = learning_rate
        else:  # GradientBoostingRegressor
            self._bias = ensemble.init_.constant_.flatten()[0]
            self._learning_rate = ensemble.learning_rate

    def __call__(
        self,
        input: gp.Parameter | gp.Variable,
        M: float | None = None,
    ) -> tuple[gp.Variable, list[gp.Equation]]:
        gb_out_list: list[gp.Variable] = []
        gb_eqn_list: list[gp.Equation] = []

        set_records_total: dict[
            Set | Alias | Parameter | Variable | Equation, Any
        ] = {}

        results = (
            regression_tree._yield_call(input, M)
            for regression_tree in self._list_of_trees
        )

        zipped_results = zip(*results)
        dt_outs = next(zipped_results)
        gb_out_list.extend(dt_outs)

        set_records_iter = next(zipped_results)
        set_of_output_dim = None
        for item, set_records_dict in set_records_iter:
            set_records_total.update(set_records_dict)
            set_of_output_dim = item

        self.container.setRecords(set_records_total)

        gb_eqn_list = list(itertools.chain.from_iterable(next(zipped_results)))

        set_of_samples = input.domain[0]

        out = gp.Variable._constructor_bypass(
            self.container,
            name=utils._generate_name("v", self._name_prefix, "real_output"),
            domain=[set_of_samples, set_of_output_dim],  # type: ignore
        )

        gb_eqn = gp.Equation._constructor_bypass(
            self.container,
            name=utils._generate_name("e", self._name_prefix, "gb_eqn"),
            domain=[set_of_samples, set_of_output_dim],  # type: ignore
            description="predicted out should be equal to the sum of gradient descent out times the learning rate.",
        )

        self.container._synch_with_gams(gams_to_gamspy=True)
        gb_eqn[...] = (
            self._bias + self._learning_rate * sum(gb_out_list) == out
        )
        gb_eqn_list.append(gb_eqn)

        return out, gb_eqn_list
