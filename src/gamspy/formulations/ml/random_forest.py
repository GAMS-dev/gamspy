from __future__ import annotations

import importlib
import uuid
from typing import TYPE_CHECKING

import gamspy as gp
import gamspy.formulations.nn.utils as utils
from gamspy.exceptions import ValidationError
from gamspy.formulations.ml.decision_tree_struct import DecisionTreeStruct
from gamspy.formulations.ml.regression_tree import RegressionTree

if TYPE_CHECKING:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor


class RandomForest(RegressionTree):
    """
    Formulation generator for Random Forests in GAMS.

    Parameters
    ----------
    container : Container
        Container that will contain the new variable and equations.
    ensemble: RandomForestRegressor | None
        - A fitted `sklearn.ensemble.RandomForestRegressor` instance,
        - If `sklearn.ensemble.RandomForestRegressor` is not utilized, the ensembled trees information
          can be supplied via a list of `DecisionTreeStruct` dataclasse instances, which represents
          the same components as those in `sklearn.tree`.
          See :meth:`DecisionTreeStruct <gamspy.formulations.DecisionTreeStruct>` for details on required attributes.

    name_prefix : str | None
        Prefix for generated GAMSPy symbols, by default None which means
        random prefix. Using the same name_prefix in different formulations causes name
        conflicts. Do not use the same name_prefix again.

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
    ...     "capacity": 7,
    ...     "children_left": np.array([ 1, -1,  3, -1,  5, -1, -1]),
    ...     "children_right": np.array([ 2, -1,  4, -1,  6, -1, -1]),
    ...     "feature": np.array([ 1, -2,  0, -2,  1, -2, -2]),
    ...     "n_features": 2,
    ...     "threshold": np.array([ 2. , -2. ,  5.5, -2. ,  8.5, -2. , -2. ]),
    ...     "value": np.array([[1.6 ],[1.  ],[1.75],[2.  ],[1.5 ],[1.  ],[2.  ]])
    ... }
    >>> tree2_attribute = {
    ...     "capacity": 3,
    ...     "children_left": np.array([ 1, -1, -1]),
    ...     "children_right": np.array([ 2, -1, -1]),
    ...     "feature": np.array([ 0, -2, -2]),
    ...     "n_features": 2,
    ...     "threshold": np.array([ 1.5, -2. , -2. ]),
    ...     "value": np.array([[1.4],[1. ],[2. ]])
    ... }
    >>> forest = [gp.formulations.DecisionTreeStruct(**tree1_attribute), gp.formulations.DecisionTreeStruct(**tree2_attribute)]
    >>> dt_model = gp.formulations.RandomForest(m, forest)
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
        ensemble: RandomForestRegressor | list[DecisionTreeStruct],
        name_prefix: str | None = None,
    ):
        if not isinstance(container, gp.Container):
            raise ValidationError(f"{container} is not a gp.Container.")

        def _validate_ensemble(
            ensemble,
        ) -> list[DecisionTreeRegressor | DecisionTreeStruct]:
            type_err = ValidationError(
                f"{ensemble} must be an instance of either >sklearn.ensemble.RandomForestRegressor< or a list of >DecisionTreeStruct<"
            )

            if isinstance(ensemble, list) and all(
                isinstance(item, DecisionTreeStruct) for item in ensemble
            ):
                return ensemble
            else:
                try:
                    sklearn_forest = importlib.import_module(
                        "sklearn.ensemble"
                    )
                    if isinstance(
                        ensemble, sklearn_forest.RandomForestRegressor
                    ):
                        if not hasattr(ensemble, "estimators_"):
                            raise ValidationError(
                                f"{ensemble} must be a trained/fitted instance of >sklearn.ensemble.RandomForestRegressor<."
                            )
                        return ensemble.estimators_
                    else:
                        raise type_err
                except ModuleNotFoundError:
                    raise type_err from None

        self.list_of_trees = _validate_ensemble(ensemble)
        self.container = container

        if name_prefix is None:
            name_prefix = str(uuid.uuid4()).split("-")[0]

        self._name_prefix = name_prefix

    def __call__(
        self,
        input: gp.Parameter | gp.Variable,
        M: float | None = None,
        **kwargs,
    ) -> tuple[gp.Variable, list[gp.Equation]]:
        rf_eqn_list = []
        rf_out_collection: dict[str, gp.Variable] = {}
        for i, tree in enumerate(self.list_of_trees):
            super().__init__(
                container=self.container,
                regressor=tree,
                name_prefix=self._name_prefix,
            )
            rf_out_collection[f"estimator{i}"], dt_eqn, set_of_output_dim = (
                super().__call__(input, M, is_random_forest=True)
            )
            rf_eqn_list += dt_eqn

        set_of_samples = input.domain[0]

        out = gp.Variable._constructor_bypass(
            self.container,
            name=utils._generate_name("v", self._name_prefix, "real_output"),
            domain=[set_of_samples, set_of_output_dim],
        )

        rf_eqn = gp.Equation._constructor_bypass(
            self.container,
            name=utils._generate_name("e", self._name_prefix, "rf_eqn"),
            domain=[set_of_samples, set_of_output_dim],
            description="perdicted out times number of estimators should be equal to the random forest out",
        )

        self.container._synch_with_gams(gams_to_gamspy=True)
        rf_eqn[...] = len(self.list_of_trees) * out == sum(
            rf_out_collection.values()
        )
        rf_eqn_list.append(rf_eqn)

        return out, rf_eqn_list
