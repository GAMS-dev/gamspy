from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import gamspy as gp
import gamspy.formulations.nn.utils as utils
from gamspy.exceptions import ValidationError
from gamspy.formulations.ml.dtStruct import DecisionTreeStruct
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
            ensemble_type = type(ensemble)
            if (
                ensemble_type.__name__ == "RandomForestRegressor"
                and ensemble_type.__module__.startswith("sklearn.ensemble")
            ):
                if not hasattr(ensemble, "estimators_"):
                    raise ValidationError(
                        f"{ensemble} must be a trained/fitted instance of >sklearn.tree.RandomForestRegressor<."
                    )
                return ensemble.estimators_
            elif isinstance(ensemble, list) and all(
                isinstance(item, DecisionTreeStruct) for item in ensemble
            ):
                return ensemble
            else:
                raise ValidationError(
                    f"{ensemble} must be an instance of either >sklearn.tree.RandomForestRegressor< or a list of >DecisionTreeStruct<"
                )

        self.list_of_trees = _validate_ensemble(ensemble)
        self.container = container

        if name_prefix is None:
            name_prefix = str(uuid.uuid4()).split("-")[0]

        self._name_prefix = name_prefix

    def __call__(self, input, M=None) -> tuple[gp.Variable, list[gp.Equation]]:
        rf_eqn_list = []
        rf_out_collection = {}
        for i, tree in enumerate(self.list_of_trees):
            super().__init__(
                container=self.container,
                regressor=tree,
                name_prefix=self._name_prefix,
            )
            rf_out_collection[f"estimator{i}"], dt_eqn = super().__call__(
                input, M
            )
            rf_eqn_list += dt_eqn

        set_of_samples = input.domain[0]
        set_of_output_dim = self.container.data.get("set_of_output_dim")

        out = gp.Variable(
            self.container,
            name=utils._generate_name("v", self._name_prefix, "real_output"),
            domain=[set_of_samples, set_of_output_dim],
        )

        rf_eqn = gp.Equation(
            self.container,
            name=utils._generate_name("e", self._name_prefix, "rf_eqn"),
            domain=[set_of_samples, set_of_output_dim],
            description="perdicted out times number of estimators should be equal to the random forest out",
        )

        rf_eqn[...] = len(self.list_of_trees) * out == sum(
            rf_out_collection.values()
        )
        rf_eqn_list.append(rf_eqn)

        return out, rf_eqn_list
