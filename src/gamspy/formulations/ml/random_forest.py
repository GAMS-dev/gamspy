from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import gamspy as gp
import gamspy.formulations.nn.utils as utils
from gamspy.exceptions import ValidationError
from gamspy.formulations.ml.regression_tree import RegressionTree

if TYPE_CHECKING:
    from sklearn.ensemble import RandomForestRegressor


class RandomForest(RegressionTree):
    """
    Formulation generator for Random Forests in GAMS.

    Parameters
    ----------
    container : Container
        Container that will contain the new variable and equations.
    ensemble: RandomForestRegressor | None
        - A fitted `sklearn.ensemble.RandomForestRegressor` object which is processed by the
          `DecisionTreeStruct` superclass to populate the underlying tree attributes.
        - If `None`, the tree structure is NOT automatically initialized.
          In this case, the user is responsible for manually populating the tree attributes
          inherited from `DecisionTreeStruct` before the formulation can be used.
          See :meth:`DecisionTreeStruct <gamspy.formulations.DecisionTreeStruct>` for details on required attributes.

    name_prefix : str | None
        Prefix for generated GAMSPy symbols, by default None which means
        random prefix. Using the same name_prefix in different formulations causes name
        conflicts. Do not use the same name_prefix again.

    """

    def __init__(
        self,
        container: gp.Container,
        ensemble: RandomForestRegressor | None = None,
        name_prefix: str | None = None,
    ):
        if not isinstance(container, gp.Container):
            raise ValidationError(f"{container} is not a gp.Container.")

        self.container = container
        self._indicator_vars = None

        if name_prefix is None:
            name_prefix = str(uuid.uuid4()).split("-")[0]

        self._name_prefix = name_prefix
        self.n_estimators = ensemble.n_estimators  # type: ignore
        self.ensemble = ensemble

    def __call__(self, input, M=None):
        output_dim = (
            self.ensemble.estimators_[0].tree_.value[:, :, 0].shape[-1]
        )
        set_of_samples, set_of_estimators, set_of_output_dim = (
            gp.math._generate_dims(
                self.container,
                dims=[input.shape[0], self.n_estimators, output_dim],
                alias=False,
            )
        )

        rf_out = gp.Variable(
            self.container,
            name=utils._generate_name("v", self._name_prefix, "output"),
            domain=[set_of_samples, set_of_estimators, set_of_output_dim],
        )

        out = gp.Variable(
            self.container,
            name=utils._generate_name("v", self._name_prefix, "output"),
            domain=[set_of_samples, set_of_output_dim],
        )

        rf_eqn_list = []
        for i in range(self.n_estimators):
            super().__init__(
                container=self.container,
                regressor=self.ensemble.estimators_[i],
            )  # type: ignore
            _out, dt_eqn = super().__call__(input, M)
            rf_out[:, i, :].records = _out.records
            rf_eqn_list += dt_eqn

        rf_eqn = gp.Equation(
            self.container,
            name=utils._generate_name("e", self._name_prefix, "rf_eqn"),
            description="perdicted out times number of estimators should be equal to the random forest out",
        )

        rf_eqn[...] = self.n_estimators * out == rf_out

        return out, rf_eqn_list.append(rf_eqn)
