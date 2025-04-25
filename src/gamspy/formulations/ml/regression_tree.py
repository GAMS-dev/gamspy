from __future__ import annotations

import uuid

# from typing import TYPE_CHECKING
import numpy as np

import gamspy as gp

# import gamspy.formulations.nn.utils as utils
from gamspy.exceptions import ValidationError

# from gamspy.math import dim


class RegressionTree:
    """
    Formulation generator for Regression Trees in GAMS.

    Parameters
    ----------
    container : Container
        Container that will contain the new variable and equations.
    regressor: dict | DecisionTreeRegressor
        Trained decision tree. This could either be a sklearn.tree.DecisionTreeRegressor object or a dictionary of the form ...
    input_data : numpy.ndarray
        Input data ...
    name_prefix : str | None
        Prefix for generated GAMSPy symbols, by default None which means
        random prefix. Using the same name_prefix in different formulations causes name
        conflicts. Do not use the same name_prefix again.

    Examples
    --------
    >>> import gamspy as gp
    >>> import numpy as np
    >>> from gamspy.math import dim

    """

    def __init__(
        self,
        container: gp.Container,
        regressor: dict,  # | DecisionTreeRegressor,
        input_data: np.ndarray,
        name_prefix: str | None = None,
    ):
        COMPLETE_TREE_DICT = {
            "children_left",
            "children_right",
            "feature",
            "threshold",
            "value",
            "capacity",
            "n_features",
        }
        if isinstance(regressor, dict):
            if len(check_keys := regressor.keys() - COMPLETE_TREE_DICT) != 0:
                raise ValidationError(
                    f"Dictionary containing the information from the Regression Tree is either incomplete or the keys are wrong. KEYS: {check_keys}"
                )
            self._tree_dict = regressor

        # TODO: Add sanity checks for populating the sklearn dictionary?
        else:
            self._tree_dict = {
                "children_left": regressor.tree_.children_left,
                "children_right": regressor.tree_.children_right,
                "feature": regressor.tree_.feature,
                "threshold": regressor.tree_.threshold,
                "value": regressor.tree_.value[:, :, 0],
                "capacity": regressor.tree_.capacity,
                "n_features": regressor.tree_.n_features,
            }

        self.container = container

        # TODO: Add sanity checks for input_data
        self.input_data = input_data

        if name_prefix is None:
            name_prefix = str(uuid.uuid4()).split("-")[0]

        self._name_prefix = name_prefix
