***********************
Classic ML Formulations
***********************

.. meta::
   :description: GAMSPy User Guide
   :keywords: Machine Learning, User, Guide, GAMSPy, gamspy, GAMS, gams, mathematical modeling

GAMSPy allows you to integrate Regression Trees (Decision Trees that predict numerical values) directly into your optimization models.

Supported formulations:

:meth:`RegressionTree <gamspy.formulations.RegressionTree>`
-----------------------------------------------------------
Formulation generator for Regression Trees in GAMS. 

Here is an example which uses a trained decision tree to embed in your optimization model.

.. code-block:: python
    
    import gamspy as gp
    import numpy as np
    from gamspy.math import dim
    np.random.seed(42)
    m = gp.Container()
    in_data = np.random.randint(0, 10, size=(5, 2))
    out_data = np.random.randint(1, 3, size=(5, 1))
    tree_dict = {
        "capacity": 3,
        "children_left": np.array([1, -1, -1]),
        "children_right": np.array([2, -1, -1]),
        "feature": np.array([0, -2, -2]),
        "n_features": 2,
        "threshold": np.array([4.0, -2.0, -2.0]),
        "value": np.array([[1.8], [1.0], [2.0]]),
    }
    dt_model = gp.formulations.RegressionTree(m, tree_dict)
    x = gp.Variable(m, "x", domain=dim((5, 2)), type="positive")
    x.up[:, :] = 10
    y, eqns = dt_model(x)
    [d.name for d in y.domain]
    # ['DenseDim5_1', 'OutputDim']
