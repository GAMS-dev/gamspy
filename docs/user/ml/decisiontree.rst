*********************
Decision Tree Example
*********************


.. meta::
   :description: GAMSPy User Guide
   :keywords: Machine Learning, User, Guide, GAMSPy, gamspy, GAMS, gams, mathematical modeling, sparsity, performance


Decision Trees are widely used for regression tasks. They split data into branches based on feature values, making them easy to interpret and visualize. Common applications include fraud detection, medical diagnosis, and customer segmentation. Their simplicity and ability to handle both numerical and categorical data make them popular in predictive modeling.

Learn more about Decision Trees `here <https://en.wikipedia.org/wiki/Decision_tree>`_.

We start with generating a simple dataset. We have three features.
# TODO: use a real-world open example rather than constructed one.


.. code-block:: python

    import numpy as np
    import gamspy as gp
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.datasets import make_regression

    
    # Generate dataset
    X, y = make_regression(
        n_samples=100,     # Number of samples
        n_features=3,      # Number of input features
        n_targets=1,       # Number of output values
        noise=0.1,         # Add some noise 
        random_state=42    # For reproducibility
    )
        
    regressor = DecisionTreeRegressor(random_state=42)
    regressor.fit(X, y)

    print("Input shape (X):", X.shape)
    # Input shape (X): (100, 3)
    print("Output shape (y):", y.shape
    # Input shape (y): (100,)

    from gamspy.formulations import RegressionTree

    m = gp.Container()

    dt_regressor = RegressionTree(m, regressor)

    sample_size = 10
    nfeatures = 3
    x = gp.Variable(m, name="input_variable", type="positive", domain=gp.math.dim((sample_size, nfeatures)))

    obj_var, eqns = dt_regressor(x)

