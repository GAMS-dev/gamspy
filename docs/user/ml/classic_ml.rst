***********************
Classic ML Formulations
***********************

.. meta::
   :description: GAMSPy User Guide
   :keywords: Machine Learning, User, Guide, GAMSPy, gamspy, GAMS, gams, mathematical modeling

We often require classical machine learning approaches for our optimization workflows.
GAMSPy currently provides a formulation for decision trees that can be directly embedded in your optimization models.
We will roll out additional formulations for other classical machine learning algorithms in the future.

Supported formulations
======================

:meth:`RegressionTree <gamspy.formulations.RegressionTree>`
-----------------------------------------------------------

When a Decision Tree is trained to predict numerical values (rather than class labels), it is referred to as a :meth:`Regression Tree <gamspy.formulations.RegressionTree>`.
Here is an example where we train a Regression tree and use the formulation to embed in an optimization model.

It should be noted we are using the `sklearn.tree.DecisionTreeRegressor`_. for convenience. You can also provide the information from the trained decision tree as a dictionary.

.. image:: ../images/regressionTree.png
  :align: center

.. code-block:: python

   import gamspy as gp
   import numpy as np
   from gamspy.math import dim
   from sklearn.tree import DecisionTreeRegressor

   np.random.seed(42)

   X = np.array(
      [
         [2, 3],
         [3, 1],
         [1, 2],
         [5, 6],
         [6, 4],
      ]
   )
   y = np.array([10, 10, 10, 15, 33])

   regressor = DecisionTreeRegressor(random_state=42)
   # This is the regressor that you want to include in
   # your optimization model
   regressor.fit(X, y)

   m = gp.Container()
   # Formulation requires the regressor
   dt_formulation = gp.formulations.RegressionTree(m, regressor)
   # Let's create a sample input
   m_input = gp.Parameter(m, "input", domain=dim((5, 2)), records=X)

   # y_pred = regressor(m_input) and eqns are the equations that
   # create this relation
   y_pred, eqns = dt_formulation(m_input)

   predict_values = gp.Model(
      m,
      "regressionTree",
      equations=eqns,
      problem="MIP",
   )
   predict_values.solve()
   print(y_pred.toDense().flatten())
   # [10. 10. 10. 15. 33.]


.. _sklearn.tree.DecisionTreeRegressor: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
