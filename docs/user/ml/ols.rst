******************************
Ordinary Least Squares Example
******************************


.. meta::
   :description: GAMSPy User Guide
   :keywords: Machine Learning, User, Guide, GAMSPy, gamspy, GAMS, gams, mathematical modeling, sparsity, performance


When trying to find a relationship between input data and a scalar output ,
it's often beneficial to start with simpler models to understand your system
better. One of the simplest methods for regression is ordinary least squares
(OLS).

In the following example, we have a training set consisting of 100
observations, each with two features. We create our labels by multiplying the
first feature by 3, the second feature by 4, and then adding 22. Afterwards, we
add some noise to our labels.

To find the relationship between labels and observations using OLS in GAMSPy,
you can follow these steps:

.. code-block:: python

   import numpy as np
   import gamspy as gp
   import sys

   # Two features, third dimension is for bias
   x = np.ones((100, 3))
   x[:, 0] = np.arange(100)
   x[:, 1] = np.random.rand(100) * 10
   x[:, 2] = 1 # this is the bias term

   noise = np.random.rand(100) * 10

   # target labels
   y = (x[:, 0] * 3) + (x[:, 1] * 4) + (x[:, 2] * 22) + noise

   m = gp.Container()
   # We know input and output, therefore they are parameters in GAMSPy
   X = gp.Parameter(m, name="X", domain=gp.math.dim((100, 3)), records=x)
   Y = gp.Parameter(m, name="Y", domain=gp.math.dim((100, 1)), records=y.reshape((100, 1)))

   # We want to learn coefficients
   w = gp.Variable(m, name="w", domain=gp.math.dim((3, 1)))

   z = gp.Variable(m, name="z")
   set_obj = gp.Equation(m, name="set_obj")
   # we want that coefficients that minimizes sum of squares of the residuals
   set_obj[...] = z == gp.math.vector_norm(Y - (X @ w)) ** 2

   OLS = gp.Model(
       m,
       name="OLS",
       equations=[set_obj],
       problem="QCP",
       sense=gp.Sense.MIN,
       objective=z,
   )

   OLS.solve(output=sys.stdout)
   w.records["level"]
   # 0     3.017193
   # 1     3.936441
   # 2    26.795031

You can see the estimated coefficients are close to their original values. In
this example, the problem is well-formed; however, in some cases, it is
necessary to add regularization terms to improve the model's performance or to
handle ill-posed problems. Regularization helps prevent overfitting by adding a
penalty to the objective function, encouraging simpler models with smaller
coefficient values.

Version with L2 regularization:

.. code-block:: python

   ...

   alpha = 2 # one can choose it looking at development set accuracy
   set_obj_reg = gp.Equation(m, name="set_obj_reg")
   set_obj_reg[...] = z == gp.math.vector_norm(Y - (X @ w)) ** 2 + \
       gp.math.vector_norm(alpha * w) ** 2


   OLS2 = gp.Model(
       m,
       name="OLS2",
       equations=[set_obj_reg],
       problem="QCP",
       sense=gp.Sense.MIN,
       objective=z,
   )

   OLS2.solve(output=sys.stdout)
   w.records["level"]
   # 0     3.062119
   # 1     4.474176
   # 2    21.108447

