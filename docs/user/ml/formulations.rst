*********************************
NN Formulations (ReLU, Conv2d...)
*********************************

.. meta::
   :description: GAMSPy User Guide
   :keywords: Machine Learning, User, Guide, GAMSPy, gamspy, GAMS, gams, mathematical modeling, sparsity, performance


A key challenge in embedding ML blocks in optimization is to come up with
formulations. To help you with this task, we implement the most commonly used
activation functions and layers so you can easily start optimizing.


.. _nn-formulations:

Layer Formulations
==================

GAMSPy provides several formulations to help you embed your neural network
structures into your optimization model. We started with formulations
for computer vision-related structures such as convolution and pooling
operations.

Convolution by definition requires no linearization, but it is tedious to write
down. Now you can use :meth:`Conv2d <gamspy.formulations.Conv2d>` to easily
embed your convolutional layer into your optimization model.

.. code-block:: python

   import gamspy as gp
   import numpy as np
   from gamspy.math import dim
   w1 = np.random.rand(2, 1, 3, 3)
   b1 = np.random.rand(2)
   m = gp.Container()
   conv1 = gp.formulations.Conv2d(
        m,
        in_channels=1,
        out_channels=2,
        kernel_size=(3, 3)
   )
   conv1.load_weights(w1, b1)
   maxpool1 = gp.formulations.MaxPool2d(m, (2, 2))

   inp = gp.Variable(m, domain=dim((10, 1, 24, 24)))
   out_var, eqs = conv1(inp)
   print([len(d) for d in out_var.domain])
   # [10, 2, 22, 22]

   out_var_2, eqs2 = maxpool1(out_var)
   print([len(d) for d in out_var_2.domain])
   # [10, 2, 11, 11]

   #out_var_2, eqs2 = maxpool1(out_var, big_m=10)

   out_var_3, eqs3 = gp.formulations.flatten_dims(out_var_2, [1, 2, 3])
   print([len(d) for d in out_var_3.domain])
   # [10, 242]

   w2 = np.random.rand(20, 242)
   b2 = np.random.rand(20)
   l1 = gp.formulations.Linear(m, 242, 20)
   l1.load_weights(w2, b2)

   out_var_4, eqs4 = l1(out_var_3)
   print([len(d) for d in out_var_4.domain])
   # [10, 20]



Supported formulations:

- :meth:`Linear <gamspy.formulations.Linear>`
- :meth:`Conv2d <gamspy.formulations.Conv2d>`
- :meth:`MaxPool2d <gamspy.formulations.MaxPool2d>`
- :meth:`MinPool2d <gamspy.formulations.MinPool2d>`
- :meth:`AvgPool2d <gamspy.formulations.AvgPool2d>`
- :meth:`flatten_dims <gamspy.formulations.flatten_dims>`


.. _pooling-linearization:

Max/Min Pooling Implementation
------------------------------

Max pooling and min pooling use big-M notation and binary variables to pick the
minimum or maximum. If the input has upper and lower bounds, big-M is calculated
using those bounds. Otherwise, big-M is 1000. Generated variables also contain
the upper and lower bounds if the input already has them.

The real formulation is more complicated because it is not scalar but indexed.
For simplicity, let us demonstrate the following example.  We will do min/max
pooling on a 4x4 input where the filter size is 2x2. From the 4 regions, we
will have 4 values. :math:`a, b, c, d` are variables in the blue region, most
likely continuous, but there is no restriction. :math:`p` is the variable that is
the output of the pooling operation on the blue region. Depending on the operation,
it is either min or max of the corresponding input points.

|
.. image:: ../images/pooling.png
  :align: center

|
The linearization of the :math:`p = \max(a,b,c,d)` is as follows:

.. math::

   p \geq a \\
   p \geq b \\
   p \geq c \\
   p \geq d \\
   p \leq a + M(1 - x_a) \\
   p \leq b + M(1 - x_b) \\
   p \leq c + M(1 - x_c) \\
   p \leq d + M(1 - x_d) \\
   x_a + x_b + x_c + x_d = 1 \\
   x_a, x_b, x_c, x_d \in \{0, 1\} \\

:math:`x_i` is a binary variable when set to 1 it means :math:`p = i`. 


The linearization of the :math:`p = \min(a,b,c,d)` is as follows:

.. math::

   p + M(1 - x_a) \geq a \\
   p + M(1 - x_b) \geq b \\
   p + M(1 - x_c) \geq c \\
   p + M(1 - x_d) \geq d \\
   p \leq a \\
   p \leq b \\
   p \leq c\\
   p \leq d\\
   x_a + x_b + x_c + x_d = 1 \\
   x_a, x_b, x_c, x_d \in \{0, 1\} \\

:math:`x_i` is a binary variable when set to 1 it means :math:`p = i`. 


.. _activation-functions:

Activation Functions
====================

One of the key reasons neural networks can learn a wide range of tasks is their
ability to approximate complex functions, including non-linear ones. Activation
functions are essential components that introduce nonlinearity to neural
networks. While understanding functions like ReLU may be straightforward,
integrating them into optimization models can be challenging. To assist you, we
have started with a small list of commonly used activation functions. So far,
we have implemented the following activation functions:

- :meth:`relu_with_binary_var <gamspy.math.relu_with_binary_var>`
- :meth:`relu_with_complementarity_var <gamspy.math.relu_with_complementarity_var>`
- :meth:`relu_with_sos1_var <gamspy.math.relu_with_sos1_var>`
- :meth:`softmax <gamspy.math.softmax>`
- :meth:`log_softmax <gamspy.math.log_softmax>`

Unlike other mathematical functions, these activation functions return a
variable instead of an expression. This is because ReLU cannot be represented
by a single expression. Directly writing ``y = max(x, 0)`` without reformulating
it would result in a Discontinuous Nonlinear Program (``DNLP``) model, which is
highly undesirable. Currently, you can either use
:meth:`relu_with_binary_var <gamspy.math.relu_with_binary_var>` to
introduce binary variables into your problem, or
:meth:`relu_with_complementarity_var <gamspy.math.relu_with_complementarity_var>`
to introduce nonlinearity.

Your model class changes depending on whether you want to embed a pre-trained
neural network into your problem or train a neural network within your problem.

If you are training a neural network, you must have non-linearity. Using
:meth:`relu_with_binary_var <gamspy.math.relu_with_binary_var>`
would result in a Mixed-Integer Nonlinear Program (``MINLP``) model. On the other
hand, :meth:`relu_with_complementarity_var <gamspy.math.relu_with_complementarity_var>`
would keep the model as a Nonlinear Program (``NLP``) model, though this does not
necessarily mean it will train faster.

If you are embedding a pre-trained neural network using
:meth:`relu_with_binary_var <gamspy.math.relu_with_binary_var>`,
you can maintain your model as a Mixed-Integer Programming (``MIP``) model,
provided you do not introduce nonlinearities elsewhere.


To read more about `classification of models
<https://www.gams.com/latest/docs/UG_ModelSolve.html#UG_ModelSolve_ModelClassificationOfModels>`_.

.. code-block:: python

   from gamspy import Container, Variable, Set
   from gamspy.math import relu_with_binary_var, log_softmax
   from gamspy.math import dim

   batch = 128
   m = Container()
   x = Variable(m, "x", domain=dim([batch, 10]))
   y, eqs1 = relu_with_binary_var(x)

   y2, eqs2 = log_softmax(x) # this creates variable and equations for you

Additionally, we offer our established functions that can also be used as
activation functions:

- :meth:`tanh <gamspy.math.tanh>`
- :meth:`sigmoid <gamspy.math.sigmoid>`

These functions return expressions like the other math functions. So, you
need to create equations and variables yourself.

.. code-block:: python

   from gamspy import Container, Variable, Set, Equation
   from gamspy.math import dim, tanh

   batch = 128
   m = Container()
   x = Variable(m, "x", domain=dim([batch, 10]))
   eq = Equation(m, "set_y", domain=dim([batch, 10]))
   y = Variable(m, "y", domain=dim([batch, 10]))
   eq[...] = y == tanh(x)

