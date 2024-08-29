.. _external_equations:

******************
External Equations
******************

Model Interface
---------------

Sometimes, you need to represent a complicated **non-linear** relationship in
your constraints. However, the relationship doesn't have an easy-to-write algebraic 
expression. If you can provide function evaluations and first derivatives, you can 
embed your relationship in GAMSPy using External Equations.


.. admonition:: Information

   This documentation is a shortened version of the GAMS documentation on
   `External Equations <https://www.gams.com/latest/docs/UG_ExternalEquations.html>`_.
   Since many parts are skipped, we suggest reading the original documentation
   after reading this one.

External equations are **not** compatible with all non-linear solvers. This is
because external module, the library you would need to implement, is limited to
providing functionality for evaluating functions (including their first
derivatives) at a specific point. Some solvers can also benefit from the use of
second-order derivatives by utilizing the Hessian Vector Product
:math:`\nabla^2f(x)v` that can be supplied by the external module. As a
result, solvers that require analysis of the algebraic structure of the model
instance **cannot** work with external equations. This limitation affects
deterministic global solvers, as noted in the "Global" column of `this table
<https://www.gams.com/latest/docs/S_MAIN.html#SOLVERS_MODEL_TYPES>`_.


.. warning::

   This feature requires a solid understanding of programming in C/C++ or Fortran,
   compilation, and linking processes.

An **external equation** delegates the handling of the relationship between 
variables to an external module. In this context, the
relationship between the variables is not explicitly defined within your system
but is instead governed by the external module. For example, you might specify
that variables `x` and `y` are related, but the nature of that relationship is
determined by the external module.

An **external module** is a library responsible for managing all external
equations. It defines and handles the relationships between variables specified
in these equations. For instance, the external module might implement a
function like :math:`y = sin(x)`.


Here is an example of how an external equation is defined:

.. code-block:: python

    import gamspy as gp
    
    m = gp.Container()
    x = gp.Variable(m)
    y = gp.Variable(m)
    eq = gp.Equation(m, type="external")
    eq[...] = 1 * x + 2 * y == 1 # This certainly is not a linear equation

The coefficients indicate the index of the variable within the external module. For
instance, in the expression `1*x`, `x` represents the first variable in the
external module, and similarly, `y` represents the second variable. The
right-hand side of the equation denotes that this is the first external
equation in the system. Consequently, attempting the following would result in
a failure:

.. code-block:: python

    import gamspy as gp

    m = gp.Container()
    x = gp.Variable(m)
    y = gp.Variable(m)
    eq1 = gp.Equation(m, type="external")
    eq2 = gp.Equation(m, type="external")
    # you cannot have two equations with the same index
    eq1[...] = 1 * x + 2 * y == 1
    eq2[...] = 1 * x + 2 * y == 1 # this should be 2

Skipping an equation index also does not work:

.. code-block:: python

    eq1[...] = 1 * x + 2 * y == 1
    eq2[...] = 1 * x + 2 * y == 3

When you have `n` external equations, the indices of these equations must
include every number from 1 to `n`, inclusive.

Variable indices must be consistent as well:

.. code-block:: python

    eq1[...] = 1 * x + 2 * y == 1
    eq2[...] = 2 * x + 3 * y == 2

The variable `x` was designated as the first variable in the external module. In
subsequent equations, its index must remain consistent, meaning it should still
be indexed as 1. It is not necessary to include every variable in each equation.

Let's assume we want to represent :math:`y_1 = sin(x_1)` and :math:`y_2 = cos(x_2)`

.. code-block:: python

    import gamspy as gp
    m = gp.Container()
    y1 = gp.Variable(m)
    y2 = gp.Variable(m)
    x1 = gp.Variable(m)
    x2 = gp.Variable(m)

    eq1 = gp.Equation(m, type="external")
    eq2 = gp.Equation(m, type="external")

    eq1[...] = 1*x1 + 3*y1 == 1
    eq2[...] = 2*x2 + 4*y2 == 2


.. admonition:: A small note on what we are representing

   Actually, instead of representing :math:`y_1 = \sin(x_1)`, we represent it
   as :math:`\sin(x_1) - y_1 = 0`. When we evaluate the function, we are asked
   to compute :math:`\sin(x_1) - y_1`. You'll notice that when this expression
   does not equal zero, the equation is not satisfied. However, the solver will
   adjust the values using derivatives to restore feasibility. Therefore, the
   derivative of :math:`\sin(x_1) - y_1` is taken with respect to both
   :math:`x_1` and :math:`y_1`. Specifically, the derivative with respect to
   :math:`x_1` is :math:`\cos(x_1)`, and the derivative with respect to
   :math:`y_1` is :math:`-1`.


Finally, we need to provide the name of the external module in the model.

.. code-block:: python

    ...
    model = gp.Model(
        container=m,
        equations=m.getEquations(),
        problem="nlp",
        sense="min",
        objective=y1 + y2,
        external_module="mylibrary",
    )

Since no file extension was specified for ``external_module``, GAMSPy will automatically search for the
appropriate file extension based on the operating system: ``.dll`` on Windows, ``.so`` on Linux, and ``.dylib`` on macOS.
The next step is generating the library.


Programming Interface
---------------------

The rest of the documentation remains unchanged, so please refer to the
`Programming Interface
<https://www.gams.com/latest/docs/UG_ExternalEquations.html#UG_ExternalEquations_ProgrammingInterface>`_
for more detailed information. In brief, your task is to download the
`geheader.h <https://www.gams.com/latest/testlib_ml/geheader.h>`_ file and
implement the ``gefunc`` function as specified within it. To assist you, we've
provided `sample external module
<https://github.com/GAMS-dev/gamspy/tree/develop/tests/integration/external_module>`_.
Starting with this template is much easier than building everything from
scratch. The ``mylib.cpp`` file contains the library code, and a ``CMakeLists.txt``
file is included to help you build the module. The example referenced in the
documentation can be found in ``example.py``. After compiling ``mylib.cpp`` into a
library, place the library next to ``example.py`` and run the script. We
understand that implementing external equations can be challenging, and we're
actively exploring automations for specific cases to ease this process.
