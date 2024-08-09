.. _external_equations:

******************
External Equations
******************

Model Interface
---------------

Sometimes, you need to represent a complicated **non-linear** relationship in
your constraints. However the relationship does not have an easy to write
algebra. If you can provide function evaluation and first derivatives, you can
embed your relationship in GAMSPy using External Equations.


.. admonition:: Information

   This documentation is a shortened version of
   `External Equations <https://www.gams.com/latest/docs/UG_ExternalEquations.html>`_.
   Since we skip many parts, we suggest reading the original documentation
   after reading this one.

External equations do **not** work with all the non-linear solvers. This is
because external module, the library you would need to implement, is limited to
providing functionality for evaluating functions (including their first
derivatives) at a specific point. As a result, solvers that require analysis of
the algebraic structure of the model instance **cannot** work with external
equations. This limitation affects deterministic global solvers, as noted in
the "Global" column of `this table
<https://www.gams.com/latest/docs/S_MAIN.html#SOLVERS_MODEL_TYPES>`_. However,
stochastic global solvers are capable of working with external equations.


.. warning::

   This feature requires a solid understanding of programming in C/C++ or Fortran,
   compilation, and linking processes.

An **external equation** is an equation where the handling of the relationship
between variables is delegated to an external module. In this context, the
relationship between the variables is not explicitly defined within your system
but is instead governed by the external module. For example, you might specify
that variables `x` and `y` are related, but the nature of that relationship is
determined by the external module.

An **external module** is a library responsible for managing all external
equations. It defines and handles the relationships between variables specified
in these equations. For instance, the external module might implement a
function like `y = sin(x)`.


Here is an example, how an external equation is defined:

.. code-block:: python

    import gamspy as gp
    m = gp.Container()
    x = gp.Variable(m, "x")
    y = gp.Variable(m, "y")
    eq = gp.Equation(m, "eq", type="external")
    # This certainly is not a line equation
    eq[...] = 1*x + 2*y == 1

Coefficients indicate the index of the variable within the external module. For
instance, in the expression `1*x`, `x` represents the first variable in the
external module, and similarly, `y` represents the second variable. The
right-hand side of the equation denotes that this is the first external
equation in the system. Consequently, attempting the following would result in
a failure:

.. code-block:: python

    import gamspy as gp
    m = gp.Container()
    x = gp.Variable(m, "x")
    y = gp.Variable(m, "y")
    eq = gp.Equation(m, "eq", type="external")
    eq2 = gp.Equation(m, "eq2", type="external")
    # you cannot have two equations with the same index
    eq[...] = 1*x + 2*y == 1
    eq2[...] = 1*x + 2*y == 1 # this should be 2

Leaving an equation index empty also does not work:

.. code-block:: python

    eq[...] = 1*x + 2*y == 1
    eq2[...] = 1*x + 2*y == 3

When you have `n` external equations, the indices of these equations must
include every number from 1 to `n`, inclusive.

Variable indices must be consistent as well:

.. code-block:: python

    eq[...] = 1*x + 2*y == 1
    eq2[...] = 2*x + 3*y == 2

The variable `x` was designated as the first variable in the external module. In
subsequent equations, its index must remain consistent, meaning it should still
be indexed as 1. It is not necessary to include every variable in each equation.

Let's assume we want to represent `y1 = sin(x1)` and `y2 = cos(x2)`

.. code-block:: python

    import gamspy as gp
    m = gp.Container()
    y1 = gp.Variable(m, "y1")
    y2 = gp.Variable(m, "y2")
    x1 = gp.Variable(m, "x1")
    x2 = gp.Variable(m, "x2")

    eq1 = gp.Equation(m, "eq1", type="external")
    eq2 = gp.Equation(m, "eq2", type="external")

    eq1[...] = 1*x1 + 3*y1 == 1
    eq2[...] = 2*x2 + 4*y2 == 2


Finally, we need to provide the name of the external module in the model.

.. code-block:: python

    ...
    model = gp.Model(
        container=m,
        name="sincos",
        equations=m.getEquations(),
        problem="NLP",
        sense="min",
        objective=y1 + y2,
        external_module="mylibrary",
    )

Since no file extension was specified, GAMS will automatically search for the
appropriate file extension based on the operating system: `.DLL` on Windows and
`.SO` on Linux. Next step is generating the library.


Programming Interface
---------------------

As the rest of the documentation remains unchanged, please refer to the
`Programming Interface
<https://www.gams.com/latest/docs/UG_ExternalEquations.html#UG_ExternalEquations_ProgrammingInterface>`_
for further details. In summary, you need to download the `geheader.h` file and
implement the `gefunc` function defined within it. Afterward, compile it into a
library. We acknowledge that this aspect of implementing external equations has
a steep learning curve, and we are exploring possible automations for certain
special cases.
