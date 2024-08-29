.. _special_features:

*******************************************
Special Features for Mathematical Programs
*******************************************

..
    TODO
    By setting priorities users may specify an order for choosing variables to branch on during 
    a branch and bound search for MIP models. Without priorities the MIP algorithm will internally 
    determine which variable is the most suitable to branch on. Priorities for individual variables 
    may be used only if the model attribute ``prioropt`` is set to 1; the respective GAMSPy statement is:


Model Scaling 
==============

..
    TODO
    The Scale Option
    -----------------

Scaling Variables
------------------

The scale factor of a variable is defined using the variable attribute 
:meth:`scale <gamspy.Variable.scale>` in the following way: ::

    myvar.scale[i, j] = c

The scale factor ``c`` is a number or a numerical expression that evaluates to a number. Note that 
the default scale factor is 1.

Note that there is one scale value for each individual component of a multidimensional variable.

Assume that :math:`c` is the scale factor of a variable :math:`V_u`. Assume further, that the variable 
seen by the algorithm is :math:`V_a`. Then we have: :math:`V_a = V_u/c`. This means that each variable, 
as seen by the user, is *divided* by the scale factor.

For example, consider the following code snippet: ::

    from gamspy import Container, Variable, Equation

    m = Container()
    x1 = Variable(m)
    x2 = Variable(m)
    
    eq = Equation(m)
    
    eq = 200 * x1 + 0.5 * x2 <= 5
    
    x1.up = 0.01
    x2.up = 10
    x1.scale = 0.01
    x2.scale = 10

By setting ``x1.scale`` to 0.01 and ``x2.scale`` to 10, the model seen by the solver is: ::

    from gamspy import Container, Variable, Equation

    m = Container()
    xPrime1 = Variable(m)
    xPrime2 = Variable(m)
    
    eq = Equation(m, "eq")
    
    eq = 2 * xPrime1 + 5 * xPrime2 <= 5
    
    xPrime1.up = 1
    xPrime2.up = 1

Note that the solver does not see the variables ``x1`` or ``x2``, but rather the scaled (and 
better-behaved) variables ``xPrime1`` and ``xPrime2``. Note further that upper and lower bounds 
on variables are automatically scaled in the same way as the variable itself.

.. warning::
    Discrete variables cannot be scaled.

.. note::
    ``enable_scaling`` option in the solve statement must be set to True for GAMSPy to apply 
    user-specified variable and equation scaling factors.


Scaling Equations
------------------

The scale factor of an equation is defined using the equation attribute 
:meth:`scale <gamspy.Equation.scale>` in the following way: ::

    mzeqn.scale[i, j] = d

The scale factor ``d`` is a number or a numerical expression that evaluates to a number. Note 
that the default scale factor is 1.

Assume that :math:`d` is the scale factor of an equation :math:`G_u`. Assume further that the 
equation seen by the algorithm is :math:`G_a`. Then we have: :math:`G_a = G_u/d`. This means 
that each equation as seen by the user is *divided* by the scale factor.

For example, consider the following equations: ::

    from gamspy import Container, Variable, Equation

    m = Container()
    y1 = Variable(m)
    y2 = Variable(m)
    
    eq1 = Equation(m)
    eq2 = Equation(m)
    
    eq1[...] = 200 * y1 + 100 * y2 <= 500
    eq2[...] = 3 * y1 - 4 * y2 >= 6

By setting ``eq1.scale = 100``, the model seen by the solver is: ::

    from gamspy import Container, Variable, Equation

    m = Container()
    y1 = Variable(m)
    y2 = Variable(m)
    
    eqPrime1 = Equation(m)
    eq2      = Equation(m)
    
    eqPrime1 = 2 * y1 + 1 * y2 <= 5
    eq2      = 3 * y1 - 4 * y2 >= 6

.. note::
    The user may have to perform a combination of equation and variable scaling to obtain a 
    well-scaled model.

Consider the following example: ::

    from gamspy import Container, Variable, Equation

    m = Container()
    y1 = Variable(m)
    y2 = Variable(m)
    
    eq1 = Equation(m)
    eq2 = Equation(m)
    
    eq1[...] = 100 * x1 + 5 * x2 >= 20
    eq2[...] = 50 * x1 - 10 * x2 <= 5
    
    x1.up = 0.2
    x2.up = 1.5

Setting the following scale values: ::

    x1.scale  = 0.1
    eq1.scale = 5
    eq2.scale = 5

will result in the solver seeing the following well-scaled model: ::

    from gamspy import Container, Variable, Equation
    m = Container()
    
    xPrime1 = Variable(m)
    x2 = Variable(m)
    
    eqPrime1 = Equation(m)
    eqPrime2 = Equation(m)
    
    eqPrime1[...] = 2 * xPrime1 + x2 >= 4
    eqPrime2[...] = xPrime1 - 2 * x2 <= 1
    
    xPrime1.up = 2
    x2.up = 1.5


Scaling Derivatives
---------------------

In nonlinear models the derivatives also need to be well-scaled. Assume that the 
derivatives in the model of the user are denoted by :math:`d(G_u)/d(V_u)`. Assume 
further, that the derivatives in the scaled model seen by the algorithm are denoted 
by :math:`d(G_a)/d(V_a)`. Then we have: :math:`\mathbf{d(G_a)/d(V_a) = d(G_u)/d(V_u) \cdot c/d}`, 
where :math:`c` is the scale factor for the variable and :math:`d` is the scale 
factor for the equation. 

The user may affect the scaling of derivatives by scaling both the equation and the variable involved.

Scaling Data
-------------

Scaling input data may contribute considerably to achieving a well-scaled model. We recommend 
users to try to define the units of the input data such that the largest values expected for decision 
variables and their marginals is under a million, if possible.

For example, in US agriculture about 325 million acres are cropped and the corn crop is 9-10 billion 
bushels per year. When defining production data, we could enter land in 1000's of acres and all other 
resources in 1000's of units. We could also define the corn crop in millions of bushels. The data will 
be simultaneously scaled, hence if resource endowments are quoted in 1000's, corn yields are divided 
by 1000. This scaling results in a corn production variable in the units of millions. Consumption 
statistics would need to be scaled accordingly. Money units could also be in millions or billions 
of dollars. Such data scaling generally greatly reduces the disparity of coefficients in the model.

Conic Programming
==================

Conic programming models minimize a linear function over the intersection of an 
affine set and the product of nonlinear cones. The problem class involving second 
order (quadratic) cones is known as Second Order Cone Programs (SOCP). These are 
nonlinear convex problems that include linear and (convex) quadratic programs as 
special cases.

Conic programs allow the formulation of a wide variety of application models, 
including problems in engineering and financial management. Examples  are portfolio 
optimization, Truss topology design in structural engineering, Finite Impulse 
Response (FIR) filter design and signal processing, antenna array weight design, 
grasping force optimization,  quadratic programming, robust linear programming and
norm minimization problems.

Introduction to Conic Programming
-----------------------------------

Conic programs can be thought of as generalized linear programs with the additional 
nonlinear constraint :math:`x \in C`, where :math:`C` is required to be a convex cone. 
The resulting class of problems is known as *conic optimization* and has the 
following form:

.. math::
    \begin{array}{rl} 
       \text{minimize}   &    c^Tx             \\
       \text{subject to} &    Ax  \le r^c,     \\
                         &    x \in [l^x, u^x] \\
                         &    x \in C          \\
    \end{array}

where :math:`A\in \mathbb{R}^{m \times n}` is the constraint matrix, :math:`x \in \mathbb{R}^n` 
the decision variable and :math:`c \in \mathbb{R}^n` the objective 
function cost coefficients. The vector :math:`r^c \in \mathbb{R}^m` represents the 
right-hand side and the vectors :math:`l^x, u^x \in \mathbb{R}^n` are lower and upper 
bounds on the decision variable :math:`x`.

Now partition the set of decision variables :math:`x` into sets :math:`S^t, t=1,...,k`, 
such that each decision variables :math:`x` is a member of at most one set :math:`S^t`. 
For example, we could have

.. math::

    S^1 = (x_1, x_4, x_7) \quad \text{and} \quad S^2 = (x_6, x_5, x_3, x_2).

Let :math:`x_{S^t}` denote the variables :math:`x` belonging to set :math:`S^t`. Then define

.. math::

    C := \left \{ x \in \mathbb{R}^n : x_{S^t} \in C_t, t=1,...,k \right \},

where :math:`C_t` must have one of the following forms:

- **Quadratic cone** (also referred to as Lorentz or ice cream cone):

.. math::

    C_t = \left \{ x \in \mathbb{R}^{n^t} : x_1 \ge
          \sqrt{\sum_{j=2}^{n^t}x_j^2} \right \}.

- **Rotated quadratic cone** (also referred to as hyperbolic constraints):

.. math::
    C_t = \left \{ x \in \mathbb{R}^{n^t} : 2x_1x_2 \ge
          \sum_{j=3}^{n^t}x_j^2, ~x_1,x_2 \ge 0 \right \}.

These two types of cones allow the formulation of quadratic, quadratically 
constrained and many other classes of nonlinear convex optimization problems.

Implementation of Conic Constraints in GAMSPy
---------------------------------------------

The recommended way to write conic constraints is by using a quadratic formulation. Many 
solvers have the capability to identify the conic constraints in a ``QCP``
model even if it is not in perfect form but can be easily reformulated to fit in the 
described form. However, some solvers (namely MOSEK) expect the conic constraints to 
be precisely in the form given above. Moreover, such solvers have other requirements 
(e.g. disjunctive cones) that can be easily fulfilled by simple reformulation steps. 
Much progress is expected on the solver side in the coming years, so we don't go into 
much detail here.

Observe that we could formulate conic problems as regular NLPs using the following 
constraints: 

- Quadratic cone: ::

      x['1'] >= gp.math.sqrt(gp.Sum(i.where[~i.sameAs('1')], gp.math.sqr(x[i])))

- Rotated quadratic cone: ::

      2 * x['1'] * x['2'] >= gp.Sum(i.where[~i.sameAs('1') & ~i.sameas('2')], sqr(x[i]))

  Here ``x['1']`` and ``x['2']`` are positive variables.

The following example illustrates the different formulations for conic programming 
problems. Note that a conic optimizer usually outperforms a general NLP method for 
the reformulated (NLP) cone problems.


Example
-----------

Consider the following example, which illustrates the use of rotated conic 
constraints. We will give reformulations of the original problem in regular NLP form 
and in conic form (with conic constraints).

The original problem is:

.. math::
    \text{minimize}  \; & \sum_{i=1}^n \frac{d_i}{x_i} \\
    \text{subject to}\; & a\,x \le b             \\
                        & x_i \in [l_i,u_i], & i=1,\ldots,n
                    
where :math:`x \in \mathbb{R}^n` is the decision variable, 
:math:`d, a, l, u \in \mathbb{R}^n` are parameters with :math:`l_i>0` and 
:math:`d_i \ge 0` and :math:`b \in \mathbb{R}` is a scalar parameter. The original model 
may be written in GAMSPy using the following equations: ::
    
    obj = gp.Sum(n, d[n] / x[n])
    e1[...] = gp.Sum(n, a[n] * x[n]) <= b

    orig = Model(m, equations=[e1], problem=Problem.NLP, sense=Sense.Min, objective=obj)

    x.lo[n] = l[n]
    x.up[n] = u[n]

We can write an equivalent QCP formulation by using the substitution :math:`t_i=1/x_i` 
in the objective function and adding a constraint. As we are dealing with a 
minimization problem, :math:`d_i \ge 0` and :math:`x_i \ge l_i > 0`, we can relax the 
equality :math:`t_ix_i=1` into an inequality :math:`t_ix_i \ge 1` which results in an 
equivalent problem with a convex feasible set:

.. math::
    \text{minimize}  \; & \sum_{i=1}^n d_i t_i  \\
    \text{subject to}\; & a\,x \le b           \\
                        & t_i x_i \ge 1, & i=1,\ldots,n \\
                        & x \in [l,u], \\
                        & t \ge 0, \\

where :math:`t \in \mathbb{R}^n` is a new decision variable. The GAMSPy formulation
of this QCP is: ::

    obj = gp.Sum(n, d[n] * t[n])
    e1[...] = Sum(n, a[n] * x[n]) <= b
    coneqcp[n] = t[n] * x[n] >= 1

    cqcp = Model(
        m, equations=[e1, coneqcp], problem=Problem.QCP, sense=Sense.Min, objective=obj
    )

    t.lo[n] = 0
    x.lo[n] = l[n]
    x.up[n] = u[n]

Note that the constraints :math:`t_i x_i \ge 1` are almost in rotated conic form. If 
we introduce a variable :math:`z \in \mathbb{R}^n` with :math:`z_i = \sqrt{2}` then we 
can reformulate the problem using conic constraints as:

.. math::
    \text{minimize}  \; & \sum_{i=1}^n d_i t_i  \\
    \text{subject to}\; & a\,x \le b           \\
                        & z_i = \sqrt{2},      & i=1,\ldots,n \\
                        & 2 t_i x_i \ge z_i^2, & i=1,\ldots,n \\
                        & x \in [l,u],\\
                        & t \ge 0, \\

The GAMSPy formulation using conic equations is as follows: ::

    obj = gp.Sum(n, d[n] * t[n])
    e1[...] = Sum(n, a[n] * x[n]) <= b
    e2[n] = z[n] == gp.math.sqrt(2)
    coneperfect[n] = 2 * x[n] * t[n] >= gp.math.sqr(z[n])

    cperfect = Model(
        m, equations=[e1, e2, coneqcp], problem=Problem.QCP, sense=Sense.Min, objective=obj
    )

    t.lo[n] = 0
    x.lo[n] = l[n]
    x.up[n] = u[n]

The complete model is listed below::
    
    from gamspy import (
        Container,
        Set,
        Parameter,
        Variable,
        Equation,
        Model,
        Sum,
        Problem,
        Sense,
        Options,
    )
    import gamspy.math as math

    m = Container()

    n = Set(m, "n", records=range(1, 11))
    d = Parameter(m, domain=n)
    a = Parameter(m, domain=n)
    l = Parameter(m, domain=n)
    u = Parameter(m, domain=n)
    b = Parameter(m)

    d[n] = math.uniform(1, 2)
    a[n] = math.uniform(10, 50)
    l[n] = math.uniform(0.1, 10)
    u[n] = l[n] + math.uniform(0, 12 - l[n])

    x = Variable(m, domain=n)
    x.l[n] = math.uniform(l[n], u[n])
    b = Sum(n, x.l[n] * a[n])

    t = Variable(m, domain=n)
    z = Variable(m, domain=n)

    e1 = Equation(m)
    e2 = Equation(m, domain=n)
    coneqcp = Equation(m, domain=n)
    coneperfect = Equation(m, domain=n)
    conenlp = Equation(m, domain=n)

    objc = Sum(n, d[n] * t[n])
    obj = Sum(n, d[n] / x[n])
    e1[...] = Sum(n, a[n] * x[n]) <= b
    coneqcp[n] = t[n] * x[n] >= 1
    e2[n] = z[n] == math.sqrt(2)
    coneperfect[n] = 2 * x[n] * t[n] >= math.sqr(z[n])

    cqcp = Model(
        m, equations=[e1, coneqcp], problem=Problem.QCP, sense=Sense.MIN, objective=objc
    )

    cperfect = Model(
        m, equations=[e1, e2, coneqcp], problem=Problem.QCP, sense=Sense.MIN, objective=objc
    )

    orig = Model(m, equations=[e1], problem=Problem.NLP, sense=Sense.MIN, objective=obj)

    t.lo[n] = 0
    x.lo[n] = l[n]
    x.up[n] = u[n]

    cqcp.solve(solver="cplex")
    cperfect.solve(solver="mosek")
    orig.solve(solver="conopt")


Other Conic Programs with MOSEK
-------------------------------

In addition to quadratic and rotated quadratic cones which can be solved by most QCP solver, the 
`MOSEK <https://www.gams.com/latest/docs/S_MOSEK.html#MOSEK_CONIC_PROGRAMMING>`_ solver
has the capability to solve other conic problems, namely problems with *power cones*, *exponential cones*, and
*semidefinite cones*. For the first two, the structure of the cones are tried to be extracted from the non-linear
algebra. For the latter, the PSD variable needs to the description ``PSDMATRIX``. Here are four  examples for these
cone types: ::

    import gamspy as gp

    m = gp.Container()

    x0 = gp.Variable(m, type="positive")
    x1 = gp.Variable(m, type="positive")
    x2 = gp.Variable(m, type="positive")
    x3 = gp.Variable(m) 
    x4 = gp.Variable(m) 
    x5 = gp.Variable(m) 

    e1 = gp.Equation(m)
    e2 = gp.Equation(m)
    e3 = gp.Equation(m)

    obj = x3 + x4 - x0

    e1[...] = x0 + x1 + 0.5 * x2 == 2
    e2[...] = x0 ** 0.2 * x1 ** 0.8 >= gp.math.abs(x3)
    e3[...] = x2 ** 0.4 * x5 ** 0.6 >= gp.math.abs(x4)

    x5.fx = 1

    power_cone1 = gp.Model(m, equations=[e1, e2, e3], objective=obj, sense="max", problem="dnlp")
    power_cone1.solve(solver="mosek")


::

    import gamspy as program

    m = gp.Container()
    i = gp.Set(m, name="i", records=range(5))

    x0 = gp.Variable(m, type="positive")
    x1 = gp.Variable(m, type="positive")
    x2 = gp.Variable(m, domain=i)

    e1 = gp.Equation(m)
    e2 = gp.Equation(m)

    obj = gp.Sum(i, gp.Ord(i) * x2[i]) - x0
    e1[...] = x0 + x1 == 2
    e2[...] = gp.math.sqrt(x0 * x1) >= gp.math.sqrt(gp.Sum(i, gp.math.sqr(x2[i])));

    power_cone2 = gp.Model(m, equations=[e1, e2], objective=obj, sense="max", problem="nlp")

    power_cone2.solve(solver="mosek")

::

    import gamspy as program

    m = gp.Container()

    x0 = gp.Variable(m, type="positive")
    x1 = gp.Variable(m, type="positive")
    x2 = gp.Variable(m)

    e1 = gp.Equation(m)
    e2 = gp.Equation(m)

    obj = x0 + x1
    e1[...] =  x0 + x1 + x2 == 1
    e2[...] = x0 >= x1 * gp.math.exp(x2 / x1)

    exp_cone = gp.Model(m, equations=[e1, e2], objective=obj, sense="min", problem="nlp")

    x1.l = 1 # avoid division by 0 at initial point
    exp_cone.solve(solver="mosek")

::

    import gamspy as gp
    import numpy as np

    m = gp.Container()

    i = gp.Set(m, name="i", records=range(3))
    ip = gp.Alias(m, name="ip", alias_with=i)

    barX = gp.Variable(m, domain=[i, i], description="PSDMATRIX")
    x = gp.Variable(m, domain=i, type="positive")

    barAobj = gp.Parameter(
        m, domain=[i, i], description="coefficients of barX in objective"
    )
    barAe1 = gp.Parameter(m, domain=[i, i], description="coefficients of barX in e1")
    barAe2 = gp.Parameter(m, domain=[i, i], description="coefficients of barX in e2")

    barAobj.setRecords(np.array([[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]]))
    barAe1[i, i] = 1.0  # identity matrix
    barAe2[i, ip] = 1.0  # all-one matrix

    e1 = gp.Equation(m)
    e2 = gp.Equation(m)
    e3 = gp.Equation(m)

    obj = gp.Sum([i, ip], barAobj[i, ip] * barX[i, ip]) + x["0"]
    e1[...] = 1 == gp.Sum([i, ip], barAe1[i, ip] * barX[i, ip]) + x["0"]
    e2[...] = 0.5 == gp.Sum([i, ip], barAe2[i, ip] * barX[i, ip]) + x["1"] + x["2"]
    e3[...] = -gp.math.sqr(x["0"]) + gp.math.sqr(x["1"]) + gp.math.sqr(x["2"]) <= 0

    psd_cone = gp.Model(
        m, equations=[e1, e2, e3], objective=obj, sense="min", problem="qcp"
    )

    psd_cone.solve(solver="mosek")
