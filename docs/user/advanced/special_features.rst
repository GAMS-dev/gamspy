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

    myvar.scale[i,j] = c

The scale factor ``c`` is a number or a numerical expression that evaluates to a number. Note that 
the default scale factor is 1.

Note that there is one scale value for each individual component of a multidimensional variable.

Assume that :math:`c` is the scale factor of a variable :math:`V_u`. Assume further, that the variable 
seen by the algorithm is :math:`V_a`. Then we have: :math:`V_a = V_u/c`. This means that each variable 
as seen by the user is *divided* by the scale factor.

For example, consider the following code snippet: ::

    from gamspy import Container, Variable, Equation
    m = Container()
    
    x1 = Variable(m, "x1", type="positive")
    x2 = Variable(m, "x2", type="positive")
    
    eq = Equation(m, "eq")
    
    eq = 200*x1 + 0.5*x2 <= 5
    
    x1.up[...] = 0.01
    x2.up[...] = 10
    x1.scale[...] = 0.01
    x2.scale[...] = 10

By setting ``x1.scale`` to 0.01 and ``x2.scale`` to 10, the model seen by the solver is: ::

    from gamspy import Container, Variable, Equation
    m = Container()
    
    xPrime1 = Variable(m, "xPrime1", type="positive")
    xPrime2 = Variable(m, "xPrime2", type="positive")
    
    eq = Equation(m, "eq")
    
    eq = 2*xPrime1 + 5*xPrime2 <= 5
    
    xPrime1.up[...] = 1
    xPrime2.up[...] = 1

Note that the solver does not see the variables ``x1`` or ``x2``, but rather the scaled (and 
better-behaved) variables ``xPrime1`` and ``xPrime2``. Note further, that upper and lower bounds 
on variables are automatically scaled in the same way as the variable itself.

.. warning::
    Discrete variables cannot be scaled.

Variable x;
x.scale = 0.1;
display x.stage;
The output is:
----      3 VARIABLE x.scale = 0.100
The field .scale has to be in a certain range ( >1e-20 and no special value), but this is only checked at model generation time. The field .prior can be any number and even +inf (but no other special values). For further information on .prior, see section Setting Priorities for Branching. For an introduction to variable and equation fields, see sections Variable Attributes and Equation Attributes respectively.

Scaling Equations
------------------

The scale factor of an equation is defined using the equation attribute 
:meth:`scale <gamspy.Equation.scale>` in the following way: ::

    mzeqn.scale[i,j] = d

The scale factor ``d`` is a number or a numerical expression that evaluates to a number. Note 
that the default scale factor is 1.

Assume that :math:`d` is the scale factor of an equation :math:`G_u`. Assume further, that the 
equation seen by the algorithm is :math:`G_a`. Then we have: :math:`G_a = G_u/d`. This means 
that each equation as seen by the user is *divided* by the scale factor.

For example, consider the following equations: ::

    from gamspy import Container, Variable, Equation
    m = Container()
    
    y1 = Variable(m, "y1", type="positive")
    y2 = Variable(m, "y2", type="positive")
    
    eq1 = Equation(m, "eq1")
    eq2 = Equation(m, "eq2")
    
    eq1 = 200*y1 + 100*y2 <= 500
    eq2 = 3*y1 - 4*y2 >= 6

By setting ``eq1.scale[...] = 100``, the model seen by the solver is: ::

    from gamspy import Container, Variable, Equation
    m = Container()
    
    y1 = Variable(m, "y1", type="positive")
    y2 = Variable(m, "y2", type="positive")
    
    eqPrime1 = Equation(m, "eqPrime1")
    eq2      = Equation(m, "eq2")
    
    eqPrime1 = 2*y1 + 1*y2 <= 5
    eq2      = 3*y1 - 4*y2 >= 6



.. note::
    The user may have to perform a combination of equation and variable scaling to obtain a 
    well-scaled model.

Consider the following example: ::

    from gamspy import Container, Variable, Equation
    m = Container()
    
    x1 = Variable(m, "x1", type="positive")
    x2 = Variable(m, "x2", type="positive")
    
    eq1 = Equation(m, "eq1")
    eq2 = Equation(m, "eq2")
    
    eq1 = 100*x1 + 5*x2 >= 20
    eq2 = 50*x1 - 10*x2 <= 5
    
    x1.up[...] = 0.2
    x2.up[...] = 1.5

Setting the following scale values: ::

    x1.scale[...]  = 0.1
    eq1.scale[...] = 5
    eq2.scale[...] = 5

will result in the solver seeing the following well-scaled model: ::

    from gamspy import Container, Variable, Equation
    m = Container()
    
    xPrime1 = Variable(m, "xPrime1", type="positive")
    x2 = Variable(m, "x2", type="positive")
    
    eqPrime1 = Equation(m, "eqPrime1")
    eqPrime2 = Equation(m, "eqPrime2")
    
    eqPrime1 = 2*xPrime1 + x2 >= 4
    eqPrime2 = xPrime1 - 2*x2 <= 1
    
    xPrime1.up[...] = 0.2
    x2.up[...] = 1.5


Scaling Derivatives
---------------------

In nonlinear models the derivatives also need to be well-scaled. Assume that the 
derivatives in the model of the user are denoted by :math:`d(G_u)/d(V_u)`. Assume 
further, that the derivatives in the scaled model seen by the algorithm are denoted 
by :math:`d(G_a)/d(V_a)`. Then we have: :math:`\mathbf{d(G_a)/d(V_a) = d(G_u)/d(V_u) \cdot c/d}`, 
where :math:`c` is the scale factor for the variable and :math:`d` is the scale 
factor for the equation. 

The user may affect the scaling of derivatives by scaling both the equation and variable involved.

Scaling Data
-------------

Scaling input data may contribute considerably towards achieving a well-scaled model. We recommend 
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

For more information, see `References and Links <references-and-links>`_ at the 
end of this section.

Introduction to Conic Programming
-----------------------------------


Conic programs can be thought of as generalized linear programs with the additional 
nonlinear constraint :math:`x \in C`, where :math:`C` is required to be a convex cone. 
The resulting class of problems is known as <em>conic optimization</em> and has the 
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
solvers have the capability to identify the conic constraints in a 
`QCP [GAMS documentation] <https://www.gams.com/latest/docs/UG_ModelSolve.html#UG_ModelSolve_modelclassificationQCP>`_ 
model even if it is not in perfect form but can be easily reformulated to fit in the 
described form. However, some solvers (namely MOSEK) expect the conic constraints to 
be precisely in the form given above. Moreover, such solvers have other requirements 
(e.g. disjunctive cones) that can be easily fulfilled by simple reformulation steps. 
Much progress is expected on the solver side in the coming years, so we don't go into 
much detail here.

Observe that we could formulate conic problems as regular NLPs using the following 
constraints: 

- Quadratic cone: ::

      x['1'] >= sqrt(Sum(i.where[~ sameAs(i,'1')], sqr(x[i])))

- Rotated quadratic cone: ::

      2*x['1']*x['2'] >= Sum(i.where[~sameAs(i,'1') & ~sameas(i,'2')], sqr(x[i]))

  Here x['1'] and x['2'] are positive variables.

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
may be written in GAMSPy using the equations: ::
    
    defobj = Sum(n, d[n]/x[n]) == obj
    e1     = Sum(n, a[n]*x[n]) <= b

    orig = Model(m, "orig", equations=[defobjc,e1], 
                 problem=Problem.NLP, 
                 sense=Sense.Min, 
                 objective=obj)

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

    defobjc    = Sum(n, d[n]*t[n]) == obj
    e1         = Sum(n, a[n]*x[n]) <= b
    coneqcp[n] = t[n]*x[n] >= 1

    cqcp = Model(m, "cqcp", equations=[defobjc,e1,coneqcp], 
                 problem=Problem.QCP, 
                 sense=Sense.Min, 
                 objective=obj)

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

The GAMSPy formulation using conic equations is: ::

    defobjc        = Sum(n, d[n]*t[n]) == obj
    e1             = Sum(n, a[n]*x[n]) <= b
    e2[n]          = z[n] == sqrt(2)
    coneperfect[n] = 2*x[n]*t[n] >= sqr(z[n])
    
    cperfect = Model(m, "cperfect", equations=[defobjc,e1,e2,coneqcp], 
                     problem=Problem.QCP, 
                     sense=Sense.Min, 
                     objective=obj)
                     
    t.lo[n] = 0
    x.lo[n] = l[n]
    x.up[n] = u[n]

The complete model is listed below::
    
    from gamspy import Container, Set, Parameter, Variable, Equation, Model, Sum, Problem, Sense, Options
    import gamspy.math as math
    m = Container()

    n = Set(m, "n", records=[("i" + str(i), i) for i in range(1, 11)])

    d = Parameter(m, "d", domain=n)
    a = Parameter(m, "a", domain=n)
    l = Parameter(m, "l", domain=n)
    u = Parameter(m, "u", domain=n)
    b = Parameter(m, "b")

    d[n] = math.uniform(1, 2)
    a[n] = math.uniform(10, 50)
    l[n] = math.uniform(0.1, 10)
    u[n] = l[n] + math.uniform(0, 12 - l[n])

    x = Variable(m, "x", domain=n)
    x.l[n] = math.uniform(l[n], u[n])
    b = Sum(n, x.l[n] * a[n])

    t = Variable(m, "t", domain=n)
    z = Variable(m, "z", domain=n)
    obj = Variable(m, "obj")

    defobjc = Equation(m, "defobjc")
    defobj = Equation(m, "defobj")
    e1 = Equation(m, "e1")
    e2 = Equation(m, "e2", domain=n)
    coneqcp = Equation(m, "coneqcp", domain=n)
    coneperfect = Equation(m, "coneperfect", domain=n)
    conenlp = Equation(m, "conenlp", domain=n)

    defobjc[...] = Sum(n, d[n] * t[n]) == obj
    defobj[...] = Sum(n, d[n] / x[n]) == obj
    e1[...] = Sum(n, a[n] * x[n]) <= b
    coneqcp[n] = t[n] * x[n] >= 1
    e2[n] = z[n] == math.sqrt(2)
    coneperfect[n] = 2 * x[n] * t[n] >= math.sqr(z[n])

    cqcp = Model(m,"cqcp",equations=[defobjc, e1, coneqcp],problem=Problem.QCP,sense=Sense.MIN,objective=obj)

    cperfect = Model(m,"cperfect",equations=[defobjc, e1, e2, coneqcp],problem=Problem.QCP,sense=Sense.MIN,objective=obj)

    orig = Model(m,"orig",equations=[defobjc, e1],problem=Problem.NLP,sense=Sense.MIN,objective=obj)

    t.lo[n] = 0
    x.lo[n] = l[n]
    x.up[n] = u[n]

    cqcp.solve(options=Options(qcp="cplex"))
    cperfect.solve(options=Options(qcp="mosek"))
    orig.solve(options=Options(qcp="cplex"))



Sample Conic Models in GAMS
----------------------------

..
    TODO: GAMSPy pendants?

- [`EMFL <https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_emfl.html>`_]: A multiple facility location problem,
- [`FDESIGN <https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_fdesign.html>`_]: Linear Phase Lowpass Filter Design,
- [`IMMUN <https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_immun.html>`_]: Financial Optimization: Risk Management,
- [`PMEANVAR <https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_pmeanvar.html>`_]: Mean-Variance Models with variable upper and lower Bounds,
- [`QP7 <https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_qp7.html>`_]: A portfolio investment model using rotated quadratic cones (quadratic program using a Markowitz model),
- [`ROBUSTLP <https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_robustlp.html>`_]: Robust linear programming as an SOCP,
- [`SPRINGCHAIN <https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_springchain.html>`_]: Equilibrium of System with Piecewise Linear Spring,
- [`TRUSSM <https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_trussm.html>`_]: Truss Toplogy Design with Multiple Loads

.. _references-and-links:

References and Links
--------------------

- A. Ben-Tal and A. Nemirovski,
  Lectures on Modern Convex Optimization: Analysis, Algorithms, and Engineering Applications,
  MPS/SIAM Series on Optimization, SIAM Press, 2001.
- M. Lobo, L. Vandenberghe, S. Boyd and H. Lebret, <a href="http://stanford.edu/~boyd/papers/socp.html">
  Applications of Second-Order Cone Programming</a>, Linear Algebra and its 
  Applications, 284:193-228, November 1998, Special Issue on Linear Algebra in 
  Control, Signals and Image Processing.
- MOSEK ApS, [MOSEK Modeling Cookbook](https://docs.mosek.com/modeling-cookbook/index.html), 2015.
- G. Pataki G and S. Schmieta, The DIMACS Library of Semidefinite-Quadratic-Linear 
  Programs. Tech. rep., Computational Optimization Research Center, Columbia 
  University, 2002.
- Seventh Dimacs Implementation Challenge on Semidefinite and Related Optimization Problems.
