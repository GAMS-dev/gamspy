.. _model_instance:

*************
Frozen Models
*************

In rare cases, the GAMSPy model generation time dominates the solver solution time and GAMSPy itself becomes the bottleneck in an optimization application.
If the model needs to be solved for multiple data sets (scenarios) the *frozen model* facility of GAMSPy can improve the overall performance.
When one :meth:`solve <gamspy.Model.solve>`s a :meth:`Model <gamspy.Model>`, GAMSPy *generates* the *model instance* from the model algebra with the current data,
passes this to the solver, and after completion populates the variable and equation symbols in the container with the solution of the model instance.
This model instance is created during the :meth:`solve <gamspy.Model.solve>` operation and discarded afterwards. This means that the next :meth:`solve <gamspy.Model.solve>` call will create a new model instance. The lifetime of a model instance of a *frozen model* goes beyond this and provides
a controlled way of modifying a model instance and solving the resulting problem repeatedly in the most efficient way, by communicating whenever possible only the frozen model
changes to the solver. Additionally, for continuous models, like LPs, some solvers can do a *hot start* from the previous basis solution without the need for refactorization.
The terms "frozen model" and "model instance" are used interchangeably in the remainder of this text.

Frozen models are particularly useful when many *scenarios* of a model need to be solved where most of the data, especially the model rim, stays constant and only some
(exogenous) parameters or bounds are subject to change. Typical examples are Monte-Carlo simulations with varying objective function coefficients.

The :meth:`freeze <gamspy.Model.freeze>` method turns a regular model into a frozen model. This method actually generates/instantiates the model instance and requires in addition to
model algebra all data of relevant model symbols. The :meth:`freeze <gamspy.Model.freeze>` method gets a list 
of :meth:`parameters <gamspy.Parameter>` and :meth:`variable <gamspy.Variable>` or :meth:`equation <gamspy.Equation>` attributes such as 
:meth:`.lo <gamspy.Variable.lo>`, :meth:`.up <gamspy.Variable.up>` and :meth:`.fx <gamspy.Variable.fx>`. 
Equation and variable attributes :meth:`.l <gamspy.Equation.l>` and
:meth:`.m <gamspy.Equation.m>` in the list of modifiables are mainly used for starting
non-linear models from different starting points.


The :meth:`solve <gamspy.Model.solve>` method takes the data of the modifiables from the container symbols and updates the frozen model. Variable and equation attributes
are applied in a direct manner. Moreover, the data of the modifiable *parameters* is taken from the container symbol and applied to the model instance. Behind the scenes,
the :meth:`freeze <gamspy.Model.freeze>` method turned these modifiable parameters into fixed *variables* with the name of the parameter plus ``_var``. After a solve 
these variables become accessible in the container. The marginal of these variables can provide useful sensitivity information about the parameter setting. In addition,
the container will contain the primal and dual solution with respect to the regular variables and equations of the model instance together with many model attributes including 
:meth:`status <gamspy.Model.status>`, 
:meth:`solve_status <gamspy.Model.solve_status>`,
:meth:`objective_value <gamspy.Model.objective_value>`,
:meth:`objective_estimation <gamspy.Model.objective_estimation>`,
:meth:`solve_model_time <gamspy.Model.solve_model_time>`,
:meth:`num_iterations <gamspy.Model.num_iterations>`, and
:meth:`num_nodes_used <gamspy.Model.num_nodes_used>`.

.. note::
    Understanding the technical details that happen under the hood is not necessary for using frozen model effectively. Nevertheless, if you need to debug a frozen model,
    these details can help to interpret the representation of the model instance inside the solver. For non-linear models the fixed variables corresponding to the modifiables
    directly enter (non-linear) expressions in the model algebra. The automatic differentiation methods of GAMS and GAMSPy take these into account. For linear models,
    terms like ``a/sqr(b)*x``, with the modifiable parameters ``a`` and ``b`` and variable ``x``, become in a sense non-linear and need to be *linearized*. The Taylor expansion
    at the initial point allows such a linearization: :math:`f(x) = f(x.l) + \frac{d f(x.l)}{d x}(x-x.l)`. If *f* was linear with respect to the original variables, this is an
    exact representation and no error term is required. So our term ``a/sqr(b)*x`` will be represented (for better readability the ``a_var`` as been shortened to ``a``, similar for ``b``) as: 
    
    ::

        a.l/sqr(b.l)*x.l + a.l/sqr(b.l)*x + x.l/sqr(b.l)*a - 2*a.l*x.l/power(b.l,3)*b
        
    or with an initial point of ``x.l=2`` and fixed modifiers ``a.fx=6`` and ``b.fx=3`` we get
    
    ::

        4/3 + 2/3*x + 2/9*a - 8/9*b

    The following example demonstrates how to look at the solver representation of such a model instance:
    
    ::

        import gamspy as gp

        m = gp.Container()
        x = gp.Variable(m, "x", records=2)
        a = gp.Parameter(m, "a", records=6)
        b = gp.Parameter(m, "b", records=3)
        e = gp.Equation(m, "e", definition=a * x / gp.math.sqr(b) == 0)
        mi = gp.Model(m, equations=[e], problem="LP", sense="FEASIBILITY")
        mi.freeze([a, b])
        mi.solve(solver="cplex", solver_options={"writelp": "mi.lp"})
        mi.unfreeze()

    The Cplex LP file ``mi.lp`` looks as follows:
    
    ::

        \ENCODING=ISO-8859-1
        \Problem name: gamsmodel

        Minimize
         _obj: constobj
        Subject To
         _e#0: 0.666666666666667 x + 0.222222222222222 a_var - 0.888888888888889 b_var
                = -1.33333333333333
        Bounds
              x Free
              a_var = 6
              b_var = 3
              constobj = 0
        End

    In the solver's presolve the fixed variables will be removed and there is minimal computational overhead with these newly introduced variables.
    
When all scenarios of a frozen model have been solved or the model needs to be changed beyond the modifiables, the :meth:`unfreeze <gamspy.Model.unfreeze>` method
needs to be called. This will releases the resources of the model instance and turns the model back to a regular model that is generated *and* solved when the
:meth:`solve <gamspy.Model.solve>` method is called.

The following example shows how to use a single multiplier ``bmult`` to adjust the demand (equation ``demand``) of the markets and solve the model with different value for this demand multiplier.

::

    from gamspy import (
        Container,
        Set,
        Parameter,
        Variable,
        Equation,
        Sum,
        Model,
        Sense,
        ModelStatus,
    )
    import numpy as np
    
    m = Container()
    
    i = Set(m, name="i")
    j = Set(m, name="j")
    
    a = Parameter(
        m,
        name="a",
        domain=i,
        domain_forwarding=True,
        records=[["seattle", 350], ["san-diego", 600]],
    )
    b = Parameter(
        m,
        name="b",
        domain=j,
        domain_forwarding=True,
        records=[["new-york", 325], ["chicago", 300], ["topeka", 275]],
    )
    d = Parameter(
        m, name="d", domain=[i, j], records=np.array([[2.5, 1.7, 1.8], [2.5, 1.8, 1.4]])
    )
    c = Parameter(m, name="c", domain=[i, j])
    c[i, j] = 90 * d[i, j] / 1000
    
    x = Variable(m, name="x", domain=[i, j], type="Positive")
    
    supply = Equation(m, name="supply", domain=i)
    demand = Equation(m, name="demand", domain=j)
    bmult = Parameter(m, name="bmult", records=1)
    
    cost = Sum((i, j), c[i, j] * x[i, j])
    supply[i] = Sum(j, x[i, j]) <= a[i]
    demand[j] = Sum(i, x[i, j]) >= bmult * b[j]
    
    transport = Model(
        m,
        name="transport",
        equations=[supply, demand],
        problem="LP",
        sense=Sense.MIN,
        objective=cost,
    )
    
    bmult_list = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    
    transport.freeze(modifiables=[bmult])
    
    for b_value in bmult_list:
        bmult.setRecords(b_value)
        transport.solve(solver="conopt")
        print(
            f'obj:{transport.objective_value if transport.status == ModelStatus.OptimalGlobal else "infeasible"}'
        )
    
    transport.unfreeze()

.. note::
    
    - Modifiable parameters cannot be used in ``.where`` conditions.
    - Variable and equation attributes used in equation
      algebra are evaluated once at model generation. Changes in the attributes will not percolate to the algebra.
      For example, the algebra ``x <= b * x.up`` will not change even if the modifiables include ``x.up``. One needs
      a parameter ``bigM`` and algebra ``x <= b * bigM`` in order to modify this algebra in a frozen solve.
    - There are limitations to modifiable parameters in quadratic constraints in quadratic models (QCP, MIQCP, RMIQCP). For details, see :ref:`quadratic_models`.
    - Even with few modifiable parameters, the frozen model may change significantly and the update of the frozen model
      may take longer than with a regular solve.
    - With a regular model the sparsity of the parameter often determines the sparsity of the constraint in a model. For example, ``Sum(j, a[j]*x[j])`` will
      only have as many non-zeros as parameter ``a`` has. If you turn parameter ``a`` into a modifiable, this constraint will become totally dense, because we might update
      an element of ``a`` to a non-zero number even though the value is currently 0. One way around this is
      to limit the indexing of the ``Sum`` to relevant elements of ``j``: ::

        jj = Set(m, "jj", domain=j)
        jj[j] = a[j]
        e = Equation(m, "e", definition=Sum(jj, a[jj]*x[jj] <= 1>))

    - Even though the solver can be switched in between solves, this is not recommended and can even lead to wrong results with 
      :meth:`FreezeOptions.update_type="accumulate" <gamspy.FreezeOptions.update_type>` and solvers supporting communication of frozen model changes only.
      

Extra Values in the Modifiable's Data
-------------------------------------

The records of the modifiable parameters or attributes of a variable or equation might contain records that are not represented in the model.
The :meth:`FreezeOptions.no_match_limit <gamspy.FreezeOptions.no_match_limit>` allows
to change the behavior in the presence of such *extra* records in the data.

In the following example, we define a standard LP model with coefficient matrix ``a``, cost vector ``c``, and modifiable right hand side vector
``b``. The variable ``x`` is indexed over `j`, while the equations `e` are indexed over ``i``. The actual model is defined over a
subset of ``i`` and ``j``, namely ``ii`` and ``jj``: ::

    import gamspy as gp

    m = gp.Container()
    i = gp.Set(m, name="i")
    j = gp.Set(m, name="j")
    a = gp.Parameter(m, name="a", domain=[i, j])
    b = gp.Parameter(m, name="b", domain=i)
    c = gp.Parameter(m, name="c", domain=j)

    x = gp.Variable(m, name="x", domain=j, type="positive")
    e = gp.Equation(m, name="e", domain=i)

    ii = gp.Set(m, domain=i, name="ii", description="active i")
    jj = gp.Set(m, domain=j, name="jj", description="active j")

    e[ii] = gp.Sum(jj, a[ii, jj] * x[jj]) >= b[ii]

    mymodel = gp.Model(
        m,
        name="mymodel",
        equations=[e],
        objective=gp.Sum(jj, c[jj] * x[jj]),
        sense="min",
        problem="lp",
    )

    i.setRecords(range(10))
    j.setRecords(range(20))

    ii.setRecords(range(5))
    jj.setRecords(range(10))

    a[ii, jj] = gp.math.uniform(0, 1)
    b[ii] = gp.math.uniform(1, 10)
    c[jj] = gp.math.uniform(1, 10)

    mymodel.freeze(modifiables=[b])
    b[i] = gp.math.uniform(1,10)  # assignment over the full set i, not over subset ii
    mymodel.solve()

With this setup we can now pass more records to the modifiable parameter ``b``
than present in the frozen model, namely the ``b`` records 5, 6, 7, 8, and 9. If we have update values with no corresponding record
in the model, this is suspicious and hence, the ``solve`` throws an exception by default: ::

    gamspy.exceptions.GamspyException: Unmatched record limit exceeded while processing modifier b, for more info check no_match_limit option.

The number of allowed *unmatched records* can be controlled with ``no_match_limit``. Here any number of 5 and larger will prevent 
the exception and the frozen model is passed to the solver: ::

    mymodel.solve(freeze_options=gp.FreezeOptions(no_match_limit=5))

Missing Values in the Modifiable's Data
---------------------------------------

As extra values in the data cause concern, explicitly missing data for the parameters or variables present in the model also need some attention.
GAMSPy and in particular the GAMS execution engine does not store default records. For parameters
this means that zeros (0) are not explicitly stored. But also variable records with default attributes, e.g. positive variables with ``level``, ``marginal``,
and ``lower`` bound 0.0, ``upper`` at ``+infinity``, and ``scale`` at 1.0, are not explicitly stored. So GAMSPy does not distinguish between a `missing` or `default value`. 
If the frozen model contains a particular record of a modifiable symbol,
but the corresponding symbol data in the container does not, GAMSPy treats this missing record to be at default value. There are several choices for how this default value should be 
applied when updating the frozen model.
The different values for :meth:`FreezeOptions.update_type <gamspy.FreezeOptions.update_type>` control
the behavior of default records in the modifiable's data.  

Basically, there are two options:
Setting ``update_type="0"`` takes the values of the modifier's data (default or not) and applies them to the records in the model. For parameters,
this means, all records in the model are set to 0 (zero) and then the non-default values in the modifiable's data are applied. For variables, this means the
default bounds are restored and the records for the bounds stored in the variable symbols's data are applied afterwards. Using ``update_type="base_case"``
or ``update_type="accumulate"`` will just ignore default values and only apply the explicit records found in the modifiable's data. The option
``update_type="base_case"`` will apply the records against the model records present when the :meth:`freeze <gamspy.Model.freeze>` was called;
``update_type="accumulate"`` will apply the records against the model records from the last solve. The difference will become clearer with the following example
that builds on the example for ``no_match_limit``. In order to make the updated ``b`` clearer we start the freeze with ``b=[1,2,3,4,5]``: ::


    b[ii] = gp.Ord(ii)
    mymodel.freeze(modifiables=[b])
    mymodel.solve()

Now, we clear ``b`` and set the last record to 100: ::

    b[ii] = 0
    b['4'] = 100

If we solve now with ``update_type="0"``, the solver sees the right hand side (which is our ``b`` parameter) of ``[0,0,0,0,100]``.
When ``update_type="base_case"`` or ``update_type="accumulate"`` is used, the solver sees the right hand side of ``[1,2,3,4,100]``. Let's continue the example
with ``update_type="0"``: ::

   mymodel.solve(freeze_options=gp.FreezeOptions(update_type="0"))

The records for ``b`` inside the model instance are now ``[0,0,0,0,100]``. We could have also accomplished this by not having any default records
and any update_type: ::

    b[ii] = gp.SpecialValues.EPS
    b['4'] = 100

Now assume, we again clear the updater symbol's data and set the first record to 200 ::

    b[ii] = 0
    b['0'] = 200

With ``update_type="0"``, the solver sees ``b=[200,0,0,0,0]``; with ``update_type="base_case"``, the solver sees ``b=[200,2,3,4,5]`` and with
``update_type="accumulate"``, the solver sees ``b=[200,0,0,0,100]`` (assuming we solved the first scenario with ``update_type="0"``).
The default for this option is ``update_type="base_case"``.

The following example builds upon the model with the demand multiplier earlier in this chapter and shows how to update the upper bound of the
transportation variable ``x`` to prevent any transportation on a particular connection: ::

    transport.freeze(modifiables=[x.up])

    for irec in i.toList():
        for jrec in j.toList():
            x.up[irec, jrec] = SpecialValues.EPS
            transport.solve()
            x.up[irec, jrec] = SpecialValues.POSINF # restore original bound
            print(
                f'obj:{transport.objective_value if transport.status == ModelStatus.OptimalGlobal else "infeasible"}'
            )

    transport.unfreeze()

.. _quadratic_models:

Modifiables in Quadratic Models
-------------------------------

Quadratic models have certain limitations when turned into frozen models.
These limitations only affect solvers that *extract* the quadratic matrices from the non-linear expressions like Cplex, COPT, Gurobi, Mosek, and Xpress.
Other solvers that work directly with the non-linear expressions and automatic differentiation like Conopt and Knitro do not have any limitations.

Variable and equation attributes, like variable bounds do not represent any limitation even for quadratic models. Only parameter modifiables can cause
some problems. Since modifiable parameters are turned into fixed variables, any expression that is not linear or quadratic, like our example ``a/sqr(b)*x``,
are rejected with the message *Detected n general nonlinear rows in model*. Acceptance of terms that would be linear or constant in a regular model,
but quadratic in a frozen model,
like ``a*x`` or even ``a*b`` with modifiable parameters ``a`` and ``b`` and variable ``x`` depends heavily on the solver. Additive use of modifiable parameters, e.g. to
change the right hand side of a constraint works without problems.

Due to limited update possibilities in the API of some solvers, quadratic models do not benefit from the minimal communication of just the model instance *changes*.
The entire instance needs to be freshly set up inside the solver with every solve. Nevertheless, using a frozen model avoids regenerating the model instance from scratch for every solve, leading to improved efficiency.