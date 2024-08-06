.. _variable:

.. meta::
   :description: Documentation of GAMSPy Variable (gamspy.Variable)
   :keywords: Variable, GAMSPy, gamspy, GAMS, gams, mathematical modeling, sparsity, performance

********
Variable
********

Introduction
=============

This chapter covers the declaration and manipulation of GAMSPy 
:meth:`variables <gamspy.Variable>`. Many concepts from previous chapters apply 
here as well.

A variable is the GAMSPy name for what are called *endogenous variables* by 
economists, *columns* or *activities* by linear programming experts, and 
*decision variables* by industrial Operations Research practitioners. They are 
the entities whose values are generally unknown until after a model has been 
solved. A crucial difference between GAMSPy variables and columns in traditional 
mathematical programming terminology is that one GAMSPy variable is likely to be 
associated with many columns in the traditional formulation.

Variable Declarations
======================

A GAMSPy variable, like all other identifiers, must be declared before it may be 
referenced.

The Syntax
-----------

The declaration of a variable is similar to a set or parameter declaration, in 
that domain lists and descriptions are allowed and recommended ::
    
    from gamspy import Container, Set, Variable

    m = Container()

    t = Set(m, name = "t", records = range(1990,2000), description = "time periods")
    
    k = Variable(m, name = "k", domain = t, description = "capital stock (trillion rupees)")
    c = Variable(m, name = "c", domain = t, description = "consumption (trillion rupees per year)")
    i = Variable(m, name = "i", domain = t, description = "investment (trillion rupees per year)")
    utility = Variable(m, "utility", description = "utility measure")

The :meth:`Variable <gamspy.Variable>` class indicates that this is a variable 
statement. ``name`` is the optional internal name of the variable in GAMSPy, it is an 
*identifier*. In the optional ``domain`` list the set or sets may be specified 
over which an indexed variable is declared. The optional ``description`` may be 
used to describe the variable for future reference and to ease readability. 

.. 
    Specifying variable data (``records``) is another optional element in the variable 
    statement. ``Records`` allow to initialize variable attributes at compile time. 
    For an example and details on variable attributes, see section 
    :ref:`variable-attributes`.

The declaration of ``k`` above implies, as usual, that references to ``k`` are restricted to 
the domain of the set ``t``. A model that includes ``k`` will probably have several 
corresponding variables in the associated mathematical programming problem: most likely one 
for each member of ``t``. In this way, very large models can be constructed using a small 
number of variables. (It is quite unusual for a model to have as many as 50 distinct 
variables.) It is still unclear from the declaration whether ``utility`` is not domain checked 
or whether it is a scalar variable, i.e., one without associated sets. Later references will be 
used to settle the issue. For more details on domain checking, see section 
`Domain Checking <https://www.gams.com/latest/docs/UG_SetDefinition.html#UG_SetDefinition_DomainChecking>`_ 
in the GAMS documentation.

.. note::
    - Variables can be defined over 0 to 20 sets.
    - The sets over which variables are declared indicate that these variables are potentially 
      defined for every element of the defining sets. However the actual definition of variables 
      does not occur until variables appear in an :ref:`equation definition <equation>` where the 
      equation needs to be part of a :ref:`model` that in turn occurs in a solve statement.


.. _variable-types:

Variable Types
---------------

There are nine basic :meth:`variable types <gamspy.VariableType>` that may be used in 
variable statements: 

=================================================  ==================================================================================================================================================================================================  ======================  ======================
Keyword                                            Description                                                                                                       Default Lower Bound   Default Upper Bound
=================================================  ==================================================================================================================================================================================================  ======================  ======================
free (default)                                     No bounds on variable. Both bounds may be changed from the default values by the user.                                                                                                              float('-inf')           float('inf')
positive or nonnegative                            No negative values are allowed for variable. The user may change both bounds from the default value.                                                                                                0                       float('inf')
negative                                           No positive values are allowed for variables. The user may change both bounds from the default value.                                                                                               float('-inf')           0
binary                                             Discrete variable that can only take values of 0 or 1. For details see section :ref:`types-of-discrete-variables`. In relaxed Model types the integrality requirement is relaxed.                   0                       1
integer                                            Discrete variable that can only take integer values between the bounds. In relaxed Model types the integrality requirement is relaxed.                                                              0                       float('inf')
sos1                                               A set of variables, such that at most one variable within a group may have a non-zero value. For details see section :ref:`types-of-discrete-variables`.                                            0                       float('inf')
sos2                                               A set of variables, such that at most two variables within a group may have non-zero values and the two non-zero values are adjacent. For details see section :ref:`types-of-discrete-variables`.   0                       float('inf')
semicont                                           Semi-continuous, must be zero or above a given minimum level. For details see section :ref:`types-of-discrete-variables`.                                                                           1                       float('inf')
semiint                                            Semi-integer, must be zero or above a given minimum level and integer. For details see section :ref:`types-of-discrete-variables`. In relaxed Model types the integrality requirement is relaxed.   1                       float('inf')
=================================================  ==================================================================================================================================================================================================  ======================  ======================

The default type is ``free``, which means that if the type of the variable is not 
specified, it will not be bounded at all. The most frequently used types are ``free`` 
and ``positive``. The type ``positive variables`` is used for variables for which 
negative values are meaningless, such as capacities, quantities or prices. Note that 
bounds may be changed using variable attributes and assignment statements, see section 
:ref:`variable-attributes`.

.. note::
    - Every optimization model must contain at least one unrestricted named variable. 
      This variable is the objective variable. Even an objective variable can have 
      lower and upper bounds assigned via the ``lo`` and ``up`` 
      :meth:`variable <gamspy.Variable>` attribute.
    - If a model is unbounded, a frequent cause for the unboundedness is that the 
      modeler forgot to make a variable positive.


.. _variable-attributes:

Variable Attributes
=====================

Introduction
-------------

While a GAMSPy :ref:`parameter <parameter>` has one number associated with each unique 
label combination, a variable has several attributes. They represent:

========================== ======  ==================================================================================================================================================================================================================================================================================================================================================
Variable Attribute         Symbol  Description
========================== ======  ==================================================================================================================================================================================================================================================================================================================================================
Lower bound                lo      Lower bound for the variable. Set by the user either explicitly or through default values associated with the variable type.
Upper bound                up      Upper bound for the variable. Set by the user either explicitly or through default values associated with the variable type.
Fixed value                fx      A fixed value for the variable. If set, it results in the upper and lower bounds of the variable being set to the value of the fx attribute.
Activity level             l       Activity level for the variable, also the current value or starting point. This attribute is reset to a new value when a model containing the variable is solved. The activity level is used to construct a `basis [GAMS documentation] <https://www.gams.com/latest/docs/UG_SolverUsage.html#ADVANCED_USAGE_Basis>`_  for the model.
Marginal                   m       The marginal value (or reduced cost) for the variable. This attribute is reset to a new value when a model containing the variable is solved. The activity level is used to construct a `basis [GAMS documentation] <https://www.gams.com/latest/docs/UG_SolverUsage.html#ADVANCED_USAGE_Basis>`_  for the model.
Scale factor               scale   Numerical scaling factor for all coefficients associated with the variable. Only applicable for continuous variables.
Branching priority         prior   Branching priority value used in mixed integer programming models. Only applicable for discrete variables.
Stage                      stage   This attribute allows you to assign variables to stages in a stochastic program or other block-structured model. Thus, among other places, it is used for 2-stage stochastic programs solved with DECIS or the Benders partition in `Cplex [GAMS documentation] <https://www.gams.com/latest/docs/UG_SolverUsage.html#ADVANCED_USAGE_Basis>`_.
========================== ======  ==================================================================================================================================================================================================================================================================================================================================================

If the data is in a convenient format, it is possible to specify initial values for these 
variable attributes within the variable constructor. This is an optional keyword argument 
and internally the variable constructor will simply call the ``setRecords`` method. In contrast 
to the ``setRecords`` methods in in either the Set or Parameter classes the ``setRecords`` method 
for variables will only accept Pandas DataFrames and specially structured ``dict`` for creating 
records from matrices. This restriction is out of necessity because to properly set a record 
for a Variable the user must pass data for the ``level``, ``marginal``, ``lower``, ``upper`` and 
``scale`` attributes. That said, any missing attributes will be filled in with the default 
record values (see :ref:`variable-types`). 

Example #1: Create a GAMS scalar variable ::

    from gamspy import Container, Variable
    import pandas as pd

    m = Container()

    pi = Variable(m, "pi", records=pd.DataFrame(data=[3.14159], columns=["level"]))
     
    # NOTE: the above syntax is equivalent to -
    # pi = Variable(m, "pi", "free")
    # pi.setRecords(pd.DataFrame(data=[3.14159], columns=["level"]))
     
    # NOTE: the above syntax is also equivalent to -
    # m.addVariable("pi", "free", records=pd.DataFrame(data=[3.14159], columns=["level"]))

::

    In [1]: pi.records
    Out[1]:
         level  marginal  lower  upper  scale
    0  3.14159       0.0   -inf    inf    1.0

Example #2 - Create a 1D variable (defined over `'*'`) from a list of tuples ::

    from gamspy import Container, Variable
    import pandas as pd

    m = Container()

    v = Variable(
        m, "v", "free", domain=["*"],
        records=pd.DataFrame(
            data=[("i" + str(i), i) for i in range(5)], columns=["domain", "marginal"]
        ),
    )

::

    In [1]: v.records
    Out[1]:
        uni    level  marginal  lower  upper  scale
    0    i0      0.0       0.0   -inf    inf    1.0
    1    i1      0.0       1.0   -inf    inf    1.0
    2    i2      0.0       2.0   -inf    inf    1.0
    3    i3      0.0       3.0   -inf    inf    1.0
    4    i4      0.0       4.0   -inf    inf    1.0

Example #3 - Create a 1D variable (defined over a set) from a list of tuples ::

    from gamspy import Container, Set, Variable
    import pandas as pd

    m = Container()

    i = Set(m, "i", ["*"], records=["i" + str(i) for i in range(5)])
    v = Variable(
        m,
        "v",
        "free",
        domain=i,
        records=pd.DataFrame(
            data=[("i" + str(i), i) for i in range(5)], columns=["domain", "marginal"]
        ),
    )

::
    
    In [1]: v.records
    Out[1]:
        i    level  marginal  lower  upper  scale
    0  i0      0.0       0.0   -inf    inf    1.0
    1  i1      0.0       1.0   -inf    inf    1.0
    2  i2      0.0       2.0   -inf    inf    1.0
    3  i3      0.0       3.0   -inf    inf    1.0
    4  i4      0.0       4.0   -inf    inf    1.0

For more examples see the `GAMS Transfer documentation <https://www.gams.com/latest/docs/API_PY_GAMSTRANSFER_MAIN_CLASSES.html#PY_GAMSTRANSFER_ADD_VARIABLE_RECORDS>`_

.. note::
    - ``fx`` and attributes ``lo`` and ``up`` on the same variable cannot be in a data 
      statement. ``fx`` sets both ``lo`` and ``up`` and hence we would have a double 
      definition of the same attribute. Since attribute ``scale`` is applicable for 
      continuous variables and attribute ``prior`` for discrete variables, they share 
      the same internal space in a GAMSPy variable. Some solvers can make use of 
      priorities even for continuous variables (e.g. 
      `BARON <https://www.gams.com/latest/docs/S_BARON.html#BARON_THE_BARON_OPTIONS>`_). 
      Such priorities need to be supplied via ``solver_options`` in the 
      :meth:`Model <gamspymodel>` class.
    - The attribute ``stage`` uses the same internal space as ``scale`` and ``prior``. 
      So a model cannot specify scale factor and branching priorities together with 
      stages.
    - Fixing a semi-continuous or semi-integer variable to a non-zero value like ``4`` 
      does not result in a truly fixed variable. The domain of the variable remains 
      ``{0,4}``. To really fix a semi-continuous or semi-integer variable, the discrete 
      restriction could be relaxed by setting the branching priority (``prior``)to 
      infinity.
    - For variables in discrete models (such as MIP, MINLP), the ``m`` attribute 
      provides the marginals obtained by fixing all the discrete variables and solving 
      the resulting continuous problem (such as LP, NLP). Many solvers allow to 
      enable/disable solving such a fixed problem. When disabled, no marginals will 
      be provided for discrete models.

In addition to the variable attributes introduced above, there are a number of variable 
attributes that cannot be assigned but may be used in computations.

===========================  =========  =========================================================================================================================================================================================================================================================================================================================================
Variable Attribute           Symbol     Description
===========================  =========  =========================================================================================================================================================================================================================================================================================================================================
Range                        range      The difference between the lower and upper bounds for a variable. It becomes zero if the lower equals the upper bound, e.g. if the ``fx`` attribute is set.
Slack upper bound            slackup    Slack from variable upper bound. This is defined as the greater of two values: zero or the difference between the upper bound and the level value of a variable.
Slack lower bound            slacklo    Slack from variable lower bound. This is defined as the greater of two values: zero or the difference between the level value and the lower bound of a variable.
Slack                        slack      Minimum slack from variable bound. This is defined as the minimum of two values: the slack from the variable lower bound and the slack from the variable upper bound.
Infeasibility                infeas     Amount by which a variable is infeasible falling below its lower bound or above its upper bound. This is defined as the smallest of three values: zero, the difference between the lower bound and the level value, the difference between the level value and the upper bound of a variable, i.e. ``max[0, lower-level, level-upper]``.
===========================  =========  =========================================================================================================================================================================================================================================================================================================================================

Bounds on Variables
--------------------

All default bounds set at declaration time may be changed using assignment statements.

.. warning::
    For discrete variable types, the consequences of the type declaration cannot be 
    completely undone (e.g. the ``scale`` attribute is not available) but their value 
    domain can be changed to continuous by setting attribute ``prior`` to infinity.

Bounds on variables are the responsibility of the user. After variables have been declared, 
default bounds have already been assigned: for many purposes, especially in linear models, 
the default bounds are sufficient. In nonlinear models, however, bounds play a far more 
important role. It may be necessary to provide bounds to prevent undefined operations, 
such as division by zero. In nonlinear programming it is often necessary to define a 
'reasonable' solution space that will assist in efficiently finding a solution.

.. warning::
    The lower bound cannot be greater than the upper bound: if you happen to impose such 
    a condition, GAMS will generate an execution error when executing a solve statement.


Fixing Variables
-----------------

GAMS allows the user to fix variables through the ``fx`` variable attribute. This is almost 
equivalent to setting the lower bound and upper bound equal to the fixed value. The attribute 
``fx`` also resets the activity level ``l`` to the fixed value. When setting ``lo`` and ``up`` 
the activity level remains unchanged. A solve statement will project the activity level within 
the active bounds. Fixed variables can subsequently be freed by changing the lower and upper 
bounds.

Activity Levels of Variables
-----------------------------

GAMS allows the user to set the activity levels of variables through the ``l`` variable 
attribute. These activity levels of the variables prior to the solve statement serve as 
initial value for the solver. This is particularly important for nonlinear programming 
problems. For discrete models in many cases the solver needs an additional indicator to 
interpret the activity levels as a feasible integer solution via a solver option 
(e.g. Cplex' `mipstart [GAMS documentation] <https://www.gams.com/latest/docs/S_CPLEX.html#CPLEXmipstart>`_).

.. note::
    - GAMS only stores variables with non-default values (similar to storing only non-zero 
      values of parameters). Non-default variables can be accidentally created by using 
      harmlessly looking assignments like ``x.up[i,j,k,l] = 0``.
      Even if the equations only reference such variables over a small subset of [i,j,k,l] 
      this statement creates card[i]*card[j]*card[k]*card[l] variable records in the GAMSPy 
      database. Such fixings of ``x[i,j,k,l]`` to 0 can be avoided by using .
      :ref:`dynamic sets in the equation algebra <conditional-equations-with-dynamic-sets>` 
      to only reference tuples of [i,j,k,l] for which x[i,j,k,l] can possible have a non-zero value.
    - In order to filter only necessary tuples for an equation the filtering conditions needs 
      to be provided only once when defining the equation (``equ[i,j,k]``). This is different for 
      variables because they appear in many equations and the filtering condition needs to be 
      potentially repeated many times. Therefore it is good practice and reduces GAMS model 
      generation time if the filtering of the variables is governed by a dynamic set: ::

          Sum(i,j).where[Ord(i)>Ord(j) & cap[i,j]>0], x[i,j])

      versus ::

          net = Set(m, name = "net", domain = [i,j])
          net[i,j] = Ord(i)>Ord(j) & cap[i,j]>0
          Sum(net[i,j], x[i,j])


Variables in Assignment Statements
===================================

Assigning Values to Variable Attributes
-----------------------------------------

Assignment statements operate on one variable attribute at a time, and require the suffix to 
specify which attribute is being used. Any index list comes after the suffix. ::

    x.up[c,i,j] = 1000
    phi.lo[...] = inf

A very common use is to bound one particular entry individually: ::

    p.up['pellets', 'ahmsa', 'mexico-df']  = 200

Or to put small lower bounds on a variable identifier used as a divisor in a nonlinear program: ::

    c.lo[t] = 0.01

Or to provide initial values for a nonlinear problem: ::

    c.l[t]   =  4*cinit[t]

It is also possible to do an assignment without any index to scalar variables: ::

    import gamspy as gp
    m = gp.Container()
    v = gp.Variable(m, "i")
    v.l = 5

Remember that the order is important in assignments, and notice that the two pairs of 
statements below produce very different results. In the first case, the lower bound for 
``c['1985']`` will be 0.01, but in the second, the lower bound is 1. ::

    # 1
    c.fx['1985'] = 1     
    c.lo[t]      = 0.01
    
    # 2
    c.lo[t]      = 0.01          
    c.fx['1985'] = 1

Everything works as described in the previous chapters, including the various mechanisms 
described there of indexed operations, subset assignments and so on. ::

    ship_sm.lo[sl,m].where[Ord(sl) = 1 & Ord(m) = 1] = 1

The lower bound of the variable ``ship_sm[sl,m]`` is set to 1 and this assignment is only 
valid for ``ship_sm['s1','d1']``, the realization of the variable where both indices are 
the first members of their respective sets.

Variable Attributes in Assignments
----------------------------------

The following examples illustrate the use of variable attributes on the right-hand side of 
assignment statements: ::

    y.l[i] = 250  
    x.l[i] = 200 
    e.l[t] =   0  
    m.l[t] =   0 
    
    g.l[t] = mew[t] + xsi[t]*m.l[t] 
    h.l[t] = gam[t] - alp[t]*e.l[t] 
    
    [...]
    
    # generating report after solve 
    cva = Sum(i, v.l[i]*x.l[i])  
    cli = Sum(i, p.l[i]*ynot[i])/Sum(i, ynot[i])
    rva = cva/cli

As with parameters, a variable must have some non-default data values associated with it 
before it can be used on the right-hand side of an assignment statement. After a solve 
statement has been processed or if non-default values have been set with an assignment 
statement, this condition is satisfied. 

.. warning::
    The ``fx`` attribute fixes the variable at a certain value, effectively setting both 
    the lower and upper bounds to this value. Therefore, it is mostly just a shorthand 
    for ``lo`` and ``up`` and can only be used on the left-hand side of an assignment 
    statement.


.. _types-of-discrete-variables:

Types of Discrete Variables
===========================

GAMS provides six discrete variable types: ``binary``, ``integer``, ``sos1``, ``sos2``, 
``semicont`` and ``semiint``. In the following subsections we will present details and 
examples for each of these discrete variable types. Note that if any discrete variables 
feature in a model, it has to be a mixed integer model or one of the related model types, 
like ``MINLP`` or ``MIQCP``. See section 
`Classification of Models [GAMS documentation] <https://www.gams.com/latest/docs/UG_ModelSolve.html#UG_ModelSolve_ModelClassificationOfModels>`_ 
for a full listing of all GAMS model types.

.. _binary-variables:

Binary Variables
-----------------

Binary variables can take values of 0 (zero) and 1 (one) only. ::

    from gamspy import Container, Set, Alias, Variable, Equation, Sum, Domain

    m = Container()
    k = Set(m, "k", description = "rows",    records = ["row1","row2","row3","row4"])
    l = Set(m, "l", description = "columns", records = ["col1","col2","col3","col4"])
    v = Set(m, "v", description = "values",  records = ["val1","val2","val3","val4"])
    
    i = Alias(m, name = "i", alias_with = v)
    j = Alias(m, name = "j", alias_with = v)
    
    x = Variable(m, "x", description = "pairs (i,j) allocated to cell(k,l)",
                 domain = [i,j,k,l], type = "binary")
    
    z = Variable(m, "z", description = "some objective")
    c1 = Equation(m, "c1", domain = [i,j], 
                  description = "for each cell pick only one item pair")
    
    c1[i,j] = Sum(Domain(k,l), x[i,j,k,l]) == 1

Note that the binary variable ``x`` is used in equation ``c1`` to model the restriction 
that in each cell only one item pair is allowed. Binary variables are often used to model 
logical conditions such as imposing mutual exclusivity or complementarity.

Note that the default lower bound is 0 (zero) and the default upper bound is 1 (one). If 
the relaxed versions of the discrete models is solved, binary variables are treated like 
positive variables with the upper bound of 1. 

Even though the only possible values are 0 and 1, a solver might return a value for binary 
variable that is only close to 0 or 1. Every solver works with tolerances and also uses a 
tolerance to determine if a value is close enough to an integer values. So it is unwise to 
use code as ``a[i].where[b.l[i]=1] = True`` because one will potentially miss some elements. 
A safe way to write such code is: ``a[i].where[b.l[i]>0.5] = True``. Rounding the level of a 
binary variable after the solve is also possible, but it is not done by the solver or the 
solver link because even small rounding can lead to infeasibilities.

A binary variable can also have a truely fractional value after a solver if the model status 
does not indicate a feasible integer solution (model status ``1`` or ``8``).


Integer Variables
------------------

Integer variables are discrete variables that can take only values between their bounds. 
The user may change both bounds from the default value. The default lower bound is 0 (zero) 
and the default upper bound inside GAMS is ``float('inf')``, and the same upper bound is passed on 
to the solver.

Note that in relaxed model types the integrality requirement is relaxed. ::

    from gamspy import Container, Set, Variable

    m = Container()

    t = Set(m, "t", 
            records = ["12pm-6am","6am-9am","9am-3pm","3pm-6pm","6pm-12pm"], 
            description = "demand blocks")

    g = Set(m, "g", records = [¨"type-1", "type-2", "type-3"], 
            description = "generators")

    x = Variable(m, "x", domain = [g,t], 
                 description = "number of generators in use")

    cost = Variable(m, "cost", 
                 description = "total operating cost (l)")

    n = Variable(m, "n", domain = [g,t], type = "integer",
                 description = "generator output (1000mw)")

The integer variable ``n`` models the number of generators of various types that are in 
use at any of the time blocks.

Special Order Sets of Type 1 (SOS1)
------------------------------------

SOS1 variables are a set of variables, such that at most one variable within the group 
may have a nonzero value. This variable may take any positive value. ::

    s1 = Variable(m, "s1", type = "sos1", domain = i)
    t1 = Variable(m, "t1", type = "sos1", domain = [k,j])
    w1 = Variable(m, "w1", type = "sos1", domain = [i,j,k])

Note that the members of the innermost (the right-most) index belong to the same SOS set. 
For example in the sets defined above, ``s1`` represents one special ordered set of type 
1 with ``i`` elements, ``t1`` defines ``k`` sets with ``j`` elements each and ``w1`` 
defines ``[i,j]`` sets with ``k`` elements each.

The default bounds for ``SOS1`` variables are ``zero`` and ``float('inf')``. As with any other 
variable, the user may change these bounds. Further, the user may explicitly provide 
whatever convexity row that the problem may need through an equation that requires 
the members of the ``SOS1`` set to be less than a certain value. Any such convexity 
row will implicitly define bounds on each of the variables.

Consider the following example: ::

    s1 = Variable(m, "s1", type = "sos1", domain = i)
    
    defsoss1 = Equation(m, "defsoss1")
    defsoss1 = Sum(i,s1[i]) <= 3.5

The equation ``defsoss1`` implicitly defines the nonzero value that one of the elements 
of the ``SOS1`` variable ``s1`` may take as equal to or smaller than ``3.5``. Note that 
it is also possible that all variables ``s1`` equal zero.

A special case arises when one of the elements of the set has to be nonzero and equal to 
a number, say 3.5. In this case equation ``defsoss1`` will be: ::

    defsoss1 = Sum(i,s1[i]) == 3.5

Frequently the nonzero value equals 1. As a result, the ``SOS1`` variable is effectively 
a binary variable. It is only treated differently by the solver at the level of the 
branch and bound algorithm. For example, consider the following example where we want 
to model that one out of n options has to be selected. This is expressed as: ::

    x = Variable(m, "x", type = "sos1", domain = i)
    
    defx = Equation(m, "defx")
    defx = Sum(i, x[i]) == 1

The variable ``x`` can be made binary without any change in meaning and the solution 
provided by the solver will be indistinguishable from the ``SOS1`` case.

The use of special ordered sets may not always improve the performance of the branch 
and bound algorithm. If there is no natural order the use of binary variables may be 
a better choice. A good example of this is the classical assignment problem 
(see [H.P. Williams (2013) `Model Building in Mathematical Programming <https://books.google.de/books?id=YJRh0tOes7UC&lpg=PP1&dq=Model%20Building%20in%20Mathematical%20Programming&pg=PP1#v=onepage&q=Model%20Building%20in%20Mathematical%20Programming&f=false>`_], 
Wiley, Section 9.3.

Note that any model with ``SOS1`` variables requires a MIP solver, because the 
solution process needs to impose the restrictions of at most one nonzero level values 
may be present.

Special Order Sets of Type 2 (SOS2)
-------------------------------------

``SOS2`` variables are a set of variables, such that at most two variables within the 
set may have nonzero values and these variables have to be adjacent. This requirement 
implies that the set is ordered, see chapter :ref:`ordered-sets` for details on ordered 
sets in GAMSPy. Note that the nonzero variables may take any positive value. ::

    i = Set(m, "i", records = [¨"i1", "i2", "i3", "i4", "i5"])
    
    s2 = Variable(m, "s2", type = "sos2", domain = i)
    t2 = Variable(m, "t2", type = "sos2", domain = [k,j])
    w2 = Variable(m, "w2", type = "sos2", domain = [i,j,k])

The members of the innermost (the right-most) index belong to the same set. For example, 
in the sets defined above, ``s2`` represents one special ordered set of type 2 with 
elements for each member of the set ``i``. At most two variables ``s2`` may be nonzero 
and they must reference adjacent elements of the set ``i``. Note that the variables 
``s2['i1']`` and ``s2['i2']`` are adjacent, but the variables ``s2['i1']`` and ``s2['i3']`` 
are not. Further, ``t2`` defines ``k`` sets of ``SOS2`` variables with ``j`` elements 
each and the adjacency requirement refers to the set ``j`` which must be ordered. 
Similarly, ``w2`` defines ``[i,j]`` sets with ``k`` elements each and the adjacency 
requirement refers to the set ``k`` which must be ordered.

The default bounds for ``SOS2`` variables are ``zero`` and ``float('inf')``. As with any other 
variable, the user may change these bounds. ``SOS2`` variables are most often used to 
model piece-wise linear approximations to nonlinear functions. 

Note that any model with ``SOS2`` variables requires a MIP solver, because the 
solution process needs to impose the restrictions of adjacency and that no more than 
two nonzero level values may be present.

Semi-Continuous Variables
--------------------------

Semi-continuous variables are either zero or above a given minimum level. This can be 
expressed algebraically as: either :math:`x = 0` or :math:`L <= x <= U` By default, the lower 
bound :math:`L` is 1 and the upper bound :math:`U` is ``float('inf')``. As usual, these 
bounds may be changed with the variable attributes ``lo`` and ``up``. ::

    x = Variable(m, "x", type = "semicont")
    x.lo[...] = 1.5
    x.up[...] = 23.1

The slice of code above declares the variable ``x`` to be a semi-continuous variable that may 
either be zero or behave as a continuous variable between 1.5 and 23.1.

Note that any model with semi-continuous variables requires a MIP solver, because the solution 
process needs to impose the discontinuous jump between zero and the threshold value.

.. note::

    - Not all MIP solvers allow semi-continuous variables. We recommend users to verify how the 
      solver they are interested in handles semi-continuous variables by checking the relevant 
      section of the respective solver manual.
    - The lower bound has to be less than the upper bound, and both bounds have to be greater 
      than zero, otherwise GAMSPy will report an error.
    - Semi-continuous variables are especially helpful if the upper bound is ``float('inf')`` 
      and no implicit bound can be easily derived. If a finite upper bound is available it can 
      be computational more efficient to replace the semi-continuous variable ``sc`` with lower 
      bound ``scLow`` by a continuous variable ``x`` and binary variable ``b`` and the 
      following equations: ::

        forceLBnd = Equation(m, "forceLBnd", 
                             description = "Force x to be greater than scLow if b is 1")
        forceZero = Equation(m, "forceZero",     
                             description = "Force x to be zero if b is zero")

        forceLBnd = x >= scLow*b
        forceZero = x <= x.up*b

Semi-Integer Variables
-----------------------

Semi-integer variables are either zero or integer and above a given minimum value. This can be 
expressed algebraically as: either :math:`x = 0` or :math:`x \in {L,...,U}`. By default, the 
lower bound :math:`L` is 1 and the upper bound :math:`U` inside GAMS is ``float('inf')`` and 
the same values are passed on to the solver. As usual, these default bounds may be changed with 
the variable attributes ``lo`` and ``up``. Note that in relaxed model types the integrality 
requirement is relaxed. ::

    x = Variable(m, "x", type = "semiint")
    x.lo[...] = 2
    x.up[...] = 25

The slice of code above declares the variable ``x`` to be a semi-integer variable that may 
either be zero or take any integer value between 2 and 25. Note that the bounds for ``semiint`` 
variables have to take integer values, otherwise GAMSPy will flag an error during model 
generation. Note further, that any model with semi-integer variables requires a MIP solver.

.. note::
    - Not all MIP solvers allow semi-integer variables. We recommend users to verify how the 
      solver they are interested in handles semi-integer variables by checking the relevant 
      section of the respective solver manual.
    - The lower bound has to be less than the upper bound, and both bounds have to be greater 
      than zero, otherwise GAMSPy will report an error.
    - Semi-integer variables are especially helpful if the upper bound is ``float('inf')`` 
      and no implicit bound can be easily derived (together with the appropriate 
      `IntVarUp [GAMS documentation] <https://www.gams.com/latest/docs/UG_GamsCall.html#GAMSAOintvarup>`_  
      setting). If a finite upper bound is available, it can be computationally more efficient 
      to replace the semi-integer variable ``si``, with lower bound ``siLow``, by an integer 
      variable ``i`` and a binary variable ``b`` and the following equations: ::

        forceLBnd = Equation(m, "forceLBnd", 
                             description = "Force i to be greater than siLow if b is 1")
        forceZero = Equation(m, "forceZero",     
                             description = "Force i to be zero if b is zero")

        forceLBnd = i >= scLow*b
        forceZero = i <= i.up*b

.. _inspecting_generated_variables:
Inspecting Generated Variables
------------------------------

The generated variables can be inspected by using :meth:`getVariableListing() <gamspy.Variable.getVariableListing>`
function after solving the model. The variable listing can be filtered with ``filters`` argument, and the number of 
variables returned can be limited with ``n`` argument.

For example, in `Mexico Steel sector model <https://github.com/GAMS-dev/gamspy/blob/develop/tests/integration/models/mexss.py>`_ 
exports variable ``e`` is defined over commodities ``c`` which contain 1 element and steel plants ``i`` which contain 
5 elements. If one prints the variable listing directly, ``getVariableListing`` would return all five generated variables. ::

  import gamspy as gp
  m = gp.Container()
  ...
  ...
  model_definition_goes_here
  ...
  ...
  model.solve(options=Options(variable_listing_limit=100))
  print(e.getVariableListing())

Generated variables: ::

    e(steel,ahmsa)
                    (.LO, .L, .UP, .M = 0, 0, +INF, 0)
           -1       mbf(steel,ahmsa)
            1       me(steel)
           -8.6876  alam
         -140       aeps,
    e(steel,fundidora)
                    (.LO, .L, .UP, .M = 0, 0, +INF, 0)
           -1       mbf(steel,fundidora)
            1       me(steel)
           -6.8564  alam
         -140       aeps,
    e(steel,sicartsa)
                    (.LO, .L, .UP, .M = 0, 0, +INF, 0)
           -1       mbf(steel,sicartsa)
            1       me(steel)
         -140       aeps,
    e(steel,hylsa)
                    (.LO, .L, .UP, .M = 0, 0, +INF, 0)
           -1       mbf(steel,hylsa)
            1       me(steel)
           -6.8564  alam
         -140       aeps,
    e(steel,hylsap)
                    (.LO, .L, .UP, .M = 0, 0, +INF, 0)
           -1       mbf(steel,hylsap)
            1       me(steel)
           -5.126   alam
         -140       aeps

One can alternatively filter certain variables by using the ``filters`` argument. For example, if one only wants to see 
the variables for hylsa and ahmsa plants, they can provide the elements as follows: ::

  import gamspy as gp
  m = gp.Container()
  ...
  ...
  model_definition_goes_here
  ...
  ...
  model.solve(options=Options(variable_listing_limit=100))
  print(mr.getVariableListing(filters=[[], ['hylsa', 'ahmsa']]))

``filters`` argument is a list of lists where each list specifies the elements to be gathered. 
If an empty list is given as in the example above, it means all elements. 

Number of variables returned can be filtered with ``n`` argument. For example, if ``n`` is set to 1,
the function return only the first variable.

.. note::

  Length of the ``filters`` argument must be equal to the dimension of the variable.