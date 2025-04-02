.. _variable:

.. meta::
   :description: Documentation of GAMSPy Variable (gamspy.Variable)
   :keywords: Variable, GAMSPy, gamspy, mathematical modeling, sparsity, performance

********
Variable
********

Introduction
============

This chapter covers the declaration and manipulation of GAMSPy 
:meth:`Variables <gamspy.Variable>`. Many concepts from previous chapters apply 
here as well.

A variable is the GAMSPy name for what are called *endogenous variables* by 
economists, *columns* or *activities* by linear programming experts, and 
*decision variables* by industrial Operations Research practitioners. They are 
the entities whose values are generally unknown until after a model has been 
solved. A crucial difference between GAMSPy variables and columns in traditional 
mathematical programming terminology is that one GAMSPy variable is likely to be 
associated with many columns in the traditional formulation.

Variable Declarations
=====================

A GAMSPy variable, like all other identifiers, must be declared before it may be 
referenced.

The Syntax
----------

The declaration of a variable is similar to a set or parameter declaration, in 
that domain lists and descriptions are allowed and recommended ::
    
    from gamspy import Container, Set, Variable

    m = Container()
    t = Set(m, name="t", description="time periods", records=range(1990, 2000))

    k = Variable(m, domain=t, description="capital stock (trillion rupees)")
    c = Variable(m, domain=t, description="consumption (trillion rupees per year)")
    i = Variable(m, domain=t, description="investment (trillion rupees per year)")
    utility = Variable(m, name="utility", description="utility measure")


The :meth:`Variable <gamspy.Variable>` class indicates that this is a variable 
statement. An external name can be supplied with the optional ``name`` attribute. 
This name needs to follow to requirements of an *identifier*. In the optional ``domain``
list the set or sets may be specified over which an indexed variable is declared.
The optional ``description`` may be used to describe the variable for future reference
and to ease readability. 

The declaration of ``k`` above implies, as usual, that references to ``k`` are restricted to 
the domain of the set ``t``. A model that includes ``k`` will probably have several 
corresponding variables in the associated mathematical programming problem: most likely one 
for each member of ``t``. In this way, very large models can be constructed using a small 
number of variable objects. It is quite unusual for a model to have as many as 50 distinct 
variable objects.

.. note::
    - GAMSPy variables can have up to 20 dimensions.
    - The sets over which variables are declared indicate that these variables are potentially 
      defined for every element of the defining sets. However the actual definition of variables 
      does not occur until variables appear in an :ref:`equation definition <equation>` where the 
      equation needs to be part of a :ref:`model` that in turn occurs in a solve statement.

.. _variable-types:

Variable Types
--------------

There are nine basic :meth:`variable types <gamspy.VariableType>` that may be used in 
variable statements: 

=================================================  ==================================================================================================================================================================================================  ======================  ======================
Keyword                                            Description                                                                                                                                                                                         Default Lower Bound     Default Upper Bound
=================================================  ==================================================================================================================================================================================================  ======================  ======================
free (default)                                     No bounds on variable. Both bounds may be changed from the default values by the user.                                                                                                              float('-inf')           float('inf')
positive                                           No negative values are allowed for variable. The user may change both bounds from the default value.                                                                                                0                       float('inf')
negative                                           No positive values are allowed for variables. The user may change both bounds from the default value.                                                                                               float('-inf')           0
binary                                             Discrete variable that can only take values of 0 or 1. For details see section :ref:`types-of-discrete-variables`. In relaxed Model types the integrality requirement is relaxed.                   0                       1
integer                                            Discrete variable that can only take integer values between the bounds. In relaxed Model types the integrality requirement is relaxed.                                                              0                       float('inf')
sos1                                               A set of variables, such that at most one variable within a group may have a non-zero value. For details see section :ref:`types-of-discrete-variables`.                                            0                       float('inf')
sos2                                               A set of variables, such that at most two variables within a group may have non-zero values and the two non-zero values are adjacent. For details see section :ref:`types-of-discrete-variables`.   0                       float('inf')
semicont                                           Semi-continuous, must be zero or above a given minimum level. For details see section :ref:`types-of-discrete-variables`.                                                                           1                       float('inf')
semiint                                            Semi-integer, must be zero or above a given minimum level and integer. For details see section :ref:`types-of-discrete-variables`. In relaxed Model types the integrality requirement is relaxed.   1                       float('inf')
=================================================  ==================================================================================================================================================================================================  ======================  ======================

The default type is ``free``, which means that if the type of the variable is not 
specified, it will not be bounded at all. The type ``positive variables`` is used for variables for which 
negative values are meaningless, such as capacities or quantities. Note that 
bounds may be changed using variable attributes and assignment statements, see section 
:ref:`variable-attributes`.

.. note::
    - If a model is unbounded, a frequent cause for the unboundedness is that the 
      modeler forgot to make a variable positive.


.. _variable-attributes:

Variable Attributes
===================

Introduction
------------

While a GAMSPy :ref:`parameter <parameter>` has one number associated with each unique 
label combination, a variable has several attributes. They represent:

=========  ================== ======================================================================================================================================================================================================================================================================================================================================
Attribute  Data column name   Description
=========  ================== ======================================================================================================================================================================================================================================================================================================================================
lo         lower              Lower bound for the variable. Set by the user either explicitly or through default values associated with the variable type.
up         upper              Upper bound for the variable. Set by the user either explicitly or through default values associated with the variable type.
fx         -                  A fixed value for the variable. If set, it results in the upper and lower bounds of the variable being set to the value of the fx attribute.
l          level              Activity level for the variable, also the current value or starting point. This attribute is reset to a new value when a model containing the variable is solved. The activity level is used to construct a `basis <https://www.gams.com/latest/docs/UG_SolverUsage.html#ADVANCED_USAGE_Basis>`_  for the model.
m          marginal           The marginal value (or reduced cost) for the variable. This attribute is reset to a new value when a model containing the variable is solved. The activity level is used to construct a `basis <https://www.gams.com/latest/docs/UG_SolverUsage.html#ADVANCED_USAGE_Basis>`_  for the model.
scale      scale              Numerical scaling factor for all coefficients associated with the variable. Only applicable for continuous variables.
prior      -                  Branching priority value used in mixed integer programming models. Only applicable for discrete variables.
stage      -                  This attribute allows you to assign variables to stages in a stochastic program or other block-structured model. Thus, among other places, it is used for 2-stage stochastic programs, for example solved with the Benders partition in `Cplex <https://www.gams.com/latest/docs/S_CPLEX.html#CPLEX_BENDERS_ALGORITHM>`_.
=========  ================== ======================================================================================================================================================================================================================================================================================================================================

If the data is in a convenient format, it is possible to specify (initial) values for these 
variable attributes within the variable constructor. This is an optional keyword argument 
and internally the variable constructor will simply call the ``setRecords`` method. In contrast 
to the ``setRecords`` methods in either the Set or Parameter classes the ``setRecords`` method 
for variables is more restricted. The `GAMS Transfer Python documentation <https://www.gams.com/latest/docs/API_PY_GAMSTRANSFER_MAIN_CLASSES.html#PY_GAMSTRANSFER_ADD_VARIABLE_RECORDS>`_  gives examples including pandas dataframes and specially structured ``dict`` for creating 
records from matrices. This restriction is out of necessity because to properly set a record 
for a variable the user passes data for the ``level``, ``marginal``, ``lower``, ``upper`` and 
``scale`` attributes. Any missing attributes will be filled in with the default 
record values (see :ref:`variable-types`). 

.. note::
    - ``fx`` sets ``lo``, ``up``, and ``l``.
    - The attribute ``stage`` uses the same internal space as ``scale`` and ``prior``. 
      Attribute ``scale`` is applicable for 
      continuous variables only and attribute ``prior`` is for discrete variables only, hence 
      they can share the same internal space in a GAMSPy variable. Some solvers can make use of 
      priorities even for continuous variables (e.g. 
      `BARON <https://www.gams.com/latest/docs/S_BARON.html#BARON_THE_BARON_OPTIONS>`_). 
      Such priorities need to be supplied via ``solver_options`` in the 
      :meth:`solve <gamspy.Model.solve>` function.
    - Attributes ``fx``, ``prior``, and ``stage`` cannot be set via the constructor ``records``
      argument, not via the ``setRecords`` method. For ``fx`` the user needs to fill the columns
      ``lower``, ``upper``, and ``level`` instead. For ``scale``, ``prior``, and ``stage`` the
      ``stage`` column needs to be filled and the variable context decides about the use of the
      values.
    - For discrete variable types, the consequences of the type declaration cannot be 
      completely undone (e.g. the ``scale`` attribute is not available) but their value 
      domain can be changed to continuous by setting attribute ``prior`` to infinity.      
    - Fixing a semi-continuous or semi-integer variable to a non-zero value like ``4`` 
      does not result in a truly fixed variable. The domain of the variable remains 
      ``{0,4}``. To really fix a semi-continuous or semi-integer variable, the discrete 
      restriction could be relaxed by setting the branching priority (``prior``) to 
      infinity.
    - For variables in discrete models (such as MIP, MINLP), the ``m`` attribute 
      provides the marginals obtained by fixing all the discrete variables and solving 
      the resulting continuous problem (such as LP, NLP). Many solvers allow to 
      enable/disable solving such a fixed problem. When disabled, no marginals will 
      be provided for discrete models.

In addition to the variable attributes introduced above, there are a number of variable 
attributes that cannot be assigned but may be referenced in assignment statements.

=========  =========================================================================================================================================================================================================================================================================================================================================
Attribute     Description
=========  =========================================================================================================================================================================================================================================================================================================================================
range      The difference between the lower and upper bounds for a variable. It becomes zero if the lower equals the upper bound, e.g. if the ``fx`` attribute is set.
slackup    Slack from variable upper bound. This is defined as the greater of two values: zero or the difference between the upper bound and the level value of a variable.
slacklo    Slack from variable lower bound. This is defined as the greater of two values: zero or the difference between the level value and the lower bound of a variable.
slack      Minimum slack from variable bound. This is defined as the minimum of two values: the slack from the variable lower bound and the slack from the variable upper bound.
infeas     Amount by which a variable is infeasible falling below its lower bound or above its upper bound. This is defined as the smallest of three values: zero, the difference between the lower bound and the level value, the difference between the level value and the upper bound of a variable, i.e. ``max[0, lower-level, level-upper]``.
=========  =========================================================================================================================================================================================================================================================================================================================================

Bounds on Variables
-------------------

All default bounds set at declaration time may be changed using assignment statements.

Bounds on variables are the responsibility of the user. After variables have been declared, 
default bounds have already been assigned: for many purposes, especially in linear models, 
the default bounds are sufficient. In nonlinear models, however, bounds play a far more 
important role. It may be necessary to provide bounds to prevent undefined operations, 
such as division by zero. In nonlinear programming it is often necessary to define a 
'reasonable' solution space that will assist in efficiently finding a solution.

.. warning::
    The lower bound cannot be greater than the upper bound: if you happen to impose such 
    a condition, GAMSPy will raise an exception when executing the :meth:`solve <gamspy.Model.solve>` 
    function.


Fixing Variables
----------------

GAMSPy allows the user to fix variables through the ``fx`` variable attribute in assignment statements. This is almost 
equivalent to setting the lower bound and upper bound equal to the fixed value. The attribute 
``fx`` also resets the activity level ``l`` to the fixed value. When setting ``lo`` and ``up`` 
the activity level remains unchanged. A solve will project the activity level within 
the active bounds. Fixed variables can subsequently be freed by changing the lower and upper 
bounds.

Activity Levels of Variables
----------------------------

GAMSPy allows the user to set the activity levels of variables through the ``l`` variable 
attribute in assignment statements. These activity levels of the variables prior to the
:meth:`solve <gamspy.Model.solve>` function serve as 
initial value for the solver. This is particularly important for nonlinear programming 
problems. For discrete models in many cases the solver needs an additional indicator to 
interpret the activity levels as a feasible integer solution via a solver option 
(e.g. Cplex' `mipstart <https://www.gams.com/latest/docs/S_CPLEX.html#CPLEXmipstart>`_).

.. note::
    - GAMSPy only stores variables with non-default values (similar to storing only non-zero 
      values of parameters). Non-default variables can be accidentally created by using 
      harmlessly looking assignments like ``x.up[i,j,k,l] = 0``.
      Even if the equations only reference such variables over a small subset of [i,j,k,l] 
      this statement creates :math:`|i|⋅|j|⋅|k|⋅|l|` variable records in the GAMSPy 
      database. Such fixings of ``x[i,j,k,l]`` to 0 can be avoided by using .
      :ref:`dynamic sets in the equation algebra <conditional-equations-with-dynamic-sets>` 
      to only reference tuples of ``[i,j,k,l]`` for which ``x[i,j,k,l]`` can possible have a non-zero value.
    - In order to filter only necessary tuples for an equation the filtering conditions needs 
      to be provided only once when defining the equation (``equ[i,j,k]``). This is different for 
      variables because they appear in many equations and the filtering condition needs to be 
      potentially repeated many times. Therefore it is good practice and reduces GAMSPy model 
      generation time if the filtering of the variables is governed by a dynamic set: ::

          Sum(Domain(i, j).where[Ord(i) > Ord(j) & cap[i, j] > 0], x[i, j])

      versus ::

          net = Set(m, domain=[i, j])
          net[i, j] = Ord(i) > Ord(j) & cap[i, j] > 0
          Sum(net[i, j], x[i, j])

      Alternatively, the ``limited_variables`` argument to the :meth:`Model <gamspy.Model>` constructor 
      can be used to limit the tuples of a variable during model generation in the :meth:`solve <gamspy.Model.solve>`
      function.

Printing Filtered Variable Records
----------------------------------

It is often useful to print the records of Variable symbols but the number of records in a Variable symbol can sometimes 
be quite large or you might just be interested in values of only one attribute of the symbol (e.g. marginals). In this case,
instead of printing the whole records with: ::

  print(your_variable.records)

you can print only the records of only one attribute as follows: ::
  
  from gamspy import Container, Set, Variable
  m = Container()
  i = Set(m, records=['elem1', 'elem2', 'elem3'])
  j = Set(m, records=['elem4', 'elem5', 'elem6'])
  v = Variable(m, domain=[i,j])

  ...
  ...
  ...
  your_model_definition here
  ...
  ...
  ...

  model.solve()

  print(your_variable.m.records)
  print(your_variable.m[i, j].records)
  print(your_variable.m[i, 'elem6'].records)
  print(your_variable.m['elem1', 'elem6'].records)

The first and second print would only print the marginals of the variable. 
The third print would only print the marginals of the records where the `j` element is equal to `elem6`. 
And the fourth print would only print the marginal of 'elem1', 'elem6' pair. 

One can also use slice and ellipsis operators to match certain indices: ::

  import gamspy as gp

  m = gp.Container()
  i1 = gp.Set(m, name="i1", records=range(2))
  i2 = gp.Set(m, name="i2", records=range(2))
  i3 = gp.Set(m, name="i3", records=range(2))
  i4 = gp.Set(m, name="i4", records=range(2))
  v1 = gp.Variable(m, "v1", domain=[i1, i2, i3, i4])
  v1.generateRecords(seed=1)
  
::

  In [0]: v1.l[i1, i2, i3, i4].records
  Out[0]:
     i1 i2 i3 i4     level
  0   0  0  0  0  0.511822
  1   0  0  0  1  0.950464
  2   0  0  1  0  0.144160
  3   0  0  1  1  0.948649
  4   0  1  0  0  0.311831
  5   0  1  0  1  0.423326
  6   0  1  1  0  0.827703
  7   0  1  1  1  0.409199
  8   1  0  0  0  0.549594
  9   1  0  0  1  0.027559
  10  1  0  1  0  0.753513
  11  1  0  1  1  0.538143
  12  1  1  0  0  0.329732
  13  1  1  0  1  0.788429
  14  1  1  1  0  0.303195
  15  1  1  1  1  0.453498

  In [1]: v1.l['0', ..., '1'].records
  Out[1]:
    i1 i2 i3 i4     level
  1  0  0  0  1  0.950464
  3  0  0  1  1  0.948649
  5  0  1  0  1  0.423326
  7  0  1  1  1  0.409199

  In [2]: v1.l['0', :, '1', '1'].records
  Out[2]:
    i1 i2 i3 i4     level
  3  0  0  1  1  0.948649
  7  0  1  1  1  0.409199

Here we first show all the generated level values in cell 0. Then, cell 1 matches all 
records where the first dimension is '0' and the last dimension is '1'. It uses the ellipsis operator 
to match all elements of the second and the third column. Cell 2 matches all records where the first dimension is 
'0', and the third and fourth dimensions are '1'. It makes use of the slice operator 
to match all elements of the second dimension. 


Variables in Assignment Statements
==================================

Assigning Values to Variable Attributes
---------------------------------------

Assignment statements operate on one variable attribute at a time, and require the suffix to 
specify which attribute is being used. Any index list comes after the suffix. ::

    x.up[c, i, j] = 1000
    phi.lo[...] = -float('inf')

A very common use is to bound one particular entry individually: ::

    p.up['pellets', 'ahmsa', 'mexico-df'] = 200

Or to put small lower bounds on a variable identifier used as a divisor in a nonlinear program: ::

    c.lo[t] = 1e-4

Or to provide initial values for a nonlinear problem: ::

    c.l[t] = 4 * cinit[t]

Unlike assignment to scalar parameters, it is also possible to do an assignment without any index to scalar variables: ::

    phi.l = 5

Remember that the order is important in assignments, and notice that the two pairs of 
statements below produce very different results. In the first case, the lower bound for 
``c['1985']`` will be 0.01, but in the second, the lower bound is 1. ::

    c.fx['1985'] = 1     
    c.lo[t] = 0.01
    
::

    c.lo[t] = 0.01          
    c.fx['1985'] = 1

Everything works as described in the previous chapters, including the various mechanisms 
described there of indexed operations, subset assignments and so on. ::

    ship_sm.lo[sl, m].where[Ord(sl) = 1 & Ord(m) = 1] = 1

The lower bound of the variable ``ship_sm[sl, m]`` is set to 1 and this assignment is only 
carried out for first set elements of ``sl`` and ``m``, e.g.  ``ship_sm['s1','d1']``.

Variable Attributes in Assignments
----------------------------------

The following examples illustrate the use of variable attributes on the right-hand side of 
assignment statements: ::

    g.l[t] = mew[t] + xsi[t] * m.l[t] 
    h.l[t] = gam[t] - alp[t] * e.l[t] 
    
::
    
    # generating report after solve 
    cva = Sum(i, v.l[i] * x.l[i])  
    cli = Sum(i, p.l[i] * ynot[i])/Sum(i, ynot[i])
    rva = cva / cli

As with parameters, a variable must have some non-default data values associated with it 
before it can be used on the right-hand side of an assignment statement. After a solve 
statement has been processed or if non-default values have been set with an assignment 
statement, this condition is satisfied. 


.. _types-of-discrete-variables:

Types of Discrete Variables
===========================

GAMSPy provides six discrete variable types: ``binary``, ``integer``, ``sos1``, ``sos2``, 
``semicont``, and ``semiint``. In the following subsections we will present details and 
examples for each of these discrete variable types. Note that if any discrete variable 
is part of a model, it has to be a mixed integer model or one of the related model types, 
like ``MINLP`` or ``RMINLP``. See section 
`Classification of Models <https://www.gams.com/latest/docs/UG_ModelSolve.html#UG_ModelSolve_ModelClassificationOfModels>`_ 
for a full listing of all GAMSPy model types.

.. _binary-variables:

Binary Variables
----------------

Binary variables can take values of 0 (zero) and 1 (one) only. ::

    from gamspy import Container, Set, Alias, Variable, Equation, Sum

    m = Container()
    k = Set(m, "k", description="rows", records=["row1", "row2", "row3", "row4"])
    l = Set(m, "l", description="columns", records=["col1", "col2", "col3", "col4"])
    v = Set(m, "v", description="values", records=["val1", "val2", "val3", "val4"])
    
    i = Alias(m, name="i", alias_with=v)
    j = Alias(m, name="j", alias_with=v)
    
    x = Variable(m, domain=[i, j, k, l], type="binary", description="pairs (i,j) allocated to cell(k,l)")
    c1 = Equation(m, domain=[i, j], description="for each cell pick only one item pair")
    c1[i, j] = Sum((k, l), x[i, j, k, l]) == 1

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
solver link because even small rounding can lead to significant infeasibilities.

A binary variable can also have a truly fractional value after a solver if the model status 
does not indicate a feasible integer solution (model status :meth:`OptimalGlobal <gamspy.ModelStatus.OptimalGlobal>` 
or :meth:`Integer <gamspy.ModelStatus.Integer>`).


Integer Variables
-----------------

Integer variables are discrete variables that can take only integer values between their bounds. 
The user may change both bounds from the default value. The default lower bound is 0 (zero) 
and the default upper bound inside GAMSPy is ``float('inf')``, and the same upper bound is passed on 
to the solver.

Note that in relaxed model types the integrality requirement is relaxed. ::

  from gamspy import Container, Set, Variable

  m = Container()
  t = Set(
      m,
      "t",
      records=["12pm-6am", "6am-9am", "9am-3pm", "3pm-6pm", "6pm-12pm"],
      description="demand blocks",
  )
  g = Set(m, "g", records=["type-1", "type-2", "type-3"], description="generators")

  x = Variable(m, domain=[g, t], description="number of generators in use")
  cost = Variable(m, description="total operating cost (l)")
  n = Variable(m, domain=[g, t], type="integer", description="generator output (1000mw)")

The integer variable ``n`` models the number of generators of various types that are in 
use at any of the time blocks.

Special Order Sets of Type 1 (SOS1)
-----------------------------------

SOS1 variables are a set of variables, such that at most one variable within the group 
may have a nonzero value. This variable may take any positive value. ::

    s1 = Variable(m, type="sos1", domain=i)
    t1 = Variable(m, type="sos1", domain=[k, j])
    w1 = Variable(m, type="sos1", domain=[i, j, k])

Note that the members of the innermost (the right-most) index belong to the same SOS set. 
For example in the sets defined above, ``s1`` represents one special ordered set of type 
1 with ``i`` elements, ``t1`` defines ``k`` sets with ``j`` elements each and ``w1`` 
defines ``[i, j]`` sets with ``k`` elements each.

The default bounds for ``SOS1`` variables are ``zero`` and ``float('inf')``. As with any other 
variable, the user may change these bounds. Further, the user may explicitly provide 
whatever convexity row that the problem may need through an equation that requires 
the members of the ``SOS1`` set to be less than a certain value. Any such convexity 
row will implicitly define bounds on each of the variables.

Consider the following example: ::

    s1 = Variable(m, type="sos1", domain=i)
    
    defsoss1 = Equation(m)
    defsoss1 = Sum(i, s1[i]) <= 3.5

The equation ``defsoss1`` implicitly defines the nonzero value that one of the elements 
of the ``SOS1`` variable ``s1`` may take as equal to or smaller than ``3.5``. Note that 
it is also possible that all variables ``s1`` equal zero.

A special case arises when one of the elements of the set has to be nonzero and equal to 
a number, say 3.5. In this case equation ``defsoss1`` will be: ::

    defsoss1 = Sum(i, s1[i]) == 3.5

Frequently the nonzero value equals 1. As a result, the ``SOS1`` variable is effectively 
a binary variable. It is only treated differently by the solver at the level of the 
branch and bound algorithm. For example, consider the following example where we want 
to model that one out of n options has to be selected. This is expressed as: ::

    x = Variable(m, type = "sos1", domain = i)
    
    defx = Equation(m)
    defx = Sum(i, x[i]) == 1

The variable ``x`` can be made binary without any change in meaning and the solution 
provided by the solver will be indistinguishable from the ``SOS1`` case.

The use of special ordered sets may not always improve the performance of the branch 
and bound algorithm. If there is no natural order the use of binary variables may be 
a better choice. A good example of this is the classical assignment problem 
(see [H.P. Williams (2013) `Model Building in Mathematical Programming <https://books.google.de/books?id=YJRh0tOes7UC>`_], 
Wiley, Section 9.3.

Special Order Sets of Type 2 (SOS2)
-----------------------------------

``SOS2`` variables are a set of variables, such that at most two variables within the 
set may have nonzero values and these variables have to be adjacent. This requirement 
implies that the set is ordered, see chapter :ref:`ordered-sets` for details on ordered 
sets in GAMSPy. Note that the nonzero variables may take any positive value. ::

    i = Set(m, "i", records=[¨f"i{i}" for i in range(5)])
    
    s2 = Variable(m, type="sos2", domain=i)
    t2 = Variable(m, type="sos2", domain=[k ,j])
    w2 = Variable(m, type="sos2", domain=[i, j, k])

The members of the innermost (the right-most) index belong to the same set. For example, 
in the sets defined above, ``s2`` represents one special ordered set of type 2 with 
elements for each member of the set ``i``. At most two variables ``s2`` may be nonzero 
and they must reference adjacent elements of the set ``i``. Note that the variables 
``s2['i0']`` and ``s2['i1']`` are adjacent, but the variables ``s2['i0']`` and ``s2['i2']`` 
are not. Further, ``t2`` defines ``k`` sets of ``SOS2`` variables with ``j`` elements 
each and the adjacency requirement refers to the set ``j`` which must be ordered. 
Similarly, ``w2`` defines ``[i, j]`` sets with ``k`` elements each and the adjacency 
requirement refers to the set ``k`` which must be ordered.

The default bounds for ``SOS2`` variables are ``zero`` and ``float('inf')``. As with any other 
variable, the user may change these bounds. ``SOS2`` variables are most often used to 
model piece-wise linear approximations to nonlinear functions. 


Semi-Continuous Variables
-------------------------

Semi-continuous variables are either zero or above a given minimum level. This can be 
expressed algebraically as: either :math:`x = 0` or :math:`lo <= x <= up` By default, the lower 
bound :math:`lo` is 1 and the upper bound :math:`up` is ``float('inf')``. As usual, these 
bounds may be changed with the variable attributes ``lo`` and ``up``. ::

    x = Variable(m, type="semicont")
    x.lo = 1.5
    x.up = 23.1

The slice of code above declares the variable ``x`` to be a semi-continuous variable that may 
either be zero or behave as a continuous variable between 1.5 and 23.1.

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

        forceLBnd = Equation(m, description="Force x to be greater than scLow if b is 1")
        forceZero = Equation(m, description="Force x to be zero if b is zero")

        forceLBnd = x >= scLow*b
        forceZero = x <= x.up*b

Semi-Integer Variables
----------------------

Semi-integer variables are either zero or integer and above a given minimum value. This can be 
expressed algebraically as: either :math:`x = 0` or :math:`x \in {lo,...,up}`. By default, the 
lower bound :math:`lo` is 1 and the upper bound :math:`up` inside GAMSPy is ``float('inf')`` and 
the same values are passed on to the solver. As usual, these default bounds may be changed with 
the variable attributes ``lo`` and ``up``. Note that in relaxed model types the integrality 
requirement is relaxed. ::

    x = Variable(m, type="semiint")
    x.lo = 2
    x.up = 25

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
      and no implicit bound can be easily derived. If a finite upper bound is available, it can be computationally more efficient 
      to replace the semi-integer variable ``si``, with lower bound ``siLow``, by an integer 
      variable ``i`` and a binary variable ``b`` and the following equations: ::

        forceLBnd = Equation(m, description="Force i to be greater than siLow if b is 1")
        forceZero = Equation(m, description="Force i to be zero if b is zero")

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

  model.solve(options=Options(variable_listing_limit=100))

::

  In [1]: e.getVariableListing()
  Out[1]:
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

.. note::
    The variable listing provides information about the value of the level (``.L``) and the
    marginal (``.M``) of the variables. This information is based on
    the *input* point, not the solution that is calculated by the solve.

One can alternatively filter certain variables by using the ``filters`` argument. For example, if one only wants to see 
the variables for hylsa and ahmsa plants, they can provide the elements as follows: ::

  In [2]: e.getVariableListing(filters=[[], ['hylsa', 'ahmsa']]))
  Out[2]:
      e(steel,ahmsa)
                      (.LO, .L, .UP, .M = 0, 0, +INF, 0)
             -1       mbf(steel,ahmsa)
              1       me(steel)
             -8.6876  alam
           -140       aeps,
      e(steel,hylsa)
                      (.LO, .L, .UP, .M = 0, 0, +INF, 0)
             -1       mbf(steel,hylsa)
              1       me(steel)
             -6.8564  alam
           -140       aeps,

``filters`` argument is a list of lists where each list specifies the elements to be gathered. 
If an empty list is given as in the example above, it means all elements. 

Number of variables returned can be filtered with ``n`` argument. For example, if ``n`` is set to 1,
the function return only the first variable.

.. note::

  Length of the ``filters`` argument must be equal to the dimension of the variable.
