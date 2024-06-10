.. _conditional_expressions_assignments_equations:

******************************************************
Conditional Expressions, Assignments and Equations
******************************************************

Introduction
=============

This chapter deals with the way in which conditional assignments, expressions and 
equations are made in GAMS. The index operations already described are very 
powerful, but it is necessary to allow for exceptions of one sort or another. 
For example, heavy trucks may not be able to use a particular route because of a 
weak bridge, or some sectors in an economy may not produce exportable products. 
Exceptions such as these may easily be modeled with a logical condition combined 
with the ``where`` operator, a very powerful feature of GAMSPy introduced in 
this chapter.

The 'where' Statement
=====================

The ``where`` operator is a very powerful feature in GAMSPy. The general syntax 
for a conditional expression is: ::

    term.where[logical_condition]

Here, ``term`` can be a number, a (indexed) symbol, and also a complex expression. 
The ``where`` operator may be read as '*under the condition that the following* 
``logical_condition`` *evaluates to TRUE*'.

Consider the following simple condition, where ``u[i]``, ``s[i]`` and ``v[i]`` are 
parameters with index ``i``: ::

    u[i].where[~ s[i]] = v[i]

Note that the ``term`` is the parameter ``u[i]`` and the ``logical condition`` is 
the expression ``~ s[i]``. If the condition is not satisfied, no assignment is made. 
To make it clear, this conditional assignment may be read as: '*given that s[i] does 
not exist, u[i] equals v[i]*'.

Logical conditions may take various forms, they are introduced in the next section. 
Conditional expressions may be used in the context of assignments, indexed 
operations and equations. These topics are covered in later sections of this chapter.

.. note::
    Logical conditions used with the ``where`` operator cannot include variables. 
    However, variable attributes are allowed.

.. note::
    The ``where`` operator is equivalent to the dollar condition ``$`` in GAMS.


Logical Conditions
===================

Logical conditions are special expressions that evaluate to either TRUE or 
FALSE. Logical conditions may be numerical expressions and numerical relations and 
they may refer to set membership. In the following subsections this is shown in the 
context of simple conditional assignments with the ``where`` operator on the 
left-hand side.

.. 
    In this section we use many examples to illustrate the concepts that are being 
    introduced. In all these examples ``a`` and ``b`` are scalars, ``s``, ``t``, ``u`` 
    and ``v`` are parameters, and ``i`` and ``j`` are sets.

Numerical Expressions
----------------------

Numerical expressions may serve as logical conditions: a result of zero is treated as 
the logical value FALSE and a non-zero result is treated as the logical value TRUE. 
The following example illustrates this point. ::

    from gamspy import Container, Set, Parameter, Number
    m = Container()
    
    i = Set(m, name="i", records=["i1","i2","i3","i4","i5"])
    s = Parameter(m, "s", domain = i,
                  records = [["i1", 3],
                             ["i2", 5],
                             ["i3", 6]])
    u = Parameter(m, "v", domain = i)
    
    u[i].where[2*s[i]-6] = 7

::

    In [1]: u.records
    Out[1]:
    	 i	value
    0	i2	  7.0
    1	i3	  7.0

Here the numerical expression is the logical condition. The numerical expression is 
zero if ``a[i]`` equals 3, and non-zero otherwise. Hence the logical value of the 
expression is FALSE for ``a[i] = 3`` and TRUE for all other values of a[i]. The 
assignment is only made if the numerical expression evaluates to TRUE, otherwise 
no assignment is made.

.. note::
    - Values of the extended range arithmetic such as ``float("inf")`` are also 
      allowed in logical conditions. If the result of a numerical expression used as 
      a logical condition takes any of these values, the logical value is TRUE.
    - Observe that :meth:`gamspy.math` functions are also allowed in logical conditions. 
      If they evaluate to zero, the logical condition is FALSE, otherwise it is TRUE. 


.. _numerical-relational-operators:    

Numerical Relational Operators
--------------------------------

`Numerical relational operators <https://www.geeksforgeeks.org/relational-operators-in-python/>`_ 
compare two numerical expressions and return a logical value. Consider the following 
examples. ::

    from gamspy import Container, Set, Parameter, Number
    m = Container()
    
    i = Set(m, name="i", records=["i1","i2","i3","i4","i5"])
    s = Parameter(m, "s", domain = i,
                  records = [["i1", 3],
                             ["i2", 5],
                             ["i3", 6]])
    u = Parameter(m, "v", domain = i)
    
    u[i].where[s[i] >= 5] = u[i] + 10
    
::

    In [1]: u.records
    Out[1]:
    	 i	 value
    0	i2	  11.0
    1	i4	  10.0

The assignment ``u[i].where[s[i] >= 5] = u[i] + 10`` depends on whether ``s[i]`` is greater or 
equal to 5. If this is the case, an assignment is made, otherwise not.

.. _bitwise-operators:

Bitwise Operators
------------------

Bitwise operators can be used to combine two or more logical conditions to build complex logical 
expressions. For example, if several expressions are required to be TRUE simultaneously, they may 
be connected with the python bitwise operator ``&``. For all available bitwise operators in python 
read more `here <https://www.w3schools.com/python/gloss_python_bitwise_operators.asp>`_. Another 
way to construct complex logical conditions is by nesting them. For details, see subsection 
`Nested Conditions <nested-conditions>`_ below.

The following somewhat artificial examples serve as illustrations. ::

    from gamspy import Container, Set, Parameter
    m = Container()
    
    i = Set(m, name="i", records=["i1","i2","i3","i4","i5"])
    
    s = Parameter(m, "s", domain = i,
                  records = [["i1", 3],
                             ["i2", 5],
                             ["i4", 8]])
    
    t = Parameter(m, "t", domain = i,
                  records = [["i1", 13],
                             ["i2", 13],
                             ["i3", 13],
                             ["i4", 13]])
    
    u = Parameter(m, "u", domain = i,
                  records = [["i2", 1]])
    
    v = Parameter(m, "v", domain = i,
                  records = [["i1", 7],
                             ["i3", 2]])
    
    u[i].where[~ s[i]] = v[i]
    u[i].where[s[i] & u[i] & t[i]] = s[i]
    u[i].where[s[i] | v[i] | t[i]] = 4

Note that there are three conditional assignments for the parameter ``u``. In the first assignment 
the logical condition is ``~ s[i]``. This condition holds for all entries of ``s`` that are not 
specified and therefore zero by default: ``s['i3']`` and ``s['i5']``. Hence ``u['i3']`` and 
``u['i5']`` are assigned the values of ``v['i3']`` and ``v['i5']`` respectively. The value of 
``v['i3']`` is 2 and the value of ``v['i5']`` is zero by default. After the first assignment we 
have ``u['i2']=1`` and ``u['i3']=2``, all other values of ``u`` are zero. Note that the logical 
condition failed for ``u['i2']`` and therefore its value remained unchanged. ::

    In [1]: u.records
    Out[1]:
    	 i	value
    0	i2	  1.0
    1	i3	  2.0


The logical condition 
in the second assignment is ``TRUE`` for those labels of the set ``i`` that have non-zero entries 
in the parameters ``s``, ``u`` and ``t`` simultaneously. This condition holds only for ``i2``. 
Therefore ``u['i2']=s['i2']=5`` and all other values of ``u`` remain unchanged, resulting in 
non-zero values only for ``u['i2']`` and ``u['i3']``. ::

    In [2]: u.records
    Out[2]:
    	 i	value
    0	i2	  5.0
    1	i3	  2.0
    
The logical condition in the last assignment 
evaluates to ``TRUE`` for all labels of the set ``i`` that have at least one non-zero entry in the 
parameters ``s``, ``v`` and ``t``. This holds for all labels except for ``i5``. Therefore 
``u['i5']`` stays zero and all other values of ``u`` are changed to ``4``.
::

    	 i	value
    0	i1	  4.0
    1	i2	  4.0
    2	i3	  4.0
    3	i4	  4.0

These examples demonstrate the power of the ``where`` operator combined with bitwise operators. 
Even more complex logical conditions are possible; see subsection 
`Mixed Logical Conditions <mixed-logical-conditions>`_ below for details.

Set Membership and Set Functions
---------------------------------

Apart from numerical and relational expressions, set membership and functions referencing set 
elements may be used as a logical condition. Consider the following example as illustration 
for set membership as logical condition. ::

    from gamspy import Container, Set, Parameter
    m = Container()
    
    i = Set(m, name="i", records=["i1","i2","i3","i4","i5"])
    j = Set(m, name="j", records=["i1","i2","i3"], domain = i)
    
    s = Parameter(m, "s", domain = i,
                  records = [["i1", 3],
                             ["i2", 5],
                             ["i3", 11],
                             ["i4", 8],
                             ["i5", 1]])
    
    t = Parameter(m, "t", domain = i)

    t[i].where[j[i]] = s[i] + 3

::
    
    In [1]: t.records
    Out[1]:
    	 i	value
    0	i1	  6.0
    1	i2	  8.0
    2	i3	 14.0

Note that the set ``j`` is a subset of the set ``i`` and that the parameter ``t`` is declared 
but not defined. The conditional expression ``t[i].where[j[i]]`` in the last line restricts 
the assignment to the members of the subset ``j`` since only they satisfy the condition 
``j[i]``. The values for ``t['i4']`` and ``t['i5']`` remain unchanged. In this case, this 
means that they are zero (by default). Note that there is an alternative formulation for 
this type of conditional assignment; for details see subsection 
`Filtering Sets in Assignments <filtering-sets-in-assignments>`_ below.

.. note::
    Only the membership of subsets and dynamic sets may be used as logical conditions.

The use of set membership as a logical condition is an extremely powerful feature of GAMSPy, 
see section `Conditional Equations <conditional-equations>`_ below for more examples.

Logical conditions may contain the method `<sameas>`_ or set `operators <card_ord>`_ 
that return particular values depending on the position of elements in sets, the size of 
sets or the comparison of set elements to each other or text strings. In the following 
example we have two sets of cities and we want to know how many of them feature in both 
sets. ::

    from gamspy import Container, Set, Parameter, Sum, Domain
    m = Container()
    
    i = Set(m, name="i", records=["Beijing","Calcutta","Mumbai","Sydney","Johannesburg","Cairo "])
    j = Set(m, name="j", records=["Rome","Paris","Boston","Cairo","Munich","Calcutta","Barcelona "])
    
    b = Parameter(m, "b")
    
    b[...] = Sum(Domain(i,j).where[i.sameAs(j)],1)

In the assignment statement we :meth:`Sum <gamspy.Sum>` over both sets and we use :meth:`sameAs <gamspy.Set.sameAs>` to 
restrict the domain of the indexed operation to those label combinations ``(i,j)`` where ``sameAs`` 
evaluates to TRUE. Thus only identical elements are counted.

The operators `ord and card <card_ord>`_ are frequently used to single out the first or last element of 
an ordered set. For example, we may want to fix a variable for the first and last elements of a set: ::

    from gamspy import Container, Set, Variable, Ord, Card 
    m = Container()
    
    i = Set(m, name="i", records=["Beijing","Calcutta","Mumbai","Sydney","Johannesburg","Cairo"])
    j = Set(m, name="j", records=["Rome","Paris","Boston","Cairo","Munich","Calcutta","Barcelona"])
    
    x = Variable(m, "x", domain=[i])
    
    x.fx[i].where[Ord(i) == 1]       = 3
    x.fx[i].where[Ord(i) == Card(i)] = 7

In the first assignment the variable ``x`` is fixed for the first element of the set ``i`` and in 
the second assignment ``x`` is fixed for the final element of ``i``.

.. note::
    As an alternative to the formulation above, one could also use the set attributes 
    :meth:`first <gamspy.Set.first>` and :meth:`last <gamspy.Set.last>` to get the same result: 
    ::

        x.fx[i].where[i.first] = 3
        x.fx[i].where[i.last]  = 7
  

.. _mixed-logical-conditions:

Mixed Logical Conditions
-------------------------

The building blocks introduced in the subsections above may be combined to generate more complex 
logical conditions. These may contain standard arithmetic operations, 
`numerical relational operations <numerical-relational-operators>`_ and 
`logical/bitwise operations <bitwise-operators>`_. All operations, their symbols and their order 
of precedence are given below. Note that 1 denotes the highest order of precedence and 7 denotes 
the lowest order of precedence. As usual, the default order of precedence holds only in the 
absence of parentheses and operators (symbols) on the same level are evaluated from left to right.

=================================  ========================================  ======================  =====================
Type of Operation                  Operation                                 Operator                Order of precedence
=================================  ========================================  ======================  =====================
Standard arithmetic operation      Exponentiation, Floor division             \*\*, //               1
Standard arithmetic operation      Multiplication, Division                   \*, /                  2
Standard arithmetic operation      Unary operators: Plus, Minus               +, -                   3
Standard arithmetic operation      Binary operators: Addition, Subtraction    +, -                   3
Numerical Relational operation     All                                        <, <=, ==, !=, >=, >   4
Logical operation                  Negation                                   ~                      5
Logical operation                  Logical Conjunction                        &                      6
Logical operation                  All other logical operations               \|, ^, <<, >>          7
=================================  ========================================  ======================  =====================

.. note::
    We recommend to use parentheses rather than relying on the order of precedence of operators. 
    Parentheses prevent errors and make the intention clear.

Consider the following example: ::

    x - 5*y & z - 5
    (x - (5*y)) & (z-5)

These two complex logical conditions are equivalent. However, the parentheses make the second 
expression easier to understand.

Some simple examples of complex logical conditions, their numerical values, and their logical 
values are given below.

=============================  ================  ==============
Logical Condition              Numerical Value   Logical Value
=============================  ================  ==============
(1 < 2) + (3 < 4)              2                 TRUE
(2 < 1) & (3 < 4)              0                 FALSE
(4*5 - 3) + (10/8)             18.25             TRUE
(4*5 - 3) \| (10 - 8)          1                 TRUE
(4 & 5) + (2*3 <= 6)           2                 TRUE
(4 & 0) + (2*3 < 6)            0                 FALSE
=============================  ================  ==============


.. _nested-conditions:

Nested Conditions
------------------

An alternative way to model complex logical conditions is by nesting them. The syntax is: ::

    term.where[logical_condition1.where[logical_condition2.where[...]]]

Note that in nested conditions all succeeding expressions after the ``where`` operator must 
be enclosed in parentheses. The nested expression is equivalent to the following conditional 
expression that uses the logical operator ``&`` instead of nesting: ::

    term.where[logical_condition1 & logical_condition2 & ...]

Consider the following example.::

    from gamspy import Container, Set, Parameter
    m = Container()
    
    i = Set(m, name="i", records=["i1","i2","i3","i4","i5"])
    j = Set(m, name="j", records=["i1","i2","i3"], domain = i)
    k = Set(m, name="k", records=["i1","i2"], domain = i)
    
    u = Parameter(m, "u", domain = i)
    
    v = Parameter(m, "v", domain = i,
                  records = [["i1", 7],
                             ["i3", 2]])
    
    u[i].where[j[i].where[k[i]]] = v[i]

::

    In [1]: u.records
    Out[1]:
    	 i	value
    0	i1	  7.0

.. note::
    We recommend to use the logical ``&`` operator instead of nesting conditions, because 
    this formulation is easier to read.  

.. _conditional-assignments:

Conditional Assignments
=======================

A conditional assignment is an assignment statement with a ``where`` condition on the 
left-hand side or on the right-hand side. Most examples until now were conditional assignments 
with the ``where`` operator on the left.

.. warning::
    he effect of the ``where`` condition is significantly different depending on which side 
    of the assignment it is located.

The next two subsections describe the use of the ``where`` condition on each side of the 
assignment. Note that in many cases it may be possible to use either of the two forms of 
the ``where`` condition to describe an assignment. We recommend to choose the clearer 
formulation.

Note that if the logical condition in an assignment statement refers to set membership, 
then under certain conditions the restriction may be expressed without the use of the 
``where`` operator. For details, see section 
`Filtering Sets in Assignments <filtering-sets-in-assignments>`_ below.

.. _where-on-the-left:

where[] on the Left
--------------------

If the ``where`` condition is on the left-hand side of an assignment, an assignment is 
made only in case the logical condition is satisfied. If the logical condition is not 
satisfied then no assignment is made and the previous content of the parameter on the left 
will remain unchanged. In case the parameter on the left-hand side of the assignment has 
not previously been initialized or assigned any values, zeros will be used for any label 
for which the assignment was suppressed.

Consider the following example. Note that the parameter ``sig`` has been previously 
defined in the model. ::

    rho[i].where[sig[i] <> 0] = (1/sig[i]) - 1

In this assignment ``rho[i]`` is calculated and the ``where`` condition on the left 
protects against dividing by zero. If any of the values associated with the parameter 
``sig`` turns out to be zero, no assignment is made and the previous values of 
``rho[i]`` remain. As it happens, ``rho[i]`` was not previously initialized, and 
therefore all the labels for which ``sig[i]`` is zero will result in a value of zero.

Now recall the convention that non-zero implies TRUE and zero implies FALSE. The 
assignment above could therefore be written as: ::

    rho[i].where[sig[i]]  =  (1/sig[i]) - 1

In the following examples ``i`` is a set and ``s`` and ``t`` are parameters. ::

    s[i].where[t[i]] = t[i]
    s[i].where[(t[i]-1) > 0] = t[i]**0.5

Note that the first assignment is suppressed if the value of the parameter ``t`` equals 
zero. The second assignment is suppressed for values of the parameter ``t`` that are 
smaller or equal to 1.


.. _where-on-the-right:

where[] on the Right
--------------------

If the ``where`` condition is on the right-hand side of an assignment statement, an 
assignment will *always* be made. In case the logical condition is not satisfied the value 
of zero is assigned. Example: ::

    u[i].where[s[i] >= 5] = 7

Now we move the ``where`` condition to the right-hand side: ::

    u[i] = Number(7).where[s[i] >= 5]

This is equivalent to: ::

    if (s[i] >= 5)   then (u[i] = 7),    else (u[i] = 0)

Note that an ``if-then-else`` type of construct is implied, but the ``else`` operation is 
predefined and never made explicit. The else could be made explicit with the following 
formulation: ::

    u[i] = Number(7).where[s[i] >= 5] + Number(0).where[s[i] < 5]

The use of this feature is more apparent in instances when an ``else`` condition needs to 
be made explicit. Consider the next example. The set ``i`` is the set of ``plants``, and we 
are calculating ``mur[i]``, the cost of transporting imported raw materials. In some cases 
a barge trip must be followed by a road trip because the plant is not alongside the river 
and we must combine the separate costs. The assignment is: ::

    mur[i] = (1.0 + 0.0030 * ied[i,'barge']).where[ied[i,'barge']]
           + (0.5 + 0.0144 * ied[i,'road' ]).where[ied[i,'road' ]]

This means that if the entry in the distance parameter ``ied`` is not zero, then the cost 
of shipping using that link is added to the total cost. If there is no distance entry, 
there is no contribution to the cost, presumably because that mode is not used.

Consider another example for a conditional assignment with the ``where`` operator on 
the right: ::

    b = Sum(i, t[i]).where[a > 0] + 4

Here ``a`` and ``b`` are scalars, ``i`` is a set and ``t`` is a parameter. If the scalar 
``a`` is positive, the scalar ``b`` is assigned the sum of all values of the parameter 
``t`` plus 4. If ``a`` is zero or negative, ``b`` becomes just 4. Note that the sum is 
only computed if the condition holds, this potentially makes the program faster.

Conditional Indexed Operations
==============================

We have seen how exceptions in assignments are modeled with ``where`` conditions. 
``where`` conditions are also used in indexed operations, where they control the 
domain of operations. This is conceptually similar to the conditional assignment 
with the ``where`` on the left.

Consider the following example adapted from a gas trade model for interrelated gas 
markets. Here the set ``i`` contains supply regions and the parameter ``supc`` models 
supply capacities. The scalar ``tsupc`` is computed with the following statement: ::

    tsupc  =  Sum(i.where[supc[i] != float("inf")], supc[i])

This assignment restricts the :meth:`Sum <gamspy.Sum>` to the finite values of the 
parameter ``supc``.

In indexed operations the logical condition is often a set. This set is called the 
*conditional set* and assignments are made only for labels that are elements of the 
conditional set. This concept plays an important role in 
`dynamic sets <dynamic-sets>`_. 

Multi-dimensional sets are introduced in section 
`Multi-Dimensional Sets <multi-dimensional-sets>`_. In the example used there a 
two-dimensional set is used to define the mapping between countries and ports. 
Another typical example for a multi-dimensional set is a set-to-set mapping that 
defines the relationship between states and regions. This is useful for aggregating 
data from the state to the regional level. Consider the following example: ::

    from gamspy import Container, Set, Parameter, Sum
    import pandas as pd
    
    m = Container()
    
    r = Set(m, name = "r", description = "regions")
    s = Set(m, name = "s", description = "states")
    
    c = pd.Series(
        index=pd.MultiIndex.from_tuples([("north", "vermont"),
                                        ("north", "maine"),
                                        ("south", "florida"),
                                        ("south", "texas")])
    )
    
    corr = Set(m, name = "corr",
            domain = [r,s],
            uels_on_axes=True,
            domain_forwarding = True,
            records = c)
    
    y = Parameter(m, "y", domain = r, description = "income for each region")
    income = Parameter(m, "income", domain = s, description = "income for each state",
                      records = [["florida", 4.5],
                                ["vermont", 4.2],
                                ["texas", 6.4],
                                ["maine", 4.1]])

The set ``corr`` links the states to their respective regions, the parameter ``income`` 
is the income of each state. The parameter ``y`` is computed with the following assignment 
statement: ::

    y[r] = Sum(s.where[corr[r,s]], income[s])


The conditional set ``corr[r,s]`` restricts the domain of the summation: for each region 
``r`` the summation over the set ``s`` is restricted to the label combinations ``(r,s)`` 
that are elements of the set ``corr[r,s]``. Conceptually, this is analogous to the Boolean
value TRUE or the arithmetic value non-zero. The effect is that only the contributions of 
``vermont`` and ``maine`` are included in the total for ``north``, and ``south`` is the 
sum of the incomes from only ``texas`` and ``florida``. ::

    In [1]: y.records
    Out[1]:
    	    r	value
    0	north	  8.3
    1	south	 10.9

Note that the summation above can also be written as: ::

    y[r] = Sum(s,income[s].where[corr[r,s]])

In this formulation the parameter ``income`` is controlled by the conditional set ``corr`` 
instead of the index ``s``. Note that both formulations yield the same result, but the second 
alternative is more difficult to read.

Note that if the logical condition in the context of indexed operations refers to set 
membership, then under certain conditions the restriction may be expressed without the use of 
the ``where`` operator. For details, see section 
`Filtering Controlling Indices in Indexed Operations <filtering-controlling-indices-in-indexed-operations>`_ 
below.


.. _conditional-equations:

Conditional Equations
======================

The ``where`` operator is also used for exception handling in equations. The next two subsections 
discuss the two main uses of ``where`` operators in the context of equations: in the body of an 
equation and over the domain of definition.

Dollar Operators within the Algebra of Equations
---------------------------------------------------

A ``where`` operator in the algebraic formulation of an equation is analogous to the ``where`` 
on the right of assignments, as presented in section `where[] on the Right <where-on-the-right>`_. 
Assuming that "the right" means the right of the ``'='`` then the analogy is even closer. As in 
the context of assignments, an if-else operation is implied. It is used to exclude parts of the 
definition from some of the generated constraints. ::

    from gamspy import Container, Set, Variable, Equation, Sum
    m = Container()
    
    i =  Set(m,
             name = "i",
             description = "sectors",
             records = ["light-ind","food+agr","heavy-ind","services"])
    t =  Set(m,
             name = "t",
             domain = i,
             description = "tradables",
             records = ["light-ind","food+agr","heavy-ind"])
    
    x = Variable(m,"x",domain = i, description = "quantity of output")
    y = Variable(m,"y",domain = i, description = "final consumption")
    e = Variable(m,"e",domain = i, description = "quantity of exports")
    n = Variable(m,"n",domain = i, description = "quantity of imports")
    
    mb = Equation(m, "mb", domain = i, description = "material balance")
    
    mb[i] = x[i] >= y[i] + (e[i] - n[i]).where[t[i]]


Note that in the equation definition in the last line, the term ``(e[i] - m[i])`` on the 
right-hand side of the equation is added only for those elements of the set ``i`` that also 
belong to the subset ``t[i]``, so that the element services is excluded.

Further, conditional indexed operations may also feature in expressions in equation definitions. 
In the following example, note that the set ``i`` contains the supply regions, the set ``j`` 
contains the demand regions, and the two-dimensional set ``ij`` is the set of feasible links; 
the variable ``x`` denotes the shipment of natural gas and the variable ``s`` denotes the 
regional supply. ::

    sb[i] = Sum(j.where[ij[i,j]), x[i,j])  <=  s[i]

Similar to the assignment example seen before, the conditional set ``ij[i,j]`` restricts the 
domain of the summation: for each supply region ``i`` the summation over the demand regions 
``j`` is restricted to the label combinations ``(i,j)`` that are elements of the set of 
feasible links ``ij[i,j]``.

Control over the Domain of Definition
--------------------------------------

In case constraints should only be included in the model if particular conditions are met, 
a ``where`` condition in the domain of definition of an equation may be used to model this 
restriction. Such a ``where`` condition is analogous to the 
`where[] control on the left <where-on-the-left>`_ of assignments. Assuming that "the left" 
means the left of the ``'='`` then the analogy is even closer.

.. note::
    The ``where`` control over the domain of definition of equations restricts the number 
    of constraints generated to less than the number implied by the domain of the defining sets.

Consider the following example: ::

    gple[w,wp,te].where[ple[w,wp]] = yw[w,te] - yw[wp,te] <= dpack

Here ``w``, ``wp`` and ``te`` are sets, ``ple`` is a two-dimensional parameter, ``yw`` is a 
variable and ``dpack`` is a scalar. Note that the ``where`` condition restricts the first 
two indices of the domain of the equation to those label combinations that have non-zero entries 
in the two-dimensional parameter ``ple``.

Sometimes the desired restriction of an equation may be achieved either way: through a condition 
in the algebra or a condition in the domain of definition. Compare the following two lines, where 
``eq1`` and ``eq2`` are equations, ``i`` and ``j`` are sets, ``b`` is a ``scalar``, ``s`` is a 
parameter and ``x`` is a two-dimensional variable. ::

    eq1[i].where[b] = Sum(j, x[i,j])          >= -s[i]
    eq2[i]          = Sum(j, x[i,j]).where[b] >= -s[i].where[b]

In the first line the ``where`` condition is in the domain of definition, in the second line 
the ``where`` conditions are in the algebraic formulation of the equation. If ``b`` is non-zero, 
the generated equations ``eq1`` and ``eq2`` will be identical. However, if ``b`` is 0, no equation 
``eq1`` will be generated, but for each ``i`` we will see a trivial equation ``eq2`` of the form 
``0 >= 0``.

Note that if the logical condition in the domain of definition of an equation refers to set 
membership, then under certain conditions the restriction may be expressed without the use of 
the ``where`` operator. For details, see section 
`Filtering the Domain of Definition <filtering-the-domain-of-definition>`_ below.


Filtering Sets
================

If the logical condition refers to set membership, the restriction modeled with a ``where`` 
condition may sometimes be achieved without the ``where`` operator. Consider the following 
statement, where ``i`` and ``j[i]`` are sets, and ``u`` and ``s`` are parameters: ::

    u[i].where[j[i]] = s[i]

Note that the assignment is made only for those elements of the set ``i`` that are also 
elements of the subset ``j``. This conditional assignment may be rewritten in a shorter way: ::

    u[j] = s[j]

In this statement the assignment has been filtered through the condition without the ``where`` 
operator by using the subset ``j`` as the domain for the parameters ``u`` and ``s``. This 
formulation is cleaner and easier to understand. It is particularly useful in the context of 
multi-dimensional sets (tuples), and it may be used in 
`assignments <filtering-sets-in-assignments>`_, 
`indexed operations <filtering-controlling-indices-in-indexed-operations>`_ and the 
`domain of definition <filtering-the-domain-of-definition>`_ of equations.


.. _filtering-sets-in-assignments:

Filtering Sets in Assignments
------------------------------

Suppose we want to compute the transportation cost between local collection sites and regional 
transportation hubs for a fictional parcel delivery service. We define sets for the collection 
sites and transportation hubs and a two-dimensional set where the collection sites are matched 
with their respective hubs: ::

    from gamspy import Container, Set, Parameter, Variable, Equation, Sum
    import pandas as pd
    m = Container()
    
    i =  Set(m, "i", description = "local collection sites")
    j =  Set(m, "j", description = "regional transportation hubs")
    
    c = pd.Series(
        index=pd.MultiIndex.from_tuples([("boston", "newyork"),
                                        ("miami", "atlanta"),
                                        ("houston", "atlanta"),
                                        ("chicago", "detroit"),
                                        ("phoenix", "losangeles")])
    )
    
    r =  Set(m, "r", domain = [i,j], 
             uels_on_axes=True, 
             domain_forwarding = True,
             description = "regional transportation hub for each local collection site",
             records = c)
    
    dist = pd.DataFrame(
        [("miami", "newyork", 1327),
         ("miami", "detroit", 1387),
         ("miami", "losangeles", 2737),
         ("miami", "atlanta", 665),
         ("boston", "newyork", 216),
         ("boston", "detroit", 699),
         ("boston", "losangeles", 3052),
         ("boston", "atlanta", 1068),
         ("chicago", "newyork", 843),
         ("chicago", "detroit", 275),
         ("chicago", "losangeles", 2095),
         ("chicago", "atlanta", 695),
         ("houston", "newyork", 1636),
         ("houston", "detroit", 1337),
         ("houston", "losangeles", 1553),
         ("houston", "atlanta", 814),
         ("phoenix", "newyork", 2459),
         ("phoenix", "detroit", 1977),
         ("phoenix", "losangeles", 398),
         ("phoenix", "atlanta", 1810)],
        columns=["i", "j", "distance in miles"],
    )
    
    distance = Parameter(m, "distance", domain = [i,j], 
                         description = "distance in miles",
                        records = dist)
    
    shipcost = Parameter(m, "shipcost", domain = [i,j], 
                         description = "cost of transporting parcels from a local collection site to a regional hub per unit")
    
    factor = 0.009
    
    shipcost[i,j].where[r[i,j]] = factor*distance[i,j]


::

    In [1]: shipcost.records
    Out[1]:
              i	         j	value
    0	 boston	   newyork	1.944
    1	  miami	   atlanta	5.985
    2	houston	   atlanta	7.326
    3	chicago	   detroit	2.475
    4	phoenix	losangeles	3.582

The distance between collection sites and transportation hubs is given in the parameter ``distance``. 
The last line is a conditional assignment for the parameter ``shipcost``. This assignment is only 
made if the label combination ``(i,j)`` is an element of the set ``r``. Note that in each instance 
the indices ``i`` and ``j`` appear together. Thus the assignment may be simply written as: ::

    shipcost[r] = factor*distance[r]

Note that the assignment is explicitly restricted to the members of the set ``r``; the ``where`` 
operator is not necessary. Observe that if the indices ``i`` or ``j`` appear separately in any 
assignment, the above simplification cannot be made. For example, consider the case where the 
shipping cost depends not only on the ``factor`` and the ``distance`` between collection sites 
and regional hubs, but also on the congestion at the regional hub. We introduce a new parameter 
``congestfac`` that models the congestion at each regional hub and is indexed only over the set 
``j``: ::

    congestfac = Parameter(m, "congestfac", domain = j, 
                           description = "congestion factor",
                           records = [["newyork", 1.5],
                                     ["detroit", 0.7],
                                     ["losangeles", 1.2],
                                     ["atlanta", 0.9]])

The new cost of shipment is computed as follows: ::

    shipcost[i,j].where[r[i,j]] = factor * congestfac[j] * distance[i,j]

Note that this conditional assignment *cannot* be reformulated as: ::

    shipcost[r] = factor * congestfac[j] * distance[r]

In the representation above the index ``j`` appears on the right-hand side, but not on the left-hand 
side. GAMSPy will flag this assignment as an error. However, the following representation will work: ::

    shipcost[r[i,j]] = factor * congestfac[j] * distance[r]

In this formulation the set ``r`` is explicitly denoted as a tuple of the sets ``i`` and ``j``. The 
set ``j`` may then appear on the right-hand side.


.. _filtering-controlling-indices-in-indexed-operations:

Filtering Controlling Indices in Indexed Operations
----------------------------------------------------

Similarly, the controlling indices in indexed operations may be filtered through the conditional set 
without the use of the ``where`` operator. We continue with the shipping cost example from the last 
subsection. The total cost of shipment is obtained through the equation that follows. We also include 
the variable definitions for clarity. ::

    shipped = Variable(m, "shipped", domain = [i,j])
    totcost = Variable(m, "totcost")
    costequ = Equation(m, "costequ")
    
    costequ = totcost == Sum(Domain(i,j).where[r[i,j]], shipcost[i,j]*shipped[i,j])

Here the variable ``shipped`` is the number of parcels shipped from the local collection site ``i`` to 
the regional transportation hub ``j``, and the variable ``totcost`` is the total cost of all shipments. 
Note that she summation in the equation is restricted to the label combinations that are elements of the 
set ``r``. Alternatively, the equation above may be written as: ::

    costequ = totcost == Sum(r, shipcost[r]*shipped[r])

In this formulation the summation is performed explicitly only over the elements of the set ``r``, no 
``where`` condition is necessary. However, if the expression in the equation included a term dependent 
only on index ``j``, then we would have to reformulate differently. Suppose the equation included also 
the congestion factor ``congestfac`` that is indexed only over ``j``: ::

    costequ = totcost == Sum(Domain(i,j).where[r[i,j]], factor*congestfac[j]*distance[i,j]*shipped[i,j])

In this case the equation needs to be simplified in the following way: ::

    costequ = totcost == Sum(r[i,j], factor*congestfac[j]*distance[r]*shipped[r])

Like before, the domain of the indexed operation ``Sum`` is the set ``r``. But this time the domain of 
``r`` has to be named explicitly, so that the parameter ``congestfac`` which is indexed only over the 
set ``j`` is permitted in the scope of the indexed operation. Note that this reasoning is analogous 
to the reasoning for filtering sets in assignments in the subsection above.

.. _filtering-the-domain-of-definition:

Filtering the Domain of Definition
----------------------------------------------------

The rules for filtering sets that we have introduced in subsections 
`Filtering Sets in Assignments <filtering-sets-in-assignments>`_ and 
`Filtering Controlling Indices in Indexed Operations <filtering-controlling-indices-in-indexed-operations>`_  
also apply in the context of equation domains. We continue with the parcel transport example introduced 
above and add a :meth:`binary variable <binary-variables>` ``bin``, the parameter ``bigM`` and the 
equation ``connect`` to the model. Recall that ``shipped[i,j]`` is a variable and ``r[i,j]`` is a set. ::

    bigM = Parameter(m, "bigM",domain = [i,j])
    bin = Variable(m, "bin", domain = [i,j], type = "binary")
    
    connect = Equation(m, "connect", domain = [i,j])
    
    connect[i,j].where[r[i,j]] = shipped[i,j] <= bigM[i,j]*bin[i,j]

The ``where`` condition restricts the domain of definition of the equation ``connect`` to those label 
combinations of the sets ``i`` and ``j`` that are elements of the set ``r``. The equation relates the 
continuous variable ``shipped[i,j]`` to the binary variable ``bin[i,j]``. Note that each domain in the 
quation is the index pair ``(i,j)``. So the equation may be simplified as follows: ::

    connect[r] = shipped[r] <= bigM[r]*bin[r]

In this formulation the domain of the equation is explicitly restricted to the members of the set ``r``, 
without the use of a ``where`` condition. Note that if the right-hand side of the equation contained 
any term that was indexed over ``i`` or ``j`` separately, then the domain of definition of the equation 
would have to be simplified as: ::

    connect[r[i,j]]

The reasoning is the same as in the case of assignments and indexed operations.