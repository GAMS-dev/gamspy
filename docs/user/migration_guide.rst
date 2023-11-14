.. _migrationguide:

*********************
Model Migration Guide
*********************

This document is for users who are interested in migrating their
existing GAMS model to GAMSPy. 

Migrating Symbols
-----------------

You can create a Set, Alias, Parameter, Variable, and Equation in the same way you create a Gams Transfer symbol. 
You don't need to change anything. Now, these symbols can be used in creating expressions.

GAMSPy:

.. code-block:: python

    import gamspy as gp

    m = gp.Container()
    i = gp.Set(m, "i", records=['i1','i2'])
    a = gp.Alias(m, "a", alias_with=i)
    p = gp.Parameter(m, 'p', domain=[i], records=[['i1','1'], ['i2','2']])
    v = gp.Variable(m, "v", domain=[i])
    z = gp.Variable(m, "z")
    e = gp.Equation(m, "e", domain=[i])
    e[i] = v[i] + p[i] <= z
    model = gp.Model(m, "my_model", equations=[e], problem="lp", sense="min", objective=z)
    model.solve()

GAMS:

.. code-block:: gams
    
    Set i / i1, i2 /;
    Alias (i, a);
    Parameter p / i1 1, i2 2 /;
    Variable v(i);
    Equation e(i);
    e(i) .. v(i) + p(i) =l= z
    Model my_model / e /;
    solve my_model using LP min z;


Migrating Operations: Sum/Product/Smin/Smax
-------------------------------------------

Frequently used GAMS operations which accept an index list and an expression can be migrated as follows.

GAMSPy:

.. code-block:: python

    from gamspy import Sum, Product, Smin, Smax
    
    m = gp.Container()
    i = gp.Set(m, "i", records=['i1','i2'])
    a = gp.Parameter(m, 'a', domain=[i], records=[['i1','1'], ['i2','2']])
    z = gp.Variable(m, 'z')

    eq = gp.Equation(m, name="eq")
    eq[...] = Sum(i, a[i]) <= z

GAMS:

.. code-block:: gams

    Set i / i1, i2 /;
    Parameter a / i1 1, i2 2 /;
    Variable z;
    Equation eq;
    eq .. sum(i, a(i)) =l= z;

Card/Ord
--------

Card and Ord operations can be migrated as follow:

GAMSPy:

.. code-block:: python

    import gamspy as gp
    import math

    m = gp.Container()
    i = Set(m, name="i", records=[str(idx) for idx in range(0, 181)])
    step = Parameter(m, name="step", records=math.pi / 180)
    omega = Parameter(m, name="omega", domain=[i])
    omega[i] = (Ord(i) - 1) * step

GAMS:

.. code-block:: gams
    
    Set i / i0..i180 /;
    Parameter step;
    step = pi / 180;
    Parameter omega(i);
    omega(i) = (Ord(i) - 1) * step;

Domain
------

This class is exclusively for conditioning on a domain with more than one set.

GAMSPy:

.. code-block:: python
    
    import gamspy as gp

    m = gp.Container()

    bus = gp.Set(m, "bus", records=["i" + str(buses) for buses in range(1, 7)])
    node = Alias(m, name="node", alias_with=bus)
    conex = Set(m,"conex",domain=[bus, bus])

    branch = Parameter(m,"branch",[bus, node, "*"],records=records)

    p = Parameter(m, name="M")
    
    conex[bus, node].where[branch[bus, node, "x"]] = True
    conex[bus, node].where[conex[node, bus]] = True

    p[...] = Smax(
        Domain(bus, node).where[conex[bus, node]],
        branch[bus, node, "bij"] * 3.14 * 2,
    )

GAMS:

.. code-block:: gams
    
    Set bus / i1..i6 /;
    Alias (bus, node);
    Set conex(bus, bus);
    
    Parameter branch(bus, node, "*") / ...... /;
    Parameter p;

    conex(bus, node)$(branch(bus, node, "x")) = yes;
    conex(bus, node)$(conex(node, bus)) = yes;

    p = smax((bus, node) $ (conex(bus, node)), branch(bus, node, "bij" * 3.14 * 2))

Number
------

This is for conditions on numbers or yes/no statements.

GAMSPy:

.. code-block:: python
    
    import gamspy as gp

    m = gp.Container()
    i = gp.Set(m, "i", records=[str(i) for i in range(1,5)])
    ie = gp.Set(m, "ie", domain=[i])
    x = gp.Variable(m, "x", domain=[i])
    ie[i] = gp.Number(1).where[x.lo[i] == x.up[i]]

GAMS:

.. code-block:: gams
    
    Set i / 1..4 /;
    Set ie(i);
    Variable x(i);
    ie(i) = yes$(x.lo(i) = x.up(i));

math package
------------

This package is for the mathematical operations of GAMS.

GAMSPy:

.. code-block:: python

    import gamspy.math as gams_math
    import math

    sigma = Variable(m, name="sigma", domain=[i, k], type="Positive")
    sigma.l[i, k] = uniform(0.1, 1) # Generates a different value from uniform distribution for each element of the domain.
    sigma.l[i, k] = math.uniform(0.1, 1) # This is not equivalent to the statement above. This generates only one value for the whole domain.

Logical Operations
------------------

Since it is not possible in Python to overload keywords such as **and**, **or**, and **not**, you need to use bitwise operatiors **&**, **|**, and **~**.

Mapping:

- **and** -> &
- **or**  -> |
- **not** -> ~

GAMSPy:

.. code-block:: python

    error01[s1,s2] = rt[s1,s2] & (~lfr[s1,s2]) | ((~rt[s1,s2]) & lfr[s1,s2])

GAMS:

.. code-block:: gams
    
    error01(s1,s2) = rt(s1,s2) and not lfr(s1,s2) or not rt(s1,s2) and lfr(s1,s2);
