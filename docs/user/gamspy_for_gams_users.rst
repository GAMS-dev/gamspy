.. _gamspyforgamsusers:

*********************
GAMSPy for GAMS Users
*********************

This document is for GAMS users who are interested in translating their
existing GAMS models to GAMSPy or vice versa. 

Translating Symbols
-------------------

The way to create symbols such as Set, Alias, Parameter, Variable, and Equation is explained 
in their respective documentation pages and you can see an example for the creation of each
symbol below: 


.. tab-set-code:: 

    .. code-block:: python
        :name: Python

        import gamspy as gp

        m = gp.Container()

        i = gp.Set(m, 'i', records=['i1','i2'])
        a = gp.Alias(m, 'a', alias_with=i)
        p = gp.Parameter(m, 'p', domain=i, records=[['i1','1'], ['i2','2']])
        v = gp.Variable(m, 'v', domain=i)
        z = gp.Variable(m, 'z')
        e = gp.Equation(m, 'e', domain=i)
        e[i] = v[i] + p[i] <= z
        my_model = gp.Model(m, equations=[e], problem="lp", sense="min", objective=z)
        my_model.solve()

    .. code-block:: text
        :name: GAMS

        Set i / i1, i2 /;
        Alias (i, a);
        Parameter p(i) / i1 1, i2 2 /;
        Variable v(i), z;
        Equation e(i);
        e(i) .. v(i) + p(i) =l= z
        Model my_model / e /;
        solve my_model using LP min z;

Translating Operations: Sum/Product/Smin/Smax
---------------------------------------------

Frequently used GAMS operations which accept an index list and an expression can be translated as follows:

.. tab-set-code::
    
    .. code-block:: python
        :name: Python

        import gamspy as gp
        
        m = gp.Container()
        i = gp.Set(m, 'i', records=['i1','i2'])
        a = gp.Parameter(m, 'a', domain=i, records=[['i1','1'], ['i2','2']])
        z = gp.Variable(m, 'z')

        eq = gp.Equation(m, name="eq")
        eq[...] = gp.Sum(i, a[i]) <= z

    .. code-block:: text
        :name: GAMS

        Set i / i1, i2 /;
        Parameter a(i) / i1 1, i2 2 /;
        Variable z;
        Equation eq;
        eq .. sum(i, a(i)) =l= z;

Card/Ord
--------

Card and Ord operations can be translated as follows:

.. tab-set-code::

    .. code-block:: python
        :name: Python

        import gamspy as gp
        import math

        m = gp.Container()
        i = gp.Set(m, name='i', records=[f'i{i}' for i in range(181)])
        step = gp.Parameter(m, name="step", records=math.pi / 180)
        omega = gp.Parameter(m, name="omega", domain=i)
        omega[i] = (gp.Ord(i) - 1) * step

    .. code-block:: text
        :name: GAMS
        
        Set i / i0*i180 /;
        Parameter step;
        step = pi / 180;
        Parameter omega(i);
        omega(i) = (ord(i) - 1) * step;

Domain
------

This class is exclusively for conditioning on a domain with more than one set.

.. tab-set-code::

    .. code-block:: python
        :name: Python
        
        import gamspy as gp

        m = gp.Container()
        bus = gp.Set(m, name="bus", records=[f"i{b}" for b in range(1, 7)])
        node = gp.Alias(m, name="node", alias_with=bus)
        conex = gp.Set(m, name="conex", domain=[bus, bus])

        branch = gp.Parameter(m,"branch", [bus, node, "*"] ,records=...)

        p = gp.Parameter(m, name="p")
        
        conex[bus, node].where[branch[bus, node, "x"]] = True
        conex[bus, node].where[conex[node, bus]] = True

        p[...] = gp.Smax(
            gp.Domain(bus, node).where[conex[bus, node]],
            branch[bus, node, "bij"] * 3.14 * 2,
        )

    .. code-block:: text
        :name: GAMS
        
        Set bus / i1*i6 /;
        Alias (bus, node);
        Set conex(bus, bus);
        
        Parameter branch(bus, node, "*") / ...... /;
        Parameter p;

        conex(bus, node)$(branch(bus, node, "x")) = yes;
        conex(bus, node)$(conex(node, bus)) = yes;

        p = smax((bus, node) $ (conex(bus, node)), branch(bus, node, "bij") * 3.14 * 2)

Number
------

This is for conditions on numbers or yes/no statements.

.. tab-set-code::

    .. code-block:: python
        :name: Python
        
        import gamspy as gp

        m = gp.Container()
        i = gp.Set(m, "i", records=range(1,5))
        ie = gp.Set(m, "ie", domain=i)
        x = gp.Variable(m, "x", domain=i)
        ie[i] = gp.Number(1).where[x.lo[i] == x.up[i]]

    .. code-block:: text
        :name: GAMS
    
        Set i / 1*4 /;
        Set ie(i);
        Variable x(i);
        ie(i) = yes$(x.lo(i) = x.up(i));

math package
------------

This package is for the mathematical operations of GAMS.

.. tab-set-code::

    .. code-block:: python
        :name: Python

        import gamspy as gp
        import gamspy.math as gams_math
        import random

        m = gp.Container()
        i = gp.Set(m, "i", records=['i1', 'i2'])
        k = gp.Set(m, "k", records=['k1', 'k2'])
        sigma = gp.Variable(m, name="sigma", domain=[i, k], type="Positive")
        sigma.l[i, k] = gams_math.uniform(0.1, 1) # Generates a different value from uniform distribution for each element of the domain.
        print(sigma.records)
        sigma.l[i, k] = random.uniform(0.1, 1) # This is not equivalent to the statement above. This generates only one value for the whole domain.
        print(sigma.records)

    .. code-block:: text
        :name: GAMS

        Set i / i1, i2 /;
        Set k / k1, k2 /;
        Positive Variable sigma(i,k);
        sigma.l(i,k) = uniform(0.1, 1);
        display sigma.l;


Logical Operations
------------------

Since it is not possible in Python to overload keywords such as **and**, **or**, and **not**, you need to use bitwise operatiors **&**, **|**, and **~**.

Mapping:

- **and** -> &
- **or**  -> |
- **not** -> ~

.. tab-set-code::

    .. code-block:: python
        :name: Python

        error01[s1, s2] = rt[s1, s2] & (~lfr[s1, s2]) | ((~rt[s1, s2]) & lfr[s1, s2])

    .. code-block:: text
        :name: GAMS
    
        error01(s1,s2) = rt(s1,s2) and not lfr(s1,s2) or not rt(s1,s2) and lfr(s1,s2);


Translating GAMS Macros
-----------------------

`Macros in GAMS <https://www.gams.com/latest/docs/UG_DollarControlOptions.html#UG_DollarControl_MacrosInGAMS>`_ can be translated to GAMSPy as functions.
The following example shows how GAMS Macro `reciprocal` can be defined as a function in Python to be used in GAMSPy:

.. tab-set-code::

    .. code-block:: python
        :name: Python

        import gamspy as gp

        def reciprocal(y):
            return 1 / y

        m = gp.Container()
        z = gp.Parameter(m, "z")
        x1 = gp.Parameter(m, "x1", records=2)
        x2 = gp.Parameter(m, "x2", records=3)
        z[:] = reciprocal(x1) + reciprocal(x2)
        print(z.records)

    .. code-block:: text
        :name: GAMS

        $macro reciprocal(y) 1/y

        scalar z, x1 /2/, x2 /3/;

        z = reciprocal(x1) + reciprocal(x2);
        display z;

Automatic Conversion of a GAMSPy Model to GAMS
----------------------------------------------

Existing GAMSPy models can be translated to a GAMS model automatically by using :meth:`Model.toGams <gamspy.Model.toGams>`: ::

    import gamspy as gp

    m = gp.Container()
    i = gp.Set(m, "i", records=["i1", "i2"])
    a = gp.Parameter(m, "a", domain=i)
    ...
    ...
    your_model = gp.Model(m, "your_model", equations=...)
    ...
    ...

    your_model.toGams(path=<gams_model_path>)

The generated GAMS model can be found under ``<gams_model_path>/<model_name>.gms``. 
If you also want to generate a pf file that contains the necessary option, you 
can provide it with `options` argument. If you want to dump the GAMS state, you can 
also set `dump_gams_state` to True.
