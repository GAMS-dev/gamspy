.. _number:

******
Number
******

A :meth:`gamspy.Number` object is needed mainly for two cases. 

The first case is to assign records to a symbol conditionally. For example:

.. code-block:: python

    import gamspy as gp
    
    m = gp.Container()
    k = gp.Set(m, "k", records=["1964-i","1964-ii","1964-iii","1964-iv"])
    ki = gp.Set(m, domain=k, description="initial period")
    ki[k] = gp.Number(1).where[gp.Ord(k) == 1]

The code snippet above would assign only `1964-i` to `ki` since only the order of `1964-i` is equal to 1.

The second case is the modeling of constant right-hand side or left-hand side. Since Python does not provide 
magic functions for `__req__`, `__rle__` and `__rge__` similar to `__radd__`, it is impossible for GAMSPy 
to distinguish, for example, `0 <= expression` and `expression >= 0`. For example: ::

    import gamspy as gp

    gp.set_options({"ALLOW_AMBIGUOUS_EQUATIONS": "yes"})

    m = gp.Container()
    c = gp.Parameter(m, name="c")
    x = gp.Variable(m, name="x", type="Negative")
    e = gp.Equation(m, name="e")
    e[...] = (x - c) >= 0
    print(e.latexRepr())

    f = gp.Equation(m, name="f")
    f[...] = 0 <= (x - c)
    print(f.latexRepr())
    
    print(e.latexRepr() == f.latexRepr())

Because of the aformentioned limitation of Python, the equation definition of `e` and `f` are the same: ::

    x - c \geq 0
    x - c \geq 0
    True

Due to this you might get marginals with a *wrong* sign. This implication is especially important for EMP,
MCP and MPEC models since the GAMSPy execution system checks appropriate bounds for equation/variable pairs.
The option `ALLOW_AMBIGUOUS_EQUATIONS` controls how GAMSPy reacts to *ambiguous* equation definitions 
(when the :meth:`gamspy.Model.solve` method is called). The default for this option is `AUTO`. For EMP, MCP and MPEC models
`AUTO` acts like setting the option to `NO`, for other model types `AUTO` behaves like setting this option
to `YES`. Instead of figuring out the best option setting, a much better solution is to use :meth:`gamspy.Number`
to ensure that the order of the components are correct: ::

    import gamspy as gp

    m = gp.Container()
    c = gp.Parameter(m, name="c")
    x = gp.Variable(m, name="x", type="Negative")
    e = gp.Equation(m, name="e")
    e[...] = (x - c) >= gp.Number(0)
    print(e.latexRepr())

    f = gp.Equation(m, name="f")
    f[...] = gp.Number(0) <= (x - c)
    print(f.latexRepr())
    
    print(e.latexRepr() == f.latexRepr())

With the usage of :meth:`gamspy.Number`, the equation components are in order again: ::

    x - c \geq 0
    0 \leq x - c
    False

    