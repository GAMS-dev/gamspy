.. _indexing:

.. meta::
   :description: Documentation of GAMSPy expressions and different types of indexing.
   :keywords: Expression, indexing, GAMSPy, gamspy, GAMS, gams, mathematical modeling, sparsity, performance

***********************
Expression and Indexing
***********************

Set-based Indexing and Literal Indexing
=======================================

Set-based indexing is at the core of the GAMSPy and GAMS execution system. They are concise, easy to read and have great performance.
Therefore, we encourage the use of it in most contexts. Yet, in certain cases, one might be inclined to do literal indexing (with a str or an int). 
Because of that GAMSPy also allows literal indexing.

.. tabs::

    .. tab:: Set-based

        .. code-block:: python
            
            import gamspy as gp

            m = gp.Container()
            i = gp.Set(m, records=['i1', 'i2'])
            a = gp.Parameter(m, domain=i, records=[('i1', 1), ('i2', 2)])
            a[i] = 5 # set-based indexing that sets all records of a to 5

    .. tab:: Literal

        .. code-block:: python

            import gamspy as gp
            
            m = gp.Container()

            i = gp.Set(m, records=['i1', 'i2'])
            a = gp.Parameter(m, "a", domain=i, records=[('i1', 1), ('i2', 2)])
            a['i1'] = 5 # literal indexing with a string that sets element 'i1' to 5 
            a['i2'] = 6 # literal indexing with a string that sets element 'i2' to 6

            j = gp.Set(m, records=range(5))
            j[1] = False # literal indexing with an integer that removes element 1 from the set.

Slices and Ellipsis
===================

GAMSPy supports NumPy-like indexing including the usage of `Slices <https://docs.python.org/3/library/functions.html?highlight=slice#slice>`_ 
and `Ellipsis <https://docs.python.org/3/library/constants.html#Ellipsis>`_. An ellipsis expands to the number of colon (`:`) objects needed for the 
selection tuple to index all dimensions. There may only be a single ellipsis present. 

Usage of Slices
---------------

Here is an example that shows how slices can be used: ::

    import gamspy as gp

    m = gp.Container()
    i = gp.Set(m, name="i")
    j = gp.Set(m, name="j")
    d = gp.Parameter(m, domain=[i, j])
    c = gp.Parameter(m, domain=[i, j])
    c[:, :] = 90 * d[:, :] / 1000

Each `:` refers to the corresponding domain item in that index for the symbol. In this example, the first `:` is `i` and the second `:` is `j`. 
Hence, it is equivalent to: ::

    import gamspy as gp
    m = gp.Container()
    i = gp.Set(m, name="i")
    j = gp.Set(m, name="j")
    d = gp.Parameter(m, domain=[i, j])
    c = gp.Parameter(m, domain=[i, j])
    c[i, j] = 90 * d[i, j] / 1000

Usage of Ellipsis
-----------------

Here is an example that shows how ellipsis can be used: ::

    import gamspy as gp

    m = gp.Container()
    i = gp.Set(m, name="i")
    j = gp.Set(m, name="j")
    d = gp.Parameter(m, domain=[i, j])
    c = gp.Parameter(m, domain=[i, j])
    c[...] = 90 * d[...] / 1000

This is also equivalent to: ::

    import gamspy as gp

    m = gp.Container()
    i = gp.Set(m, name="i")
    j = gp.Set(m, name="j")
    d = gp.Parameter(m, domain=[i, j])
    c = gp.Parameter(m, domain=[i, j])
    c[i, j] = 90 * d[i ,j] / 1000


For scalar symbols (symbols with no domain), slice and ellipsis means the same thing: ::

    import gamspy as gp
    
    m = gp.Container()
    c = gp.Parameter(m)
    c[...] = 90
    # or
    c[:] = 90

.. _gp_expression:

Expression
==========

GAMSPy lazily executes :meth:`expressions <gamspy.Expression>` to improve performance. Therefore, whenever 
you express an operation on GAMSPy symbols (addition, multiplication etc.), GAMSPy generates an expression 
instead of executing it right away. For example: ::

    import gamspy as gp

    m = gp.Container()
    a = gp.Parameter(m, "a")
    b = gp.Parameter(m, "b")
    print(a + b)

would print an expression: ::

    Expression(left=Parameter(name='a', domain=[]), data=+, right=Parameter(name='b', domain=[]))

As you can see in the output, each expression has a left operand, right operand and an operator. In this 
example, left and right operands are :meth:`parameters <gamspy.Parameter>` and the operator is an addition operator. 
If one wants to evaluate the result of the expression, they can directly call :meth:`.records <gamspy.Expression.records>` on it. ::

    import gamspy as gp

    m = gp.Container()
    a = gp.Parameter(m, "a", records=5)
    b = gp.Parameter(m, "b", records=10)
    print((a + b).records)

This would return: ::

       value
    0   15.0

For scalar expressions such as the one above, one can also call :meth:`.toValue <gamspy.Expression.toValue>` to get the value directly.
instead of getting a DataFrame as a result. ::

    import gamspy as gp

    m = gp.Container()
    a = gp.Parameter(m, "a", records=5)
    b = gp.Parameter(m, "b", records=10)
    print((a + b).toValue())

This would return 15 as a float directly.

For indexed expressions, one can call :meth:`.toList <gamspy.Expression.toList>` to get the result as a list of values. ::

    import numpy as np
    import gamspy as gp

    m = gp.Container()
    i = gp.Set(m, "i", records=range(2))
    a = gp.Parameter(m, "a", domain=i, records=np.array([3, 5]))
    b = gp.Parameter(m, "b", domain=i, records=np.array([2, 4]))
    print((a + b).toList())

This would return the values of the expression as a list as follows: ::

    [['0', 5.0], ['1', 9.0]]

Chained Expressions
-------------------

Expressions can be arbitrarily long depending on your definition/assignment statement. For example: ::

    import gamspy as gp

    m = gp.Container()
    t = gp.Set(m, "t", records=range(3))
    price = gp.Parameter(m, "price", domain=t, records=np.array([1, 2, 3]))
    buy = gp.Parameter(m, "buy", domain=t, records=np.array([1, 2, 3]))
    sell = gp.Parameter(m, "sell", domain=t, records=np.array([2, 3, 4]))
    stock = gp.Parameter(m, "stock", domain=t, records=np.array([4, 1, 5]))
    storecost = gp.Parameter(m, "storecost", records=5)
    result = gp.Parameter(m, "result", domain=t)
    expr = price[t] * (buy[t] - sell[t]) + storecost * stock[t]
    print(expr)

This would result in four nested expressions. Instead of executing each executing eagerly, GAMSPy prepares 
the expression tree until it's needed. For example, if we you need to see the result of the expression, then 
GAMSPy lets GAMS to run the expression and return the result. ::

    print(expr.records)

The output would look like as follows: ::

       t  value
    0  0   19.0
    1  1    3.0
    2  2   22.0