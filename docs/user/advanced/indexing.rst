.. _indexing:

.. meta::
   :description: Documentation of different types of indexing in GAMSPy
   :keywords: Indexing, GAMSPy, gamspy, GAMS, gams, mathematical modeling, sparsity, performance

********
Indexing
********

Set-based Indexing and Literal Indexing
=======================================

Set-based indexing is at the core of the GAMSPy and GAMS execution system. They are concise, easy to read and have great performance.
Therefore, we encourage the use of it in most contexts. Yet, in certain cases, one might be inclined to do literal indexing. Because of that 
GAMSPy also allows literal indexing.

.. tab-set-code::

    .. code-block:: Set-based

        m = gp.Container()
        i = gp.Set(m, records=['i1', 'i2'])
        a = gp.Parameter(m, "a", domain=i, records=[('i1', 1), ('i2', 2)])

        a[i] = 5 # set-based indexing that sets all records of a to 5
        print(a.records)

    .. code-block:: Literal

        import gamspy as gp
        
        m = gp.Container()
        i = gp.Set(m, records=['i1', 'i2'])
        a = gp.Parameter(m, "a", domain=i, records=[('i1', 1), ('i2', 2)])
        a['i1'] = 5 # literal indexing that sets element 'i1' to 5 
        a['i2'] = 6 # literal indexing that sets element 'i1' to 6

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
    d = gp.Parameter(m, name="d", domain=[i, j])
    c = gp.Parameter(m, name="c", domain=[i, j])
    c[:, :] = 90 * d[:, :] / 1000

Each `:` refers to the corresponding domain item in that index for the symbol. In this example, the first `:` is `i` and the second `:` is `j`. 
Hence, it is equivalent to: ::

    import gamspy as gp
    m = gp.Container()
    i = gp.Set(m, name="i")
    j = gp.Set(m, name="j")
    d = gp.Parameter(m, name="d", domain=[i, j])
    c = gp.Parameter(m, name="c", domain=[i, j])
    c[i, j] = 90 * d[i, j] / 1000

Usage of Ellipsis
-----------------

Here is an example that shows how ellipsis can be used: ::

    import gamspy as gp
    m = gp.Container()
    i = gp.Set(m, name="i")
    j = gp.Set(m, name="j")
    d = gp.Parameter(m, name="d", domain=[i, j])
    c = gp.Parameter(m, name="c", domain=[i, j])
    c[...] = 90 * d[...] / 1000

This is also equivalent to: ::

    import gamspy as gp
    m = gp.Container()
    i = gp.Set(m, name="i")
    j = gp.Set(m, name="j")
    d = gp.Parameter(m, name="d", domain=[i, j])
    c = gp.Parameter(m, name="c", domain=[i, j])
    c[i, j] = 90 * d[i ,j] / 1000


For scalar symbols (symbols with no domain), slice and ellipsis means the same thing: ::

    import gamspy as gp
    m = gp.Container()
    c = gp.Parameter(m, name="c")
    c[...] = 90
    # or
    c[:] = 90

