.. _indexing:

.. meta::
   :description: Documentation of different types of indexing in GAMSPy
   :keywords: Indexing, GAMSPy, gamspy, GAMS, gams, mathematical modeling, sparsity, performance

********
Indexing
********

GAMSPy supports Numpy-like indexing including the usage of `Slices <https://docs.python.org/3/library/functions.html?highlight=slice#slice>`_ 
and `Ellipsis <https://docs.python.org/3/library/constants.html#Ellipsis>`_. An ellipsis expands to the number of colon (`:`) objects needed for the 
selection tuple to index all dimensions. There may only be a single ellipsis present. 

Usage of Slice
==============

The equivalent representation of the example above with slices would be: ::

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
=================

The equivalent representation of the example above with ellipsis would be: ::

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

