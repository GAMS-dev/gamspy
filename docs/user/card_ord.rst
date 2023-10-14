.. _card_ord:

************
Card and Ord
************

The ``Card`` and ``Ord`` operators help to formulate position related
expressions on sets which hold labels that do not have a numerical
representation. Both operators ``Card`` and ``Ord``return integer values
when applied to sets. While the integer values returned do not represent
the numerical value of the label, they can be used for the same purpose.


Ord
==========

The ``Ord`` operator can be used with one-dimensional sets that are static and
ordered as well as corresponding aliases. It returns the relative position of
elements. The following example shows how ``Ord`` can be used::

    import gamspy as gp
    
    m = gp.Container()
    t = gp.Set(
        m,
        name="t",
        description="time periods",
        records=[str(x) for x in range(1985, 1996)],
    )
    val = gp.Parameter(m, name="val", domain=[t])
    val[t] = gp.Ord(t)

Note that as a result of the statements above, the value of ``val["1985"]`` will be
``1``, ``val["1986"]`` will be ``2`` and so on.

A common use of ``Ord`` is in setting up vectors that represent quantities
growing in some analytically specified way. For example, suppose a country has
56 million people in the base period and the population is growing at the rate
of 1.5 percent per year. Then the population in succeeding years can be
calculated as follows::

    population[t] = 56*(1.015**(gp.Ord(t) - 1))

It is often useful to simulate general matrix operations in GAMSPy. The first
index of a two dimensional parameter can conveniently represent the rows, the
second the columns and order is necessary. The example below shows how to set
the upper triangle of a matrix equal to the row index plus the column index,
and the diagonal and lower triangle to zero::

    import gamspy as gp
    
    m = gp.Container()
    
    i = gp.Set(
        m,
        name="i",
        description="row and column labels",
        records=[f"x{x+1}" for x in range(10)],
    )
    j = gp.Alias(m, name="j", alias_with=i)
    a = gp.Parameter(
        m, name="a", description="a general square matrix", domain=[i, j]
    )
    a[i, j].where[gp.Ord(i) < gp.Ord(j)] = gp.Ord(i) + gp.Ord(j)

Note that in the assignment statement the logical condition
``[gp.Ord(i) < gp.Ord(j)]`` restricts the assignment to the entries of the
upper triangle.


Card
==========

The ``Card`` operator takes any symbol as argument and returns its number of
records. It can be used in any expression like e.g. equation definitions and
parameter assignments. When used with in instance of type ``Model``, the number
of equation symbols contained in the model (plus 1 for the objective) is
returned. The following example shows how ``Card`` can be used to get the
number of records of a set::

    import gamspy as gp
    
    m = gp.Container()
    
    t = gp.Set(
       m,
       name="t",
       description="time periods",
       records=[str(x) for x in range(1985, 1996)],
    )
    s = gp.Parameter(m, name="s")
    s.assignment = gp.Card(t)

Note that ``s`` will be assigned the value 11 since the set ``t`` has 11 elements.

A common use case for the combination of both operators is to formulate a
condition that is only valid for the last element of a set. The following
example does fix a variable for the final period found in set ``t`` only::

    import gamspy as gp
    
    m = gp.Container()
    t = gp.Set(m, name="i", records=["t1", "t2", "t3"])
    c = gp.Variable(m, name="c", domain=[t])
    c.fx[t].where[gp.Ord(t) == gp.Card(t)] = 1

Note that the logical condition ``[gp.Ord(t) == gp.Card(t)]`` restricts the
assignment to the last element of the set ``t``: no assignment is made for
other elements of ``t``. The advantage of this way of fixing the variable ``c``
is that the membership of ``t`` can be changed safely and this statement will
always fix ``c`` for the last element.