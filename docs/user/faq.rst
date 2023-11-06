.. _examples:

****************************
Frequently Asked Questions
****************************

Why can't I redefine a GAMSPy symbol?
--------------------------------------

Trying to run the following lines of code will raise an error.

.. code-block:: 

    from gamspy import Container, Set, Parameter
    m = Container()
    p = Set(container=m, name="p", description="products")
    price = Parameter(container=m, name="p", domain=p, description="price for product p")

The problem with the above code is that the ``Set`` statement creates a symbol in the GAMSPy database
with name "p". Consequently, the namespace "p" is now exclusively reserved for a ``Set``. The following
``Parameter`` statement attempts to create a GAMSPy ``Parameter`` within the same namespace "p", which is 
already reserved for the ``Set`` ``p``. Thus, you want to keep in mind that the type for a GAMSPy symbol 
is fixed once it was declared. 


Why do I need a GAMSPy ``Alias``?
----------------------------------

Consider the following example code::

    from gamspy import Container, Set, Parameter
    m = Container()
    i = j = Set(container=m, name="i", records=range(3))
    p = Parameter(container=m, name="p", domain=[i, j])

    p[i, j] = 1

You would probably expect that the value for :math:`p_{i,j}` is equal to one for each combination of :math:`(i,j)`

::

    >>> p.records
                value
        i  j
        0  0   1
        0  1   1
        0  2   1
        1  0   1
        1  1   1
        1  2   1
        2  0   1
        2  1   1
        2  2   1

However, the above lines of code give you::

    >>> p.records
                  value
        i_0  i_1
        0    0    1
        1    1    1
        2    2    1

Only by declaring ``j`` an ``Alias`` of ``i`` you will get the desired outcome::

    from gamspy import Alias, Container, Set, Parameter
    m = Container()
    i = Set(container=m, name="i", records=range(3))
    j = Alias(container=m, name='j', alias_with=i)
    p = Parameter(container=m, name="p", domain=[i, j])

    p[i, j] = 1

::

    >>> p.records
            value
    i  j
    0  0   1
    1      1
    2      1
    1  0   1
    1      1
    2      1
    2  0   1
    1      1
    2      1


Do I use a ``Parameter`` or a Python variable to represent scalar parameters?
------------------------------------------------------------------------------

.. code-block::

    from gamspy import Container, Parameter
    m = Container()
    p_python = 40
    p_parameter = Parameter(container=m, name="p", records=40)


Technically it does not matter whether a scalar ``Parameter`` or a Python variable is used. 
It is more a matter of taste and convenience as::
    
    eq = Equation(container=m, name="eq", domain=i)
    eq[i] = Sum(j, x[i, j]) <= p_python

is equivalent to::

    eq = Equation(container=m, name="eq", domain=i)
    eq[i] = Sum(j, x[i, j]) <= p_parameter

