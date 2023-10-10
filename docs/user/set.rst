.. _set:

***
Set
***

Simple Sets
============

Introduction
-------------

Sets are the basic building blocks of a GAMSPy model, corresponding exactly 
to the indices in the algebraic representations of models. Example: ::

    m = Container()
    i = Set(m, name = "i", description = "canning plants", records = ["seattle", "san-diego"])
    j = Set(m, name = "j", description = "markets", records = ['new-york', 'chicago', ‘topeka’])

The effect of these statements is probably self-evident. first, we created a ``Container``, 
which encapsulate all relevant information for a GAMSPy Model. This Container acts as a 
centralized hub, gathering essential data, sets, parameters, variables, and constraints, 
providing a clear structure for our optimization problem.

We then declared two sets using the ``Set()`` class and gave them the names ``i`` and ``j``. 
We also assigned members to the sets as follows:

- :math:`i = \{Seattle, San Diego\}`
- :math:`j = \{New York, Chicago, Topeka\}`

They are labels, but are often referred to as elements or members. The optional ``description`` 
may be used to describe the set for future reference and to ease readability.

Besides using the ``Set()`` class directly, one can also facilitate the ``addSet()`` method 
of the ``Container`` class: ::

    i = m.addSet(name="i", description="canning plants", records = ["seattle", "san-diego"])

Set declaration and data assignment can also be done separately: ::
     
    m = Container()
    i = Set(m, "i", description="canning plants")
    i.setRecords(["seattle", "san-diego"])

Not only sets themselves, but also the individual elements can have a description, 
which is called *element text*: ::
     
    m = Container()
    i = Set(m, "i", records=[
                           ("seattle", "home of sub pop records"),
                           ("san-diego",),
                           ("washington_dc", "former gams hq"),
                       ],
    )
    
    In [1]: i.records
    Out[1]:
               uni               element_text
    0        seattle  home of sub pop records
    1      san-diego
    2  washington_dc           former gams hq

The order in which the set members are listed is usually not important. 
However, if the members represent, for example, time periods, then it 
may be useful to refer to the *next* or *previous* member. 
There are special operations to do this, and they are  discussed in 
chapter "Sets as Sequences: Ordered Sets". For now, 
it is enough to remember that the order in which set elements are 
specified is not relevant, unless and until some operation implying 
order is used. At that time, the rules change, and the set becomes what 
we will later call an *ordered* set. 


Subsets
--------

It is often necessary to define sets whose members must all be members of 
some larger set. For instance, we may wish to define the sectors in an 
economic model: ::

    i =  Set(m, 
             name = "i",
             description = "all sectors",  
             records = ["light-ind","food+agr","heavy-ind","services"])
    t =  Set(m, 
             name = "t",  
             domain = "i", 
             description = "traded sectors",
             records = ["light-ind","food+agr","heavy-ind"])
    nt = Set(m, 
             name = "nt",
             description = "non-traded sectors", 
             records = ["services"])

Some types of economic activity, for example exporting and importing,
may be logically restricted to a subset of all sectors. In order to model
the trade balance we need to know which sectors are traded, and one obvious
way is to list them explicitly, as in the definition of the set ``t`` above.
The *domain* specification for Set ``t`` means that each member of the set ``t`` 
must also be a member of the set ``i``. GAMS will enforce this relationship, 
which is called *domain checking*. Obviously, the order of declaration and definition 
is important: the membership of ``i`` must be known before ``t`` is defined,
otherwise checking cannot be done.

.. note::
    All elements of the subset must also be elements of the superset.

It is legal but unwise to define a subset without reference to the larger set, 
as is done above for the set ``nt``. In this case domain checking cannot be 
performed: if services were misspelled no error would be marked, but the model 
may give incorrect results. Hence, it is recommended to use domain checking 
whenever possible. It catches errors and allows to write models that are 
conceptually cleaner because logical relationships are made explicit.

An alternative way to define elements of a subset is with assignments: ::

    i =  Set(m, 
             name = "i",
             description = "all sectors",  
             records = ["light-ind","food+agr","heavy-ind","services"])
    t =  Set(m, 
             name = "t",  
             domain = "i", 
             description = "traded sectors",
             records = ["light-ind","heavy-ind"])
    t['food+agr'] = True

In the last line the element ``food+agr`` of the set ``i`` is assigned to the subset 
``t``. Assignments may also be used to remove an element from a subset:

    t['light-ind'] = False


.. note::
    - Note that if a subset is assigned to, it then becomes a dynamic set.
    - A subset can be used as a domain in the declaration of other sets, variables, 
      parameters and in equations as long as it is no dynamic set.


Multi-Dimensional Sets
=======================

It is often necessary to provide mappings between elements of different sets. For 
this purpose, GAMSPy allows the use of multi-dimensional sets. The current maximum 
number of permitted dimensions is 20. The next two subsections explain how 
to express one-to-one and many-to-many mappings between sets.

One-to-one Mapping
-------------------

Consider a set whose elements are pairs: :math:`A = \{(b,d),(a,c),(c,e)\}`. In this 
set there are three elements and each element consists of a pair of letters. This kind 
of set is useful in many types of modeling. In the following example a port has to be 
associated with a nearby mining region: ::

    