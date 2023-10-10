.. _set:

***
Set
***

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
which is called element text: ::
     
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