.. _set:

***
Set
***

Simple Sets
============

Introduction
-------------

Sets are the basic building blocks of a GAMSPy model, corresponding exactly 
to the indices in the algebraic representations of models. A simple set 
consists of a set name and the elements of the set. Example: ::

    m = Container()
    i = Set(m, name = "i", records = ["seattle", "san-diego"], description = "plants")
    j = Set(m, name = "j", records = ['new-york', 'chicago', ‘topeka’], description = "markets")

The effect of these statements is probably self-evident. We declared two sets using 
the :meth:`gamspy.Set` class and gave them the names ``i`` and ``j``. We also 
assigned members to the sets as follows:

- :math:`i = \{Seattle, San Diego\}`
- :math:`j = \{New York, Chicago, Topeka\}`

They are labels, but are often referred to as elements or members. The optional ``description`` 
may be used to describe the set for future reference and to ease readability.

Besides using the ``Set()`` class directly, one can also facilitate the ``addSet()`` method 
of the :meth:`gamspy.Container` class: ::

    i = m.addSet(name="i", records = ["seattle", "san-diego"], description="plants")

Set declaration and data assignment can also be done separately: ::
     
    m = Container()
    i = Set(m, "i", description="plants")
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
                 uni             element_text
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
             domain = i, 
             description = "traded sectors",
             records = ["light-ind","food+agr","heavy-ind"])
    nt = Set(m, 
             name = "nt",
             description = "non-traded sectors", 
             records = ["services"])

====  ==========  ==========  ==============
  ..  i           t           nt
====  ==========  ==========  ==============
   0  light-ind   light-ind   
   1  food+agr    food+agr
   2  heavy-ind   heavy-ind
   3  services                services
====  ==========  ==========  ==============

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
             domain = i, 
             description = "traded sectors",
             records = ["light-ind","heavy-ind"])
    t['food+agr'] = True

In the last line the element ``food+agr`` of the set ``i`` is assigned to the subset 
``t``. Assignments may also be used to remove an element from a subset: ::

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

    m = Container()

    i = Set(m, 
            name = "i", 
            description = "mining regions", 
            records = ["china","ghana","russia","s-leone"])
    n = Set(m, 
            name = "n", 
            description = "ports", 
            records = ["accra","freetown","leningrad","shanghai"])
    
    s = pd.Series(
       index=pd.MultiIndex.from_tuples([("china", "shanghai"), 
                                        ("ghana", "accra"), 
                                        ("russia", "leningrad"), 
                                        ("s-leone", "freetown")])
    )
    
    # Alternative:
    #
    # s = pd.DataFrame([("china", "shanghai"),
    #                   ("ghana", "accra"),
    #                   ("russia", "leningrad"),
    #                   ("s-leone", "freetown")], 
    #                  columns=["i","n"])
    #
    # Note that uels_on_axes needs to be set to False in multi_in in this case.

    multi_in = Set(m, 
                   name = "in", 
                   domain = [i, n], 
                   description = "mines to ports map", 
                   uels_on_axes=True, 
                   records=s)

::

    In [1]: multi_in.records
    Out[1]:
    	      i	        n	element_text
    0	  china	 shanghai	
    1	  ghana     accra	
    2	 russia	leningrad	
    3	s-leone	 freetown	


Here ``i`` is the set of mining regions, ``n`` is the set of ports and ``in`` is a two 
dimensional set that associates each port with a mining region. The pairs are created 
using tuples in a pandas MultiIndex object. The set in has four elements, and each 
element consists of a region-port pair. The ``domain = [i,n]`` indicates that the 
first member of each pair must be a member of the set ``i`` of mining regions, and 
that the second must be in the set ``n`` of ports. GAMS will domain check the set 
elements to ensure that all members belong to the appropriate sets.


Many-to-Many Mapping
---------------------

A many-to-many mapping is needed in certain cases. Consider the following sets: ::

    m = Container()
    
    i = Set(m, name = "i", records = ["a","b"])
    j = Set(m, name = "j", records = ["c","d","e"])
    
    ij1_data = pd.Series(
       index=pd.MultiIndex.from_tuples([("a", "c"), 
                                        ("a", "d")])
    )

    ij2_data = pd.Series(
       index=pd.MultiIndex.from_tuples([("a", "c"), 
                                        ("b", "c")])
    )

    ij3_data = pd.Series(
       index=pd.MultiIndex.from_tuples([("a", "c"), 
                                        ("b", "c"), 
                                        ("a", "d"), 
                                        ("b", "d")])
    )
    
    ij1 = Set(m, name = "ij1", domain = [i, j], uels_on_axes=True, records=ij1_data)
    ij2 = Set(m, name = "ij2", domain = [i, j], uels_on_axes=True, records=ij2_data)
    ij3 = Set(m, name = "ij3", domain = [i, j], uels_on_axes=True, records=ij3_data)

Here the set ``ij1`` presents a *one-to-many* mapping where one element of the set ``i`` 
maps onto many elements of the set ``j``. The set ``ij2`` represents a *many-to-one* 
mapping where many elements of the set ``i`` map onto one element of the set ``j``. 
The set ``ij3`` is the most general case: a *many-to-many* mapping where many elements 
of the set ``i`` map to many elements of the set ``j``:

::

    In [1]: ij3.records
    Out[1]:
    	i	j	element_text
    0	a	c	
    1	b	c	
    2	a	d	
    3	b	d	

..
    # TODO: Projection and Aggregation of Sets?



Singleton Sets
===============

A singleton set in GAMS is a special set that has at most one element (zero elements 
are allowed as well). Like other sets, singleton sets may have a domain with several 
dimensions. Singleton sets are declared with the boolean ``is_singleton`` in the 
:meth:`gamspy.Set` class (or the :meth:`gamspy.Container` class). ::

    m = Container()

    i = Set(m, name = "i", records = ["a","b","c"])
    j = Set(m, name = "j", is_singleton = True, records = ["d"])
    k = Set(m, name = "k", is_singleton = True, domain = i, records = ["b"])
    l = Set(m, name = "l", is_singleton = True, uels_on_axes=True, domain = [i,i], 
            records = pd.Series(
               index=pd.MultiIndex.from_tuples([("b", "c")])
            ))

    In [1]: i.records
    Out[1]:
      uni	element_text
    0	a	
    1	b	
    2	c	

    In [2]: j.records
    Out[2]:
      uni	element_text
    0	d	

    In [3]: k.records
    Out[3]:
      uni	element_text
    0	b	

    In [4]: l.records
    Out[4]:
      i_0	i_1	element_text
    0	b	  c	

The sets ``j``, ``k`` and ``l`` are declared as singleton sets, each of them has just 
one element. The set ``k`` is a subset of the set ``i`` and the set ``l`` is a 
two-dimensional set.

Note that a data statement for a singleton set with more than one element will create 
a compilation error: ::

    m = Container()
    j = Set(m, name = "j", is_singleton = True, records = range(1,5))

..
    #TODO: Add compilation error as soon as GAMSPy is fixed

It also possible to assign an element to a singleton set. In this case the singleton set 
is automatically cleared of the previous element first. For example, adding the following 
line to the code above will result in set ``k`` containing only element ``a`` after 
execution: ::

    k['a'] = True

Singleton sets can be especially useful in assignment statements since they do not need to 
be controlled by a controlling index or an indexed operator like other sets. Consider the 
following example: ::

    m = Container()

    i = Set(m, name = "i", records = ["a","b","c"])
    k = Set(m, name = "k", is_singleton = True, domain = i, records = ["b"])
    h = Set(m, name = "h", is_singleton = True, domain = i, records = ["a"])
    n = Parameter(m, name = "n", domain = i, records = [['a', 2],['b', 3],['c', 5]])
    
    z1 = Parameter(m, name = "z1")
    z2 = Parameter(m, name = "z2")
    
    z1.assignment = n[k]
    z2.assignment = n[k] + 100*n[h]

The singleton sets ``k`` and ``h`` are both subsets of the set ``i``. The parameter ``n`` 
is defined over the set ``i``. The scalar ``z1`` is assigned a value of the parameter ``n`` 
without naming the respective label explicitly in the assignment. It is already specified 
in the definition of the singleton set ``k``. The assignment statement for the scalar ``z2`` 
contains an expression where the singleton sets ``k`` and ``h`` are referenced without a 
controlling index or an indexed operation.

.. note::
    Singleton sets cannot be used as domains.


The Universal Set: * as Set Identifier
=======================================

GAMS provides the universal set denoted by ``*`` for cases where the user wishes not to 
specify an index but have only a placeholder for it. The following examples show two ways 
how the universal set is introduced in a model. We will discuss the advantages and 
disadvantages of using the universal set later. First example:  ::

    m = Container()
    r = Set(m, name = "r", description = "raw materials", records = ["scrap","new"])
    misc = Parameter(m, name = "misc", domain = ['*',r], 
                     records = [['max-stock', "scrap", 400],
                                ['max-stock', "new", 275],
                                ['storage-c', "scrap", 0.5],
                                ['storage-c', "new", 2],
                                ['res-value', "scrap", 15],
                                ['res-value', "new", 25]])

In our example, the first index of parameter ``misc``` is the universal set `'*'` and the 
second index is the previously defined set ``r``. Since the first index is the universal set 
any entry whatsoever is allowed in this position. In the second position elements of the set 
``r`` must appear, they are domain checked, as usual.

The second example illustrates how the universal set is introduced in a model with an 
:meth:`gamspy.Alias` statement: ::

    m = Container()
    r = UniverseAlias(m, name = "new_universe")
    k = Set(m, name = "k", domain = new_universe, records = "Chicago")

The :meth:`gamspy.UniverseAlias` statement links the universal set with the set name 
``new_universe``. Set ``k`` is a subset of the universal set and ``Chicago`` is declared to 
be an element of ``k``. Any item may be added freely to ``k``.

.. note::
    It is recommended to not use the universal set for data input, since there is no domain 
    checking and thus typos will not be detected and data that the user intends to be in the 
    model might actually not be part of it.

Observe that in GAMS a simple set is always regarded as a subset of the universal set. Thus the 
set definition ::

    i = Set(m, "i", records = range(1,10))

is the same as ::

    i = Set(m, "i", domain = '*', records = range(1,10))

GAMS follows the concept of a domain tree for domains in GAMS. It is assumed that a set and its 
subset are connected by an arc where the two sets are nodes. Now consider the following one 
dimensional subsets: ::

    m = Container()
    i   = Set(m, "i")
    ii  = Set(m, "ii",  domain = i)
    j   = Set(m, "j",   domain = i)
    jj  = Set(m, "jj",  domain = j)
    jjj = Set(m, "jjj", domain = jj)

These subsets are connected with arcs to the set ``i`` and thus form a domain tree that is rooted 
in the universe node ``'*'``. This particular domain tree may be represented as follows: ::

    * - i - ii
          |
          - j - jj - jjj 

Note that with the construct ``Set(m, "i",  domain = jjj)`` we may access ``ii`` iterating through 
the members of ``jjj``.

Observe that the universal set is assumed to be ordered and operators for ordered sets such ord, 
lag and lead may be applied to any sets aliased with the universal set.


Set and Set Element Referencing
===============================

Sets or set elements are referenced in many contexts, including assignments, calculations, 
equation definitions and loops. Usually GAMS statements refer to the whole set or a single set 
element. In addition, GAMS provides several ways to refer to more than one, but not all elements 
of a set. In the following subsections we will show by example how this is done. 


Referencing the Whole Set
-------------------------

Most commonly whole sets are referenced as in the following examples: ::

    m = Container()

    i = Set(m, "i", records = [("i" + str(i), i) for i in range(1,101)])

    k = Parameter(m, "k", domain = i)
    k[i].assignment = 4
    
    z = Parameter(m, "z")
    z.assignment = Sum(i, k[i]) 

The parameter ``k`` is declared over the set ``i``, in the assignment statement in the next line 
all elements of the set ``i`` are assigned the value 4. The scalar ``z`` is defined to be the 
:meth:`gamspy.Sum` of all values of the parameter k(i).

Referencing a Single Element
----------------------------

Sometimes it is necessary to refer to specific set elements. This is done by using quotes around 
the label(s). We may add the following line to the example above: ::

    k['i77'] = 15

Referencing a Part of a Set
----------------------------

There are multiple ways to restrict the domain to more than one element, e.g. subsets, 
conditionals and tuples. Suppose we want the parameter ``k`` from the example above to be 
assigned the value 10 for the first 8 elements of the set ``i``. The following two lines of 
code illustrate how easily this may be accomplished with a subset: ::
    
    j = Set(m, "j", domain = i, records = i.records[0:8])
    k[j] = 10

First we define the set ``j`` to be a subset of the set ``i`` with exactly the elements we are 
interested in. Then we assign the new value to the elements of this subset. The other values of 
the parameter ``k`` remain unchanged. For examples using conditionals and tuples, see sections 
Restricting the Domain: Conditionals and Restricting the Domain: Tuples respectively.

..
    #TODO: Add links



Set Attributes
==============

A GAMSPy set has several attributes attached to it. For a complete list see :meth:`gamspy.Set`. 
The attributes may be accessed like in the following example: ::

    data[set_name] = set_name.attribute

Here ``data`` is a parameter, ``set_name`` is the name of the set and ``.attribute`` is one of 
the attributes listed in :meth:`gamspy.Set`. The following example serves as illustration: ::

    m = Container()

    id = Set(m, "id", records = [("Madison","Wisconsin"),
                                 ("tea-time","5"),
                                 ("-inf",""),
                                 ("-7",""), 
                                 ("13.14","")])
    
    attr = Parameter(m, "attr", domain = [id, '*'], description = "Set attribute values")
    
    attr[id,'position']    = id.pos 
    attr[id,'reverse']     = id.rev 
    attr[id,'offset']      = id.off 
    attr[id,'length']      = id.len 
    attr[id,'textLength']  = id.tlen 
    attr[id,'first']       = id.first
    attr[id,'last']        = id.last 

The parameter ``attr`` is declared to have two dimensions with the set ``id`` in the first 
position and the universal set in the second position. In the following seven statements the 
values of ``attr`` are defined for seven entries of the universal set.

========  ==========  =========  ========  ========  ============  =======  ======
..          position    reverse    offset    length    textLength    first    last
========  ==========  =========  ========  ========  ============  =======  ======
Madison            1          4                   7             9        1        
tea-time           2          3         1         8             1
-inf               3          2         2         4           
-7                 4          1         3         2           
13.14              5                    4         5                              1
========  ==========  =========  ========  ========  ============  =======  ======


Implicit Set Definition
=======================

Sets can be defined through data statements in the declaration. Alternatively, sets can be 
defined implicitly through data statements of other symbols which use these sets as domains. 
This is illustrated in the following example, which is derived from the 
[:ref:`trnsport <trnsport>`] model: ::

    m = Container()

    distances = pd.DataFrame(
        [
            ["seattle", "new-york", 2.5],
            ["seattle", "chicago", 1.7],
            ["seattle", "topeka", 1.8],
            ["san-diego", "new-york", 2.5],
            ["san-diego", "chicago", 1.8],
            ["san-diego", "topeka", 1.4],
        ],
        columns=["from", "to", "distance"],
    ).set_index(["from", "to"])
    
    i = Set(m, name="i", description="plants")
    j = Set(m, name="j", description="markets")
    
    d = Parameter(m, name="d", 
                  domain=[i, j],
                  description="distance in thousands of miles",
                  records = distances.reset_index(),
                  domain_forwarding = True
    )

The ``domain_forwarding = True`` in the declaration of :meth:`gamspy.Parameter` ``d`` 
forces set elements to be recursively included in all parent sets. Here set ``i`` 
will therefore contain all elements which define the first dimension of symbol ``d`` 
and set ``j`` will contain all elements which define the second dimension of symbol 
``d``. ::

    In [1]: i.records
    Out[1]:
    	      uni	element_text
    0	  seattle	
    1	san-diego	

    In [2]: j.records
    Out[2]:
             uni	element_text
    0	new-york	
    1	 chicago	
    2	  topeka	
        
Note, that ``domain_forwarding`` can also pass as a list of *bool* to control which 
domains to forward. Also ``domain_forwarding`` is not limited to one symbol. One 
domain set can be defined through multiple symbols using the same domain.


Dynamic Sets
============

Introduction
-------------

In this section we introduce a special type of sets: *dynamic sets*. The sets that 
we discuss in detail above have their elements stated at compile time and during 
execution time the membership is never changed. Therefore they are called *static* 
*sets*. In contrast, the elements of dynamic sets are not fixed, but may be added 
and removed during execution of the program. Dynamic sets are most often used as 
controlling indices in assignments or equation definitions and as the conditional 
set in an indexed operation. We will first show how assignments 
are used to change set membership in dynamic sets. Then we will introduce set 
operations and the last part of this chapter covers dynamic sets in the context 
of conditions.

Assigning Membership to Dynamic Sets
-------------------------------------

The Syntax
^^^^^^^^^^
Like any other set, a dynamic set has to be declared before it may be used in the 
model. Often, a dynamic set is declared as subset of a static set. Dynamic sets in 
GAMS may also be multi-dimensional like static sets. The maximum number of permitted 
dimensions follows the rules of the basic Data Types and Definitions. For 
multi-dimensional dynamic sets the index sets can also be specified explicitly at 
declaration. That way dynamic sets are domain checked. Of course it is also possible 
to use dynamic sets that are not domain checked. This provides additional power and 
flexibility but also a lack of intelligibility and danger. Any label is legal as long 
as such a set's dimension, once established, is preserved.

In general, the syntax for assigning membership to dynamic sets in GAMS is: ::

    set_name[index_list | label] = True | False

``Set_name`` is the internal name of the set in GAMS, ``index_list`` refers to the 
domain of the dynamic set and ``label`` is one specific element of the domain. An 
assignment statement may assign membership to the dynamic set either to the whole 
domain or to a subset of the domain or to one specific element. Note that, as usual, 
a label must appear in quotes.

Illustrative Example
^^^^^^^^^^^^^^^^^^^^^

We start with assignments of membership to dynamic sets ::

    m = Container()

    item     = Set(m, name="item", records = ["dish", "ink", "lipstick", "pen", "pencil", "perfume"])
    subitem1 = Set(m, name="subitem1", records = ["pen", "pencil"], domain = item)
    subitem2 = Set(m, name="subitem2", domain = item)
    
    subitem1["ink"]      = True 
    subitem1["lipstick"] = True 
    subitem2[item]       = True 
    subitem2["perfume"]  = False

Note that the sets ``subitem1`` and ``subitem2`` are declared like any other set. The 
two sets become dynamic as soon as they are assigned to. They are also domain checked: 
the only members they will ever be able to have must also be members of the set 
``item``.
The first assignment not only makes the set ``subitem1`` dynamic, it also has the effect 
that its superset ``item`` becomes a static set and from then on its membership is 
frozen. The first two assignments each add one new element to ``subitem1``. Note that both 
are also elements of ``item``, as required. The third assignment is an example of the 
familiar indexed assignment: ``subitem2`` is assigned all the members of ``item``. The last 
assignment removes the label ``'perfume'`` from the dynamic set ``subitem2``. ::

    In [1]: print(*subitem1.records["items"], sep=", ")
    Out[1]: ink, lipstick, pen, pencil

    In [2]: print(*subitem2.records["items"], sep=", ")
    Out[2]: dish, ink, lipstick, pen, pencil

Note that even though the labels ``'pen'`` and ``'pencil'`` were declared to be members of 
the set ``subitem1`` before the assignment statements that added the labels ``'ink'`` and 
``'lipstick'`` to the set, they appear in the listing above at the end. The reason is that 
elements are displayed in the internal order, which in this case is the order specified in 
the declaration of the set item.

Dynamic Sets with Multiple Indices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Dynamic sets may be multi-dimensional. The following lines continue the example above and 
illustrate assignments for multi-dimensional sets. ::

    sold = Set(m, "sold", records = ["pencil", "pen"], domain = item)
    sup  = Set(m, "sup", records = ["bic", "parker", "waterman"])
    supply = Set(m, "supply", domain = [sold, sup])
    
    supply["pencil", "bic"] = True
    supply["pen", sup] = True


::

    In [1]: supply.records
    Out[1]:
    	  sold	     sup	element_text
    0	   pen	     bic	
    1	   pen	  parker	
    2	   pen	waterman	
    3	pencil	     bic	


Equations Defined over the Domain of Dynamic Sets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generally, dynamic sets are not permitted as domains in *declarations* of sets, variables, 
parameters and equations. However, they may be *referenced* and sometimes it is necessary 
to define an equation over a dynamic set.

.. note::
    The trick is to declare the equation over the entire domain but define it over the dynamic 
    set.

For example, defining an equation over a dynamic set can be necessary in models that will be 
solved for arbitrary groupings of regions simultaneously. We assume there are no explicit links 
between regions, but that we have a number of independent models with a common data definition 
and common logic. We illustrate with an artificial example, leaving out lots of details. ::

    m = Container()

    allr = Set(m, "allr", records = ["N", "S", "W", "E", "N-E", "S-W"], description = "all regions")
    r    = Set(m, "r", domain = allr, description = "region subset for particular solution")
    type = Set(m, "type", description = "set for various types of data")
    
    price = Parameter(m, "price", records = 10)
    data = Parameter(m, "data", domain = [allr, type], description = "all other data ...")
    
    activity1 = Variable(m, "activity1", domain = allr, description = "first activity")
    activity1 = Variable(m, "activity2", domain = allr, description = "second activity")
    revenue = Variable(m, "revenue", domain = allr, description = "revenue")
    
    resource1 = Equation(m, "resource1", domain = allr, description = "first resource constraint ...")
    prodbal1 = Equation(m, "prodbal1", domain = allr, description = "first production balance ...")
    
    resource1[r] =  activity1[r]       <=  data[r,'resource-1']
    prodbal1[r] =   activity2[r]*price == revenue[r]

To repeat the important point: the equation is *declared* over the set ``allr``, but 
*defined* over ``r``, a subset. Note that the variables and data are *declared* over 
``allr`` but referenced over ``r``. Then the set ``r`` may be assigned arbitrary 
combinations of elements of the set ``allr``, and the model may be solved any number 
of times for the chosen groupings of regions.

Assigning Membership to Singleton Sets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Singleton sets have only one element. Hence any assignment to a singleton set first 
clears or empties the set, no explicit action to clear the set is necessary. This is 
illustrated with the following example: ::

    m = Container()

    i  = Set(m, "i", records = ["a", "b", "c"], description = "Static Set")
    ii = Set(m, "ii", domain = i, records = "b", description = "Dynamic Set")
    si = Set(m, "si", domain = i, records = "b", is_singleton = True, description = "Dynamic Singleton Set")
    
    ii["c"] = True
    si["c"] = True

Note that both ``ii`` and ``si`` are subsets of the set ``i``, but only ``si`` is declared as a 
*singleton set*. The assignment statements assign to both sets the element ``'c'``. While ``'c'`` 
is *added* to the set ``ii``, ``it`` *replaces* the original element in the singleton set ``si``: ::

    In [1]: print(*ii.records["i"], sep=", ")
    Out[1]: b, c

    In [2]: print(*si.records["i"], sep=", ")
    Out[2]: c

The assignment behavior can be changed with :meth:`gamspy.Container.addOptions` 
`strictSingleton <https://www.gams.com/latest/docs/UG_GamsCall.html#GAMSAOstrictsingleton>`_  
which affects the behavior of a membership assignment to a Singleton Set. With 
``strictSingleton=0`` GAMS does not complain about an assignment with more than one element on the 
right hand side but takes the first one. With ``strictSingleton=1`` (default), such an assignment 
raises an error. Consider the following example: ::

    m = Container()

    i = Set(m, "i", records = ["a", "b", "c"], description = "Static Set")
    si = Set(m, "s", domain = i, is_singleton = True)
    
    si[i].where[Ord(i) > 1] = True

By default, the above code will trigger an error as an assignment to a singleton set with more than 
one element on the right hand side is forbidden: ::

    **** Exec Error at line 5: Multiple assignment to Singleton Set not allowed (see option strictSingleton)

However, with option ``strictSingleton=0`` GAMS does not complain about such an assignment with more than 
one element on the right hand side but takes the first one: ::

    m = Container()

    i = Set(m, "i", records = ["a", "b", "c"], description = "Static Set")
    si = Set(m, "s", domain = i, is_singleton = True)
    
    m.addOptions({"strictSingleton": 0})
    si[i].where[Ord(i) > 1] = True

::

    In [1]: print(*si.records["i"])
    Out[1]: b


Set Operations
---------------

GAMSPy provides symbols for arithmetic set operations that may be used with dynamic sets. An 
overview of the set operations in GAMS is given below. Examples and alternative formulations 
for each operation follow. Note that in the table below the set ``i`` is the static superset 
and the sets ``j`` and ``k`` are dynamic sets.

=====================================  ===============  =====================================================================================================
Set Operation                          Operator         Description
=====================================  ===============  =====================================================================================================
Set Union                              j(i) + k(i)      Returns a subset of i that contains all the elements of the sets j and k.
Set Intersection                       j(i) * k(i)      Returns a subset of i that contains the elements of the set j that are also elements of the set k.
Set Complement                         not j(i)         Returns a subset of i that contains all the elements of the set i that are not elements of the set j.
Set Difference                         j(i) - k(i)      Returns a subset of i that contains all the elements of the set j that are not elements of the set k.
=====================================  ===============  =====================================================================================================

Example: The set ``item`` is the superset of the dynamic sets ``subitem1`` and ``subitem2``. 
We add new dynamic sets for the results of the respective set operations. ::

    m = Container()

    item     = Set(m, name="item", records = ["dish", "ink", "lipstick", "pen", "pencil", "perfume"])
    subitem1 = Set(m, name="subitem1", records = ["pen", "pencil"], domain = item)
    subitem2 = Set(m, name="subitem2", domain = item)
    
    subitem1["ink"]      = True
    subitem1["lipstick"] = True
    subitem2[item]       = True
    subitem2["perfume"]  = False
    
    union1        = Set(m, "union1", domain = item)
    union2        = Set(m, "union2", domain = item)
    intersection1 = Set(m, "intersection1", domain = item)
    intersection2 = Set(m, "intersection2", domain = item)
    complement1   = Set(m, "complement1", domain = item)
    complement2   = Set(m, "complement2", domain = item)
    difference1   = Set(m, "difference1", domain = item)
    difference2   = Set(m, "difference2", domain = item)
    
    union1[item]     = subitem2[item] + subitem1[item]
    union2[subitem1] = True
    union2[subitem2] = True
    
    intersection1[item] = subitem2[item] * subitem1[item]
    intersection2[item] = Number(1).where[subitem1[item] & subitem2[item]]
    
    complement1[item]     = ~subitem1[item]
    complement2[item]     = True
    complement2[subitem1] = False
    
    difference1[item]     = subitem2[item] - subitem1[item]
    difference2[item]     = Number(1).where[subitem2[item]]
    difference2[subitem1] = False

::

    In [1]: print(*intersection1.records["item"], sep=", ")
    Out[1]: ink, lipstick, pen, pencil

Looking at the results of each operation will show that the above assignment statements 
for each operation result in the same dynamic set like using the set operator. Observe 
that the alternative formulations for the set intersection and set difference involve 
conditional assignments. Conditional assignments in the context of dynamic sets are 
discussed in depth in the next section.

.. note::
    The indexed operation :meth:`gamspy.Sum` may be used for set unions. Similarly, 
    the indexed operation :meth:`gamspy.Product` may be used for set intersections. 
    For examples see section "Conditional Indexed Operations with Dynamic Sets" below.


Controlling Dynamic Sets
-------------------------

