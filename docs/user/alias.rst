.. _alias:

*****
Alias: Multiple Names for a Set
*****

Sometimes it is necessary to have more than one name for the same set. In input-output 
models for example, each commodity may be used in the production of all other commodities 
and it is necessary to have two names for the set of commodities to specify the problem 
without ambiguity. Example: ::
    
    m = Container()
    
    i = Set(m, name = "i", records=["i" + str(i) for i in range(5)])
    ip = Alias(m, name = "ip", , alias_with = i)

A second name ``ip`` for the set ``i`` is established that can be used instead of the original 
set name ``i``. 

.. note::
    The newly introduced set name may be used as an alternative name for the original set; 
    the associated set will always contain the same elements as the original set.

Besides using the ``Alias()`` class directly, one can also facilitate the ``addAlias()`` method 
of the ``Container`` class (which internally calls ``Alias()``): ::

    h = Set(m, name="h", records=list(range(1, 6)))
    hp = Alias(m, name="hp", alias_with=h)

It is possible to create an alias from another alias object. In this case a recursive search 
will be performed to find the root parent set – this is the set that will ultimately be stored 
as the ``alias_with`` property. We can see this behavior in the following example: ::
    
    m = Container()
    
    i = Set(m, name = "i", records=["i" + str(i) for i in range(5)])
    ip = Alias(m, name = "ip", alias_with = i)
    ipp = Alias(m, "ipp", ip)

::

    In [1]: ip.alias_with.name
    Out[1]: 'i'
     
    In [2]: ipp.alias_with.name
    Out[2]: 'i'

Typical examples for the usage of aliases are problems where transportation costs between 
members of one set have to be modeled. Example: ::

    from gamspy import Set, Alias, Parameter
    import pandas as pd
    
    m = Container()
    
    i = Set(m, name = "i", description = "plant locations", records=[
        "palmasola", "pto-suarez", "potosi", "baranquill", "cartagena"
    ])
    
    ip = Alias(m, name = "ip", , alias_with = i)
    
    cost = pd.DataFrame(
        [
            ("pto-suarez", "palmasola", 87.22),
            ("potosi",     "palmasola", 31.25),
            ("potosi",     "pto-suarez", 55.97),
            ("baranquill", "palmasola", 89.80),
            ("baranquill", "pto-suarez", 114.56),
            ("baranquill", "potosi", 70.68),
            ("cartagena",  "palmasola", 89.80),
            ("cartagena",  "pto-suarez", 114.56),
            ("cartagena",  "potosi", 70.68),
            ("cartagena",  "baranquill", 5.00),
        ],
        columns=["from", "to", "us$ per ton"],
    )
    
    tran = Parameter(m, name="tran", 
                           description = "transport cost for interplant shipments (us$ per ton)", 
                           domain=[i, i], records=cost)

This is how the transport cost look like: ::

    tran.records.pivot(columns = 'to', index='from', values = 'value')


::

    mui = Parameter(m, name = "mui",
                          description = "transport cost: interplant shipments (us$ per ton)",
                          domain = [i, ip])
    
    mui[i,ip] = tran[i,ip] + tran[ip,i]

Resulting in the following data for ``mui``::

    mui.records.pivot(columns = 'ip', index='i', values = 'value')


ip	palmasola	pto-suarez	potosi	baranquill	cartagena
i					
palmasola	NaN	87.22	31.25	89.80	89.80
pto-suarez	87.22	NaN	55.97	114.56	114.56
potosi	31.25	55.97	NaN	70.68	70.68
baranquill	89.80	114.56	70.68	NaN	5.00
cartagena	89.80	114.56	70.68	5.00	NaN


The alias statement introduces ``ip`` as another name for the set ``i``. The table ``tran`` is 
two-dimensional and both indices are the set ``i``. The data for the transport cost between 
the plants is given in this table; note that the transport costs are given only for one 
direction here, i.e. the costs from ``pto-suarez`` to ``palmasola`` are explicitly specified in 
the table while the costs in the opposite direction are not given at all. The parameter 
``mui`` is also two-dimensional and both indices refer to the set ``i``, but this time the alias 
``ip`` is used in the second position. The parameter ``mui`` is defined in the next line: 
``mui`` contains the transport costs from one plant location to 
the other, in both directions. Note that if ``mui`` were defined without the alias, then all 
its entries would have been zero. 

