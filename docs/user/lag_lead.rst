.. _lag_lead:

************
Lag and Lead
************

``Lag`` and ``Lead`` operators can be used on ordered sets only via the methods
``lag()`` and ``lead()``. They are used to relate the current member of an
ordered set to the previous or next member of the set. Both ``lag()`` and
``lead()`` require the argument ``n`` indicating the element offset to be
applied. The optional argument ``type="linear"`` can be used to specify
the behavior of the operator (``"linear"`` or ``"circular"``). The following
table gives an overview of the available combinations:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Operation
     - Description
   * - ``i.lag(n, "linear")``
     - Refers to the element of the ordered set ``i`` whose relative position in the set is ``Ord(i)-n``.
   * - ``i.lead(n, "linear")``
     - Refers to the element of the ordered set ``i`` whose relative position in the set is ``Ord(i)+n``.
   * - ``i.lag(n, "circular")``
     - Same as ``i.lag(n, "linear")``, only here the first element of the set is assumed to be preceded by the last element of the set, thus forming a circle of elements.
   * - ``i.lead(n, "circular")``
     - Same as ``i.lead(n, "linear")``, only here the last element of the set is assumed to be followed by the first element of the set, thus forming a circle of elements.

Note that the only difference between ``type="linear"`` and ``type="circular"``
is how endpoints are treated. Linear operators assume that there are no
elements preceding the first and following the last element of the ordered set.
This assumption may result in elements of the set being referenced that
actually do not exist. Therefore the user must think carefully about the
treatment of endpoints. Models with linear lag and lead operators will need
special exception handling logic to deal with them. The following sections will
describe how this issue is handled in ``GAMSPy`` in the context in which these
operators are typically used: assignments and equation definitions. Linear lag and lead
operators are useful for modeling time periods that do not repeat, like a set
of years (say ``"1990"`` to ``"1997"``).

Circular lag and lead operators assume that the first and last element of the
set are adjacent, so as to form a circular sequence of members. All references
and assignments are defined. The assumption of circularity is useful for
modeling time periods that repeat, such as months of the year or hours in the
day. It is quite natural to think of January as the month following December.
Agricultural farm budget models and workforce scheduling models are examples of
applications where circular leads occur naturally.

Lags and Leads in Assignments
-----------------------------

Lag and lead operators may be used in assignments. The use of a lag or lead
operator on the right-hand side of an assignment is called a reference,
while their use on the left-hand side is called an assignment and involves the
definition of a domain of the assignment. The concepts behind reference and
assignment are equally valid for the linear and circular forms of the lag and
lead operators. However, the importance of the distinction between reference
and assignment is not pronounced for circular lag and lead operators, because
non-existent elements are not referred to in this case.

**Linear Lag and Lead Operators in Assignments - Reference**

The following example shows the use of ``lag()`` on the right-hand side of an assignment::

    import gamspy as gp
    
    m = gp.Container()
    t = gp.Set(
        m,
        name="t",
        description="time sequence",
        records=[f"y-{x}" for x in range(1987, 1992)],
    )
    a = gp.Parameter(m, name="a", domain=[t])
    b = gp.Parameter(m, name="b", domain=[t])
    
    a[t] = 1986 + gp.Ord(t)
    b[t] = -1
    b[t] = a[t.lag(1, "linear")]
    
    print(a.records)
    print(b.records)

This sets the values for the parameter ``a`` to ``1987``, ``1988`` up to ``1991``
corresponding to the labels ``"y-1987"``, ``"y-1988"`` and so on.
Observe that the parameter ``b`` is initialized to ``-1`` so that the result of
the next assignment can be seen clearly. The last assignment
uses ``lag()`` on the right-hand side, resulting in the values for ``b`` to
equal the values for ``a`` from the previous period. If there is no previous
period, as with the first element, ``"y-1987"``, the value zero is assigned,
replacing the previous value of ``-1`` (values of zero for parameters are not
displayed).

**Linear Lag and Lead Operators in Assignments - Assignment**

The next examples is a variation of the first one and uses ``lead()`` on the
left-hand side of an assignment::

    import gamspy as gp
    
    m = gp.Container()
    t = gp.Set(
        m,
        name="t",
        description="time sequence",
        records=[f"y-{x}" for x in range(1987, 1992)],
    )
    a = gp.Parameter(m, name="a", domain=[t])
    c = gp.Parameter(m, name="c", domain=[t])
    
    a[t] = 1986 + gp.Ord(t)
    c[t] = -1
    c[t.lead(2, "linear")] = a[t]
    
    print(a.records)
    print(c.records)

Here, the assignment to ``c`` involves the ``lead()`` operator on the left-hand
side. It is best to spell out step by step how this assignment is made. For
each element in ``t``, find the element of ``c`` associated with ``t+2``. If it
exists, replace its value with the value of ``a[t]``. If not (as with labels
``"y-1990"`` and ``"y-1991"``) make no assignment. The first element of the set ``t`` is
``"y-1987"``, therefore the first assignment is made to ``c["y-1989"]`` which takes
the value of ``a["y-1987"]``, that is ``1987``. No assignments at all are made to
``c["y-1987"]`` and ``c["y-1988"]``: these two retain their previous values of
``-1``.

**Circular Lag and Lead Operators in Assignments**

The following example illustrates the use of circular lag and lead operators in
assignment statements::

    import gamspy as gp
    
    m = gp.Container()
    s = gp.Set(
        m,
        name="s",
        description="seasons",
        records=["spring", "summer", "autumn", "winter"],
    )
    val = gp.Parameter(
        m,
        name="val",
        domain=[s],
        records=[["spring", 10], ["summer", 15], ["autumn", 12], ["winter", 8]],
    )
    lagval = gp.Parameter(m, name="lagval", domain=[s])
    leadval = gp.Parameter(m, name="leadval", domain=[s])
    
    lagval[s] = -1
    lagval[s] = val[s.lag(2, "circular")]
    leadval[s] = -1
    leadval[s.lead(1, "circular")] = val[s]
    
    print(val.records)
    print(lagval.records)
    print(leadval.records)

In the example parameter ``lagval`` is used for reference while ``leadval`` is
used for assignment. Notice that the case of circular lag and lead operators
does not refer to any non-existent elements. The difference between reference
and assignment is therefore not important. Note that the following two
statements from the example above::

    lagval[s] = val[s.lag(2, "circular")]
    leadval[s.lead(1, "circular")] = val[s]

are equivalent to::

    lagval[s.lead(2, "cicular")] = val[s]
    leadval[s] = val[s.lag(1, "circular")]

The use of reference and assignment have been reversed with no difference in effect.


Lags and Leads in Equations
---------------------------

A ``lag()`` or ``lead()`` to the left of an equation definition is a modification of the
domain of definition of the equation. The linear form may cause one or more
individual equations to be suppressed. A lag or lead operation to the right of
an equation definition is a reference. If the associated label is not defined,
the term vanishes.

**Linear Lag and Lead Operators in Equations - Domain Control**

Consider the following simple artificial multi-period example. We specify a
complete model and encourage users to solve it and further explore it::

    import gamspy as gp
    
    m = gp.Container()
    
    t = gp.Set(m, name="t", records=[f"t{x+1}" for x in range(5)])
    tfirst = gp.Set(m, name="tfirst", domain=[t])
    i = gp.Parameter(m, name="i", domain=[t])
    i[t] = 1
    k0 = gp.Parameter(m, name="k0", records=[3.0])
    tfirst[t] = gp.Number(1).where[gp.Ord(t) == 1]
    
    k = gp.Variable(m, name="k", domain=[t])
    z = gp.Variable(m, name="z")
    k.fx[tfirst] = k0
    
    kk = gp.Equation(m, name="kk", domain=[t])
    dummy = gp.Equation(m, name="dummy")
    kk[t.lead(1)] = k[t.lead(1)] == k[t] + i[t]
    dummy.definition = z == 0
    
    m1 = gp.Model(
        m,
        name="m1",
        equations=m.getEquations(),
        problem="LP",
        sense=gp.Sense.MIN,
        objective=z,
    )
    m1.solve()

Note that the equation ``kk`` is declared over the set ``t``, but it is defined
over the domain ``t.lead(1)``. Therefore the first equation that will be generated is the following::

    k["t2"]  ==  k["t1"] + i["t1"]

Note that the value of the variable ``k["t1"]`` is fixed at the value of scalar
``k0``. Observe that for the last element of ``t``, the term ``k[t.lead(1)]``
is not defined and therefore the equation will not be generated.

To summarize, the lead operator in the domain of definition has restricted the
number of constraints generated so that there are no references to non-existent
variables.

For a more realistic model that illustrates the usage of linear lag operators
in equations, see for example the optimal economic growth model ``ramsey.py``.

**Linear Lag and Lead Operators in Equations - Reference**

In the previous subsection we showed how to write the equation ``kk`` using the
lead operator for domain control in combination with fixing the variable
``k[tfirst]`` to ``k0``. An alternative formulation could neglect the fixing of
``k[tfirst]`` and use a lag operator and a condition in the expression of the
equation while the domain of definition is unrestricted::

    kk[t] = k[t] == k[t.lag(1)] + i[t.lag(1)] + k0.where[tfirst[t]]

Note that for the first element of the set ``t`` the terms ``k[t.lag(1)]`` and
``i[t.lag(1)]`` are not defined and therefore vanish. Without the conditional
term, the resulting equation would be::

    k["t1"] == 0

However, this would lead to different results as ``k["t1"]`` would not be set
to the value of ``k0`` anymore. Therefore the conditional expression
``k0.where[tfirst[t]]`` is added. Observe that in this formulation equations
are generated for all time periods, no equation is suppressed.

In general, the choice between using lag and lead operators as reference
like in the last example or in domain control is often a matter of taste.

**Circular Lag and Lead Operators in Equations**

In the case of circular lag and lead operators, the difference between their
use in domain control and as reference is not important because it does not
lead to any equations or terms being suppressed. Consider the following
artificial example::

    import gamspy as gp
    
    m = gp.Container()
    
    s = gp.Set(
        m,
        name="s",
        description="seasons",
        records=["spring", "summer", "autumn", "winter"],
    )
    produ = gp.Variable(
        m,
        name="produ",
        description="amount of goods produced in each season",
        domain=[s],
    )
    avail = gp.Variable(
        m,
        name="avail",
        description="amount of goods available in each season",
        domain=[s],
    )
    sold = gp.Variable(
        m,
        name="sold",
        description="amount of goods sold in each season",
        domain=[s],
    )
    matbal = gp.Equation(m, name="matbal", domain=[s])
    
    matbal[s] = avail[s.lead(1, "circular")] == avail[s] + produ[s] - sold[s]

In this example four individual equations are generated. They are listed below::

    avail["summer"] == avail["spring"] + produ["spring"] - sold["spring"]
    avail["autumn"] == avail["summer"] + produ["summer"] - sold["summer"]
    avail["winter"] == avail["autumn"] + produ["autumn"] - sold["autumn"]
    avail["spring"] == avail["winter"] + produ["winter"] - sold["winter"]

Note that for the last element of the set ``s`` the term
``avail[s.lead(1, "circular")]`` is evaluated to ``avail["spring"]``.
This term is well defined and therefore it does not vanish. Similarly, using
the circular lead operator in the domain of definition like in the following
line will result in the same four equations being generated as above and no
equation being suppressed::

    matbal[s.lead(1, "circular")] = avail[s.lead(1, "circular")] == avail[s] + produ[s] - sold[s]