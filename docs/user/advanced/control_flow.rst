.. _control:

Control Flow Structures
=======================

GAMSPy provides programming flow control features that map directly to native GAMS flow 
control statements. Because Python's native execution model evaluates eagerly, writing 
standard Python ``for`` loops to iterate over GAMS data can introduce massive 
performance overhead due to constant communication between Python and GAMS. 

To solve this, GAMSPy uses specialized constructs like context managers (``with`` statements) 
to generate the underlying GAMS execution statements dynamically, executing the entire 
block directly within the highly optimized GAMS engine.

Vectorized Assignment vs. The Loop Statement
--------------------------------------------
Before using a loop, consider whether your logic can be expressed as a standard parallel 
assignment. For example, simple recursive or arithmetic updates **do not** require a loop:

.. code-block:: python

    # BAD: Unnecessary loop
    with gp.Loop(t):
        pop[t + 1] = pop[t] + growth[t]
        
    # GOOD: Fast, native parallel assignment
    pop[t + 1] = pop[t] + growth[t]

The :meth:`Loop <gamspy.Loop>` context manager should be reserved for cases where parallel 
assignments are not sufficient. This includes complex iterative calculations, dynamic 
algorithmic heuristics, and modifying models to solve them repeatedly.

Iteratively Solving Models
^^^^^^^^^^^^^^^^^^^^^^^^^^
A common use case in operations research is modifying model parameters and solving the model 
multiple times within a loop. Executing this via a standard Python ``for`` loop is notoriously 
slow. By using :meth:`Loop <gamspy.Loop>`, you push the entire iterative solve process to the 
GAMS backend:

.. code-block:: python
    
    import gamspy as gp
    
    # ... (Assume sets i, j, variables x, and equations are already defined) ...
    
    transport = gp.Model(
        m,
        name="transport",
        equations=m.getEquations(),
        problem="LP",
        sense=gp.Sense.MIN,
        objective=gp.Sum((i, j), c[i, j] * x[i, j]),
    )
    
    k = gp.Set(m, domain=[i, j])
    k[i, j] = True

    # Slow iteration in Python
    for ival in i.toList():
        for jval in j.toList():
            transport.solve()
            c[ival, jval] = c[ival, jval] * 1.1
    
    # Fast iteration directly inside GAMS
    with gp.Loop(k):
        transport.solve()
        c[i, j] = c[i, j] * 1.1  # Update cost dynamically for the next solve

Dollar Conditions in Loops
^^^^^^^^^^^^^^^^^^^^^^^^^^
The domain of the loop can be restricted by a logical condition using the ``.where`` attribute. 
This acts identically to a dollar condition on a loop in GAMS, allowing you to filter the iteration 
space efficiently.

.. code-block:: python

    i = gp.Set(m, records=[f"i{idx}" for idx in range(1, 4)])
    j = gp.Set(m, records=[f"j{idx}" for idx in range(1, 6)])
    q = gp.Parameter(m, domain=[i, j], records=[("i1", "j1", 1), ("i1", "j2", 3)])
    x = gp.Parameter(m, records=1)
    
    # Only iterate over combinations where q[i, j] is strictly positive
    with gp.Loop(gp.Domain(i, j).where[q[i, j] > 0]):
        x[...] = x[...] + q[i, j]

The For Statement
-----------------
The :meth:`For <gamspy.For>` class maps to the GAMS ``for`` statement. While :meth:`Loop <gamspy.Loop>` 
is used to iterate over members of a set, :meth:`For <gamspy.For>` allows you to iterate over a range 
of numerical values, incrementing or decrementing a scalar parameter at each step. It is useful for 
iterative algorithmic calculations that require a numerical counter.

.. code-block:: python

    import gamspy as gp

    m = gp.Container()
    i = gp.Parameter(m) # Scalar parameter acting as the loop counter
    cnt = gp.Parameter(m, records=0)
    
    # Simple iteration over a numerical range (1 to 10)
    with gp.For(i, 1, 10):
        cnt[...] += i

You can also specify a custom step size. The step must be positive. To step downwards, 
one can specify the direction as follows: ::

.. code-block:: python

    x = gp.Parameter(m, records=10)
    
    # Iterating backwards from 10 down to 1 with a step size of -2
    with gp.For(i, 10, 1, 2, direction="downto"):
        x[...] = x[...] - 2

You can also use other Parameters or Expressions to define the start, end, and step boundaries of the loop dynamically.

The While Statement
-------------------
The :meth:`While <gamspy.While>` class maps to the GAMS ``while`` statement. It allows you to repeat a block of execution 
statements as long as a specific logical condition evaluates to ``True``.

Unlike :meth:`Loop <gamspy.Loop>` and :meth:`For <gamspy.For>`, a ``while`` loop continues for an unknown number of 
iterations until its condition is broken.

.. code-block:: python

    import gamspy as gp

    m = gp.Container()
    x = gp.Parameter(m, records=100)
    cnt = gp.Parameter(m, records=0)
    
    # Iteratively halve x until it is less than or equal to 1
    with gp.While(x > 1):
        x[...] = x / 2
        cnt[...] += 1

Just like loops, you can also capture the while instance using the ``as`` keyword to utilize :meth:`Break <gamspy.While.Break>` 
and :meth:`Continue <gamspy.While.Continue>` statements.

If, ElseIf, and Else Statements
-------------------------------
The :meth:`If <gamspy.If>`, :meth:`ElseIf <gamspy.ElseIf>`, and :meth:`Else <gamspy.Else>` classes map directly to the 
GAMS ``if``, ``elseif``, and ``else`` statements. They allow you to branch conditionally around a group of execution 
statements within control flow constructs like loops.

Currently, their use is restricted within a :meth:`Loop <gamspy.Loop>` or :meth:`For <gamspy.For>` statement. Like loops, 
they are implemented as Python context managers.

Note that an ``ElseIf`` or ``Else`` block must **immediately** follow a preceding ``If`` or ``ElseIf`` block without 
any intervening statements.

.. code-block:: python

    import gamspy as gp
    
    m = gp.Container()
    i = gp.Set(m, records=[f"i{idx}" for idx in range(1, 11)])
    cnt_small = gp.Parameter(m, records=0)
    cnt_medium = gp.Parameter(m, records=0)
    cnt_large = gp.Parameter(m, records=0)
    
    with gp.Loop(i):
        # Chain conditions together to branch execution
        with gp.If(gp.Ord(i) <= 3):
            cnt_small[...] += 1
        with gp.ElseIf(gp.Ord(i) <= 7):
            cnt_medium[...] += 1
        with gp.Else():
            cnt_large[...] += 1

Break and Continue
------------------
When iterating using a :meth:`Loop <gamspy.Loop>` or :meth:`For <gamspy.For>`, 
you can capture the loop instance using the ``as`` keyword (e.g., ``with gp.Loop(i) as loop:`` 
or ``with gp.For(i, 1, 10) as loop:``). This instance provides access to the ``Break`` 
and ``Continue`` properties, which map to the GAMS ``break`` and ``continue`` statements.

* **Continue:** Skips the remaining statements in the current iteration and proceeds to the next.
* **Break:** Terminates the execution of the current loop prematurely.

Using Continue
^^^^^^^^^^^^^^
You can combine :meth:`If <gamspy.If>` and :meth:`Continue <gamspy.Loop.Continue>` to skip specific 
iterations dynamically.

.. code-block:: python

    with gp.Loop(i) as loop:
        # Skip even numbers
        with gp.If(gp.math.mod(gp.Ord(i), 2) == 0):
            loop.Continue
            
        cnt[...] += 1

Using Break
^^^^^^^^^^^
You can use :meth:`Break <gamspy.Loop.Break>` to exit a loop early, such as when a heuristic search finds 
an acceptable candidate. Note that you can only break out of the innermost loop currently executing.

.. code-block:: python

    j = gp.Set(m, records=[f"j{idx}" for idx in range(1, 11)])

    with gp.Loop(i) as loop_i:
        with gp.Loop(j) as loop_j:
            
            with gp.If(j.sameAs("j5")):
                loop_j.Break  # Breaks the inner loop (j)
                
            cnt[...] += 1
            
        with gp.If(i.sameAs("i5")):
            loop_i.Break  # Breaks the outer loop (i)