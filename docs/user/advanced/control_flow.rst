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

The If Statement
----------------
The :meth:`If <gamspy.If>` class maps directly to the GAMS ``if`` statement. It allows you to branch 
conditionally around a group of execution statements within control flow constructs like loops.
Currently, its use is restricted within a :meth:`Loop <gamspy.Loop>` statement.
Like :meth:`Loop <gamspy.Loop>`, it is implemented as a Python context manager and takes a GAMSPy 
logical condition as its argument:

.. code-block:: python

    import gamspy as gp
    
    m = gp.Container()
    i = gp.Set(m, records=[f"i{idx}" for idx in range(1, 11)])
    cnt = gp.Parameter(m, records=0)
    
    with gp.Loop(i):
        # Only execute the following block if the condition is met
        with gp.If(gp.Ord(i) > 5):
            cnt[...] += 1


Break and Continue
------------------
When iterating using a :meth:`Loop <gamspy.Loop>`, you can capture the loop instance using the ``as`` 
keyword (e.g., ``with gp.Loop(i) as loop:``). This instance provides access to the :meth:`Break <gamspy.Loop.Break>` 
and :meth:`Continue <gamspy.Loop.Continue>` properties, which map to the GAMS ``break`` and 
``continue`` statements.

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