.. _sddp_state_variables:

.. meta::
   :description: State variables in GAMSPy SDDP: add_state, bounds, and the initial state
   :keywords: SDDP, state variable, add_state, bounds, initial state, GAMSPy, gamspy, GAMS

***************
State Variables
***************

The **state** is the information SDDP carries from one stage to the next: the
reservoir level in ClearLake, an inventory, a stored energy level. It is what
links the stages together: everything the future needs to know about the past is
summarised in it. You register each state variable with ``add_state()``.

Registering a state
===================

.. code-block:: python

   sddp.add_state(variable=L, initial_state=100, upper_bound=250)

``add_state()`` takes the GAMSPy ``variable`` (indexed over the time set) plus
three optional descriptors: ``lower_bound``, ``upper_bound`` and
``initial_state``. The variable is an ordinary decision variable in your model;
registering it tells SDDP to track it across stages and to build cuts in terms
of it.

Bounds
======

A state's bounds do two jobs: they set the range the first iteration's trial
trajectories are seeded across, and they clamp the adaptive trial update in
later iterations. They are resolved in this order:

1. the value you pass to ``add_state()``;
2. otherwise the variable's recorded bounds (``variable.lo`` / ``variable.up``);
3. otherwise the default for the variable's type (a ``positive`` variable gives
   ``(0, +inf)``, and so on).

If you supply a bound that disagrees with the variable's recorded bound, your
value is used and a ``UserWarning`` is raised so the mismatch is not silent. The
lower bound must be strictly below the upper bound.

The initial state
=================

``initial_state`` is the value the state holds **before** the first stage: the
predecessor the stage-1 solve sees. In ClearLake the reservoir starts at 100, so
the ``jan`` balance is computed against that level. If you omit it, the state
falls back to its lower bound.

.. note::
   The state variable is referenced across stages through the *last* time step
   of each stage, which is the value handed to the next stage as its incoming
   state.

Multiple states
===============

A problem can have more than one state (coupled reservoirs, or joint inventory
and capacity) by calling ``add_state()`` once per variable. SDDP then builds a
single cut with one slope per state, and ``policy()`` takes the incoming state
as a ``dict`` keyed by variable name instead of a scalar. The single-state case
above is the common starting point; the multi-state workflow is covered in its
own example.

.. seealso::
   The :doc:`ClearLake tutorial <clearlake>` registers a state in context.
