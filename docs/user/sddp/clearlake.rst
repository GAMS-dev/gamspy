:html_theme.sidebar_secondary.remove:

.. _sddp_clearlake:

.. meta::
   :description: A worked SDDP example in GAMSPy: the ClearLake reservoir
   :keywords: SDDP, stochastic programming, example, reservoir, ClearLake, GAMSPy, gamspy, GAMS, policy, simulate

*******************
ClearLake Reservoir
*******************

This tutorial builds the reservoir problem from the :doc:`introduction
<introduction>` into a complete, runnable SDDP model. We declare the stage
problem in ordinary GAMSPy, hand it to an :meth:`SDDP
<gamspy.formulations.SDDP>` instance, train a policy, and then ask it what to
do and how well it performs.

The problem
===========

ClearLake is operated over four months. The **state** is the reservoir level
``L``; each month the operator picks a controlled release ``R``, a flood spill
``F`` and an import ``Z``. Water balance ties the level to the previous month's
level and the realised inflow:

.. math::

   L_t = L_{t-1} + \xi_t - R_t - F_t + Z_t,

where :math:`\xi_t` is the (uncertain) net inflow. Spilling costs 10 per unit
and importing 5 per unit; the operator minimises the expected total cost. The
level is capped at 250 and the release at 200, the reservoir starts at 100, and
each month the inflow takes one of three values with probabilities ``0.25``,
``0.50`` and ``0.25``.

Setting up the model
====================

We start from the data and a container. The inflow scenarios are a
``(n_stages, n_scenarios)`` array, with one row per month, one column per scenario.

.. code-block:: python

   import numpy as np
   import gamspy as gp
   from gamspy.formulations import SDDP

   # Problem data
   L_FLOOD     = 250.0   # reservoir capacity (flood threshold)
   L0          = 100.0   # initial water level
   R_MAX       = 200.0   # maximum controlled release
   FLOOD_COST  = 10.0    # cost per unit spilled
   IMPORT_COST = 5.0     # cost per unit imported

   # Three inflow scenarios per month and their probabilities.
   scenario_data = np.array([
       [ 50.0, 150.0, 350.0],   # jan
       [ 50.0, 150.0, 350.0],   # feb
       [-50.0, 100.0, 250.0],   # mar
       [-50.0, 100.0, 250.0],   # apr
   ])
   scenario_probs = [0.25, 0.50, 0.25]

The :meth:`SDDP <gamspy.formulations.SDDP>` instance is created early, because
it owns a special set, the **active-stage** singleton, that the stage
equations are gated on. ``stage_set`` is the set of stages, ``n_trials`` is the
number of forward trial paths per iteration, and ``seed`` fixes the sampler.

.. code-block:: python

   m = gp.Container()
   t = gp.Set(m, "t", records=["jan", "feb", "mar", "apr"], description="months")

   sddp = SDDP(m, stage_set=t, n_trials=2, seed=42)
   stage = sddp.active_stage

Now the model itself. ``precip`` is the inflow parameter; the sddp instance
overwrites it with the sampled value before each solve, so we leave it empty.

.. code-block:: python

   precip = gp.Parameter(m, "precip", description="net inflow (set each solve by sddp)")

   L = gp.Variable(m, "L", type="positive", domain=t, description="water level")
   R = gp.Variable(m, "R", type="positive", domain=t, description="controlled release")
   F = gp.Variable(m, "F", type="positive", domain=t, description="flood spill")
   Z = gp.Variable(m, "Z", type="positive", domain=t, description="imported water")
   cost = gp.Variable(m, "COST", description="stage cost")

   L.up[t] = L_FLOOD
   R.up[t] = R_MAX

Each stage-indexed equation is gated with ``.where[stage[t]]``. SDDP solves the
model one stage at a time, and the active-stage singleton activates only the
equations belonging to the stage currently being solved.

.. code-block:: python

   balance = gp.Equation(m, "balance", domain=t)
   obj = gp.Equation(m, "obj")

   balance[t].where[stage[t]] = (
       L[t] - L[t.lag(1, "circular")] + R[t] + F[t] - Z[t] == precip
   )
   obj[...] = cost == gp.Sum(stage[t], FLOOD_COST * F[t] + IMPORT_COST * Z[t])

.. note::
   The circular lag ``L[t.lag(1, "circular")]`` gives every stage a predecessor.
   For the very first stage there is no real predecessor, so the sddp instance
   substitutes the ``initial_state`` you register below.

Registering the state and the noise
===================================

We tell the sddp instance which variable is the state and how the noise is
distributed, then ``build()`` injects the SDDP machinery. The state's bounds
come from the variable itself here (``L.up`` is already the flood threshold);
passing ``upper_bound=L_FLOOD`` to ``add_state()`` would work just as well
(see :doc:`state_variables`). ``stage_cost`` is the per-stage cost variable,
*without* any future-cost term, which SDDP adds itself.

.. code-block:: python

   sddp.add_state(variable=L, initial_state=L0)
   sddp.set_noise(parameter=precip, scenario_data=scenario_data, probabilities=scenario_probs)
   sddp.build(stage_cost=cost)

Training the policy
===================

``train()`` runs the forward/backward iterations. ``n_iter`` caps the number of
iterations; ``rel_tol`` and ``patience`` stop early once the lower bound has
plateaued (here, improving by less than 0.1% for three iterations in a row).
``gap_paths`` runs an out-of-sample Monte Carlo of the trained policy at the end
to estimate the optimality gap.

.. code-block:: python

   result = sddp.train(n_iter=20, rel_tol=1e-3, patience=3, gap_paths=500)

Because the instance is verbose by default, training prints one row per
iteration and a summary:

.. code-block:: text

       j1:  bound =    1.074219E+2   sim cost =    9.375000E+2 ±  7.954951E+2
       j2:  bound =    1.074219E+2   sim cost =    1.718750E+2 ±  2.430680E+2
       j3:  bound =    1.113281E+2   sim cost =    2.500000E+2 ±  3.535534E+2
       j4:  bound =    1.116536E+2   sim cost =    0.000000E+0 ±  0.000000E+0
       j5:  bound =    1.123047E+2   sim cost =    0.000000E+0 ±  0.000000E+0
       j6:  bound =    1.123047E+2   sim cost =    0.000000E+0 ±  0.000000E+0
       j7:  bound =    1.123047E+2   sim cost =    0.000000E+0 ±  0.000000E+0
       j8:  bound =    1.123047E+2   sim cost =    0.000000E+0 ±  0.000000E+0

   ========================================================================
     Lower bound          :    1.123047E+2
     Iterations run       :              8
     Stop reason          :      converged
     Total time           :           9.63 s
     --------------------------------------------------------------------
     Policy cost          :    1.210000E+2 ±  3.626809E+1   (500 MC paths, 95% CI)
     Optimality gap       :        7.1862 %
   ========================================================================

The ``bound`` column is the **lower bound**: a rigorous under-estimate of the
optimal expected cost that rises as :doc:`cuts <how_it_works>` accumulate. It
climbs from 107.42 and settles at 112.3047, where the plateau rule stops
training after 8 of the 20 allowed iterations. The ``sim cost`` column is a small-sample (``n_trials``)
diagnostic of the forward paths, *not* a bound; it is noisy and can even be
zero, which is why training watches the bound, not this column.

The result object carries the same numbers:

.. code-block:: python

   result.lower_bound      # 112.3046875
   result.iterations_run   # 8
   result.stop_reason      # 'converged'

Querying the policy
===================

A trained policy answers operational questions. Suppose it is ``mar``, the
reservoir came in at level 150, and this month's inflow turned out to be a large
350. What should the operator do?

.. code-block:: python

   decision = sddp.policy(stage="mar", state=150, noise=350, report=[R, L, Z, F])

   decision.decisions          # {'R': 200.0, 'L': 250.0, 'Z': 0.0, 'F': 50.0}
   decision.approx_cost_to_go  # 625.0

With a high level and a large inflow the reservoir would overflow, so the policy
releases the maximum 200, spills the excess 50 in a controlled flood, imports
nothing, and ends the month at the 250 cap. The expected cost from ``mar`` to
the end of the season under this situation is 625.

Simulating the policy
=====================

``simulate()`` evaluates the policy on fresh Monte Carlo paths and returns the
realised cost distribution.

.. code-block:: python

   sim = sddp.simulate(n_paths=1000, seed=0)
   print(sim.summary)

.. code-block:: text

   n_paths    1000.000000
   mean        140.000000
   std         438.283119
   p5            0.000000
   p50           0.000000
   p95        1500.000000
   max        2500.000000
   Name: total_cost, dtype: float64

The distribution is strongly skewed: on most paths the policy keeps the cost at
zero (the median is 0), but a minority of high-inflow paths force expensive
flooding, lifting the mean to 140 and the 95th percentile to 1500. Shaping that
tail, rather than just the average, is what risk-averse training addresses.

Saving and reloading
====================

A trained policy can be serialised to a single ``.sddp`` file and reloaded in a
later session to run ``policy()`` and ``simulate()`` without retraining.

.. code-block:: python

   sddp.save("clearlake.sddp")

   # later, in a fresh session
   sddp = SDDP.load("clearlake.sddp")
   sddp.policy(stage="mar", state=150, noise=350, report=[R, L, Z, F])

.. seealso::
   The :doc:`introduction <introduction>` explains the concepts (stages,
   state, scenarios and the cost-to-go) behind every step above.
