.. _sddp_policy_and_simulation:

.. meta::
   :description: Using a trained SDDP policy in GAMSPy: policy() point queries and simulate() Monte Carlo
   :keywords: SDDP, policy, simulate, Monte Carlo, decisions, cost-to-go, SimulationResult, GAMSPy, gamspy

**********************
Using a Trained Policy
**********************

Training produces a policy: the collection of cuts. Two methods put it to work:
``policy()`` answers a single "what should I do here?" question, and
``simulate()`` evaluates the policy across many random paths.

Point queries with ``policy()``
===============================

``policy()`` solves one stage at a given situation: a stage, the incoming state,
and the realised noise.

.. code-block:: python

   decision = sddp.policy(stage="mar", state=150, noise=350, report=[R, L, Z, F])

   decision.decisions          # {'R': 200.0, 'L': 250.0, 'Z': 0.0, 'F': 50.0}
   decision.approx_cost_to_go  # 625.0

The returned ``PolicyResult`` carries the queried ``stage``, the
``incoming_state`` and ``noise``, the optimal ``decisions`` for the reported
variables, and ``approx_cost_to_go``, the immediate stage cost plus the
cut-approximated future cost from here on.

- ``state`` is a scalar for a single state variable, or a ``dict`` keyed by
  state-variable name when there are several.
- ``report`` lists the variables whose optimal level to return; it defaults to
  the state variables. A time-only variable reports a ``float``; a variable with
  extra dimensions reports a ``dict`` keyed by the non-time labels.

.. note::
   ``policy()`` needs a trained policy. Calling it before ``train()`` warns that
   no cuts exist yet and returns a decision that is *not* the trained policy.

Monte Carlo with ``simulate()``
===============================

``simulate()`` runs the policy forward on fresh sampled paths and returns the
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

The ``SimulationResult`` holds the per-path ``total_cost`` together with the
per-(path, stage) ``stage_costs``, ``noise`` and reported ``variables``; its
``summary`` reduces ``total_cost`` to the mean, standard deviation and
percentiles above. ``report`` defaults to the state variables, and ``seed``
defaults to the training seed plus one.

Both ``policy()`` and ``simulate()`` work on an instance reloaded from a
``.sddp`` file, so a saved policy can be queried without retraining.

.. seealso::
   The :doc:`ClearLake tutorial <clearlake>` runs both methods on a complete
   model; :doc:`risk` interprets the skewed cost distribution that
   ``simulate()`` reveals here.
