.. _sddp_scenarios:

.. meta::
   :description: Scenarios and probabilities in GAMSPy SDDP: set_noise, scenario_data, probabilities
   :keywords: SDDP, scenarios, noise, probabilities, set_noise, stochastic, GAMSPy, gamspy, GAMS

***************************
Scenarios and Probabilities
***************************

The **noise** is the uncertain quantity revealed at each stage: the inflow in
ClearLake. SDDP works with a finite set of possible noise outcomes per stage,
the **scenarios**, each with a probability. You describe them with
``set_noise()``.

Describing the noise
====================

.. code-block:: python

   scenario_data = np.array([
       [ 50.0, 150.0, 350.0],   # jan
       [ 50.0, 150.0, 350.0],   # feb
       [-50.0, 100.0, 250.0],   # mar
       [-50.0, 100.0, 250.0],   # apr
   ])

   sddp.set_noise(parameter=precip, scenario_data=scenario_data)

``set_noise()`` takes:

- ``parameter``: the GAMSPy ``Parameter`` that carries the noise into your
  equations. You declare it empty; before each per-scenario solve the engine
  overwrites it with the value for the scenario being solved.
- ``scenario_data``: a 2-D array of shape ``(n_stages, n_scenarios)``: one row
  per stage, one column per scenario, holding the noise value for that
  combination. The ClearLake array above gives three inflow outcomes for each of
  the four months.

Probabilities
=============

By default the scenarios are equally likely. Pass ``probabilities`` to weight
them, a 1-D array of length ``n_scenarios`` that is non-negative and sums to 1:

.. code-block:: python

   sddp.set_noise(
       parameter=precip,
       scenario_data=scenario_data,
       probabilities=[0.25, 0.50, 0.25],
   )

The same probabilities apply at every stage. They weight the dual aggregation in
the backward pass and the path sampling in the forward pass, so they shape both
the cuts and the lower bound. Weighting scenarios unevenly is also the substrate
for :doc:`risk-averse training <risk>`.

.. note::
   ``set_noise()`` validates the inputs: ``scenario_data`` must be 2-D with one
   row per stage and at least one scenario column, and ``probabilities`` (when
   given) must be 1-D, match the number of scenarios, be non-negative, and sum
   to 1 within ``1e-9``. Omitting ``probabilities`` is equivalent to a uniform
   ``1 / n_scenarios`` for every scenario.

.. seealso::
   The :doc:`ClearLake tutorial <clearlake>` wires up the noise in context;
   :doc:`risk` uses uneven scenario weights to train against the costly tail.
