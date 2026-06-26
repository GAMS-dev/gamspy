.. _sddp_training:

.. meta::
   :description: Training an SDDP policy in GAMSPy: iterations, early stopping, the optimality gap
   :keywords: SDDP, train, convergence, early stopping, rel_tol, patience, gap_paths, optimality gap, GAMSPy, gamspy

************************
Training and Convergence
************************

``train()`` runs the SDDP forward/backward iterations and returns an
``SDDPResult`` holding the lower bound and the convergence history.

.. code-block:: python

   result = sddp.train(n_iter=20, rel_tol=1e-3, patience=3)

Iterations and early stopping
=============================

``n_iter`` is the hard cap on the number of iterations. On its own the run does
exactly that many and reports ``stop_reason == "max_iter"``.

The lower bound usually plateaus well before the cap, so ``rel_tol`` and
``patience`` add a stopping rule: training stops once the bound improves by less
than ``rel_tol`` (relative) for ``patience`` consecutive iterations, reporting
``stop_reason == "converged"``. On ClearLake, ``rel_tol=1e-3, patience=3`` stops
after 8 of the 20 allowed iterations. Leaving ``rel_tol=None`` (the default)
disables early stopping and runs all ``n_iter``.

To train a risk-averse policy, pass ``risk=``; see :doc:`risk`.

The result
==========

``SDDPResult`` carries the outcome of the run:

- ``lower_bound``: the rigorous lower bound at the final iteration.
- ``iterations_run``: how many iterations actually ran.
- ``stop_reason``: ``"converged"``, ``"max_iter"`` or ``"interrupted"``.
- ``convergence_table``: the per-iteration bounds.

``print(result)`` prints the summary box shown in the
:doc:`tutorial <clearlake>`. When the instance is verbose (the default),
training also prints one row per iteration as it goes.

Measuring the optimality gap
============================

The lower bound tells you how good the policy *could* be, not how good it *is*.
Passing ``gap_paths`` runs an out-of-sample Monte Carlo of the trained policy
after training and reports a rigorous gap:

.. code-block:: python

   result = sddp.train(n_iter=20, rel_tol=1e-3, patience=3, gap_paths=500)

.. code-block:: text

     Policy cost          :    1.210000E+2 ±  3.626809E+1   (500 MC paths, 95% CI)
     Optimality gap       :        7.1862 %

The policy's mean realised cost upper-bounds the true optimum, which the lower
bound bounds from below, so their difference is the optimality gap
(``result.optimality_gap_pct``), reported with the Monte Carlo confidence
interval (``policy_cost_mean`` ± ``policy_cost_stderr``). ``gap_paths=0`` (the
default) skips this entirely and is perf-neutral.

Interrupting training
=====================

Long runs can be stopped gracefully. Pressing :kbd:`Ctrl+C` once finishes the
current iteration, then returns the policy trained so far with
``stop_reason == "interrupted"``, and the cuts learned up to that point are intact
and usable. Pressing :kbd:`Ctrl+C` a second time aborts hard.

.. seealso::
   The :doc:`ClearLake tutorial <clearlake>` shows a full training run and its
   summary.
